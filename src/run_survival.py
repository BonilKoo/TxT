import argparse
import copy
import math
import os
import random

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sksurv.metrics import concordance_index_censored, integrated_brier_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

from datasets import SurvivalDataset
from evaluation import eval_result_survival
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--embed_file', required=True)
    parser.add_argument('--result_dir', required=True)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--scaler', choices=[None, 'MinMax', 'Standard'], default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_time_intervals', type=int, default=64) # survival
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--xavier_uniform', action='store_true')
    parser.add_argument('--noam', action='store_true')
    parser.add_argument('--noam_factor', type=int, default=2)
    parser.add_argument('--noam_warmup', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--max_epoch', type=int, default=1000)
    
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--n_layers', type=int, default=2)
    
    parser.add_argument('--aggfunc', choices=['Flatten', 'Avgpool'], default='Flatten')
    parser.add_argument('--d_hidden1', type=int, default=128)
    parser.add_argument('--d_hidden2', type=int, default=64)
    parser.add_argument('--slope', type=float, default=0.2)
    
    return parser.parse_args()

# def load_dataset(input_file, output_file, val_ratio, test_ratio, scaler, batch_size): # regression & classification
def load_dataset(input_file, output_file, val_ratio, test_ratio, scaler, batch_size, n_time_intervals): # survival
#     dataset = CustomDataset(input_file, output_file) # regression & classification
    dataset = SurvivalDataset(input_file, output_file, n_time_intervals) # survival
    
    dataset_size = len(dataset)
    train_ratio = 1 - val_ratio - test_ratio
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    elif scaler == 'Standard':
        scaler = StandardScaler()
    
    if scaler is not None:
        train_X_scaled = scaler.fit_transform(dataset.x[train_dataset.indices])
        val_X_scaled = scaler.transform(dataset.x[val_dataset.indices])
        test_X_scaled = scaler.transform(dataset.x[test_dataset.indices])

        dataset.x[train_dataset.indices] = train_X_scaled
        dataset.x[val_dataset.indices] = val_X_scaled
        dataset.x[test_dataset.indices] = test_X_scaled
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    gene_list = dataset.gene_list
    
    return train_dataloader, val_dataloader, test_dataloader, gene_list

class Embedder(nn.Module):
    def __init__(self, embed_file, gene_list):
        super(Embedder, self).__init__()
        
        embed_df = pd.read_csv(embed_file, index_col=0)
        embed_df = embed_df.loc[gene_list]
        
        self.embed = nn.Embedding.from_pretrained(torch.Tensor(embed_df.values), freeze=False)
    
    def forward(self, e):
        # e: (batch_size, n_genes)
        return self.embed(e) # (batch_size, n_genes, d_embed)

class TUPE_A(nn.Module):
    def __init__(self, n_heads, d_model):
        super(TUPE_A, self).__init__()
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head * n_heads == d_model, '"d_model" must be divisible by "n_heads".'
        
        self.Q_linear = nn.Linear(1, d_model, bias=False)
        self.K_linear = nn.Linear(1, d_model, bias=False)
    
    def forward(self, x):
        # x: (batch_size, n_genes)
        batch_size = x.size(0)
        
        x = x.unsqueeze(-1)
        # x: (batch_size, n_genes, 1)
        
        Q = self.Q_linear(x).view(batch_size, -1, self.n_heads, self.d_head)
        K = self.K_linear(x).view(batch_size, -1, self.n_heads, self.d_head)
        # Q,K: (batch_size, n_genes, d_model) => (batch_size, n_genes, n_heads, d_head)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        # Q,K: (batch_size, n_heads, n_genes, d_head)
        
        untied_PE = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(2 * self.d_head)
        # untied_PE: (batch_size, n_heads, n_genes, n_genes)
        
        return untied_PE

def calculate_attention(Q, K, V, TUPE, mask=None, dropout=None):
    # Q,K,V: (batch_size, n_heads, n_genes, d_head)
    # TUPE: (batch_size, n_heads, n_genes, n_genes)
    d_head = Q.size(-1)
    
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(2 * d_head) + TUPE
    # attention_scores: (batch_size, n_heads, n_genes, n_genes)
    
    if mask is not None:
        # mask: (n_genes, n_genes)
        # Fill "attention_scores" with small number (-1e9) where mask is 0
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
    attention_scores = F.softmax(attention_scores, dim=-1)
    # attention_scores: (batch_size, n_heads, n_genes, n_genes)
    
    if dropout is not None:
        attention_scores = dropout(attention_scores)
    
    output = torch.matmul(attention_scores, V)
    # output: (batch_size, n_heads, n_genes, d_head)
    
    return output, attention_scores

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=512, d_embed=512, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        assert d_model % n_heads == 0, '"d_model" must be divisible by "n_heads".'
        self.d_head = d_model // n_heads
        
        self.Q_linear = nn.Linear(d_embed, d_model, bias=False)
        self.K_linear = nn.Linear(d_embed, d_model, bias=False)
        self.V_linear = nn.Linear(d_embed, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.attention_weights = None
        
        self.linear = nn.Linear(d_model, d_embed, bias=False)
    
    def forward(self, Q, K, V, TUPE, mask=None):
        # Q,K,V: (batch_size, n_genes, d_embed)
        # TUPE: (batch_size, n_heads, n_genes, n_genes)
        batch_size = Q.size(0)
        
        Q = self.Q_linear(Q).view(batch_size, -1, self.n_heads, self.d_head)
        K = self.K_linear(K).view(batch_size, -1, self.n_heads, self.d_head)
        V = self.V_linear(V).view(batch_size, -1, self.n_heads, self.d_head)
        # Q,K,V: (batch_size, n_genes, d_model) => (batch_size, n_genes, n_heads, d_head)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # Q,K,V: (batch_size, n_heads, n_genes, d_head)
        
        attention_scores, self.attention_weights = calculate_attention(Q, K, V, TUPE, mask, self.dropout)
        # attention_scores: (batch_size, n_heads, n_genes, d_head)
        
        concat = attention_scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # concat: (batch_size, n_genes, n_heads, d_head) => (batch_size, n_genes, d_model)
        
        output = self.linear(concat)
        # output: (batch_size, n_genes, d_embed)
        
        return output

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed=512, d_ff=2048, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        
        self.linear_1 = nn.Linear(d_embed, d_ff, bias=False)
        self.linear_2 = nn.Linear(d_ff, d_embed, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, e):
        # e: (batch_size, n_genes, d_embed)
        
        e = self.linear_2(self.dropout(F.relu(self.linear_1(e))))
        # e: (batch_size, n_genes, d_ff) => (batch_size, n_genes, d_embed)
        
        return e

class EncoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=512, d_embed=512, dropout=0.1, d_ff=2048, norm_first=False):
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention_layer = MultiHeadAttentionLayer(n_heads, d_model, d_embed, dropout)
        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
        
        self.layer_norm_1 = nn.LayerNorm(d_embed)
        self.layer_norm_2 = nn.LayerNorm(d_embed)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        self.norm_first = norm_first
    
    def forward(self, e, TUPE, mask=None):
        # e: (batch_size, n_genes, d_embed)
        # TUPE: (batch_size, n_heads, n_genes, n_genes)
        
        if self.norm_first:
            e2 = self.layer_norm_1(e)
            e = e + self.dropout_1(self.multi_head_attention_layer(e2, e2, e2, TUPE, mask))
            
            e2 = self.layer_norm_2(e)
            e = e + self.dropout_2(self.position_wise_feed_forward_layer(e2))
        
        else:
            e = self.layer_norm_1(e + self.dropout_1(self.multi_head_attention_layer(e, e, e, TUPE, mask)))
            e = self.layer_norm_2(e + self.dropout_2(self.position_wise_feed_forward_layer(e)))
        
        # e: (batch_size, n_genes, d_embed)
        return e

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, embed_file, gene_list, device,
                 n_heads=8, d_model=512, dropout=0.1, d_ff=2048, norm_first=False, n_layers=6):
        super(Encoder, self).__init__()
        
        self.embed = Embedder(embed_file, gene_list)
        self.d_embed = self.embed.embed.embedding_dim
        
        self.device = device
        
        self.TUPE_A = TUPE_A(n_heads, d_model)
        
        self.layers = get_clones(EncoderLayer(n_heads, d_model, self.d_embed, dropout, d_ff, norm_first), n_layers)
        
        self.layer_norm = nn.LayerNorm(self.d_embed)
    
    def forward(self, x, mask=None):
        # x: (batch_size, n_genes)
        batch_size = x.size(0)
        n_genes = x.size(1)
        
        e = torch.arange(n_genes, device=self.device).repeat(batch_size).view(batch_size, -1)
        # e: (n_genes) => (n_genes * batch_size) => (batch_size, n_genes)
        
        e = self.embed(e)
        # e: (batch_size, n_genes, d_embed)
        
        TUPE = self.TUPE_A(x)
        # TUPE: (batch_size, n_heads, n_genes, n_genes)
        
        for layer in self.layers:
            e = layer(e, TUPE, mask)
        # e: (batch_size, n_genes, d_embed)
        
        return self.layer_norm(e)

class Transformer(nn.Module):
    def __init__(self, embed_file, gene_list, device,
                 n_heads=8, d_model=512, dropout=0.1, d_ff=2048, norm_first=False, n_layers=6):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(embed_file, gene_list, device,
                               n_heads, d_model, dropout, d_ff, norm_first, n_layers)
    
    def encode(self, x, mask=None):
        # x: (batch_size, n_genes)
        
        e_outputs = self.encoder(x, mask)
        # e_outputs: (batch_size, n_genes, d_embed)
        
        return e_outputs
    
    def forward(self, x, mask=None):
        # x: (batch_size, n_genes)
        
        e_outputs = self.encode(x, mask)
        # e_outputs: (batch_size, n_genes, d_embed)
        
        return e_outputs

class TaskSpecificLayer(nn.Module):
#     def __init__(self, n_genes, d_embed, dropout=0.1, aggfunc='Flatten', d_hidden1=128, d_hidden2=64, slope=0.2): # regression
    def __init__(self, n_genes, d_embed, dropout=0.1, aggfunc='Flatten', d_hidden1=128, d_hidden2=64, slope=0.2, d_output=1): # classification & survival
        super(TaskSpecificLayer, self).__init__()
        
        if aggfunc == 'Flatten':
            self.linear_1 = nn.Linear(n_genes*d_embed, d_hidden1) # (1)
        else:
            self.linear_1 = nn.Linear(n_genes, d_hidden1) # (2)
        
        self.batch_norm_1 = nn.BatchNorm1d(d_hidden1)
        self.activation_1 = nn.LeakyReLU(slope, inplace=True)
        self.dropout_1 = nn.Dropout(dropout)
        
        self.linear_2 = nn.Linear(d_hidden1, d_hidden2)
        self.batch_norm_2 = nn.BatchNorm1d(d_hidden2)
        self.activation_2 = nn.LeakyReLU(slope, inplace=True)
        self.dropout_2 = nn.Dropout(dropout)
        
#         self.linear_3 = nn.Linear(d_hidden2, 1) # regression
        self.linear_3 = nn.Linear(d_hidden2, d_output) # classification & survival
    
    def forward(self, x):
        x = self.dropout_1(self.activation_1(self.batch_norm_1(self.linear_1(x))))
        x = self.dropout_2(self.activation_2(self.batch_norm_2(self.linear_2(x))))
        x = self.linear_3(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, embed_file, gene_list, device,
                 n_heads=8, d_model=512, dropout=0.1, d_ff=2048, norm_first=False, n_layers=6,
#                  aggfunc='Flatten', d_hidden1=128, d_hidden2=64, slope=0.2): # regression
                 aggfunc='Flatten', d_hidden1=128, d_hidden2=64, slope=0.2, d_output=1): # classification & survival
        super(CustomModel, self).__init__()
        
        self.transformer = Transformer(embed_file, gene_list, device,
                                       n_heads, d_model, dropout, d_ff, norm_first, n_layers)
        
        self.d_embed = self.transformer.encoder.d_embed
        
        self.aggfunc = aggfunc
        if aggfunc == 'Flatten':
            self.flatten = nn.Flatten(start_dim=1) # (1)
            self.dropout = nn.Dropout(dropout) # (1)
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1) # (2)
        
        self.task_specific_layer = TaskSpecificLayer(len(gene_list), self.d_embed, dropout,
#                                                      aggfunc, d_hidden1, d_hidden2, slope).to(device) # regression
                                                     aggfunc, d_hidden1, d_hidden2, slope, d_output).to(device) # classification & survival
    
    def forward(self, x, mask=None):
        # x: (batch_size, n_genes)
        
        x = self.transformer(x, mask)
        # x: (batch_size, n_genes, d_embed)
        
        if self.aggfunc == 'Flatten':
            x = self.dropout(self.flatten(x)) # (1)
            # x: (batch_size, n_genes*d_embed)
        else:
            x = self.pooling(x).squeeze(-1) # (2)
        # x: (batch_size, n_genes)
        
        output = self.task_specific_layer(x)
        
        return output

def SurvivalLoss(input, target, E, Triangle, reduction='mean'): # survival
    score_cens = input[E == 0]
    score_uncens = input[E == 1]
    target_cens = target[E == 0]
    target_uncens = target[E == 1]
    
    phi_uncens = torch.exp(torch.mm(score_uncens, Triangle))
    reduc_phi_uncens = torch.sum(phi_uncens * target_uncens, dim=1)
    
    phi_cens = torch.exp(torch.mm(score_cens, Triangle))
    reduc_phi_cens = torch.sum(phi_cens * target_cens, dim=1)
    
    z_uncens = torch.exp(torch.mm(score_uncens, Triangle))
    reduc_z_uncens = torch.sum(z_uncens, dim=1)
    
    z_cens = torch.exp(torch.mm(score_cens, Triangle))
    reduc_z_cens = torch.sum(z_cens, dim=1)
    
    loss = - (
                torch.sum(torch.log(reduc_phi_uncens)) \
                + torch.sum(torch.log(reduc_phi_cens)) \
                - torch.sum(torch.log(reduc_z_uncens)) \
                - torch.sum(torch.log(reduc_z_cens))
            )
    
    if reduction == 'mean':
        loss = loss / E.shape[0]
    
    return loss

# def print_save_result(model, dataloader, device, log, dataset_type): # regression
# def print_save_result(model, dataloader, device, log, dataset_type, n_classes): # classification
def print_save_result(model, dataloader, device, log, dataset_type, num_times, times): # survival
#     MAE, RMSE, PCC = eval_result_regression(model, dataloader, device) # regression
#     log.write(f'[{dataset_type}] - MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}\n') # regression
#     print(f'[{dataset_type}] - MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}') # regression

#     accuracy, precision, recall, f1, auroc = eval_result_classification(model, dataloader, device, n_classes) # classification
#     log.write(f'[{dataset_type:^10s}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}\n') # classification
#     print(f'[{dataset_type:^10s}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}') # classification
    
    C_Index, IBS = eval_result_survival(model, dataloader, device, num_times, times) # survival
    log.write(f'[{dataset_type:^10s}] C-Index: {C_Index:.4f}, IBS: {IBS:.4f}\n') # survival
    print(f'[{dataset_type:^10s}] C-Index: {C_Index:.4f}, IBS: {IBS:.4f}') # survival

def run(args):
    set_seed(args.seed)
    
#     train_dataloader, val_dataloader, test_dataloader, gene_list = load_dataset(input_file, output_file, val_ratio, test_ratio, scaler, batch_size) # regression & classification
    train_dataloader, val_dataloader, test_dataloader, gene_list = load_dataset(args.input_file, args.output_file, args.val_ratio, args.test_ratio, args.scaler, args.batch_size, args.n_time_intervals) # survival
#     n_classes = train_dataloader.dataset.dataset.n_classes # classification
    num_times = train_dataloader.dataset.dataset.num_times # survival
    time_buckets = train_dataloader.dataset.dataset.time_buckets # survival
    times = train_dataloader.dataset.dataset.times # survival
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    model = CustomModel(args.embed_file, gene_list, device,
#                         n_heads, d_model, dropout, d_ff, norm_first, n_layers).to(device) # regression
                        args.n_heads, args.d_model, args.dropout, args.d_ff, args.norm_first, args.n_layers, # classification & survival
#                         args.aggfunc, args.d_hidden1, args.d_hidden2, args.slope, n_classes).to(device) # classification
                        args.aggfunc, args.d_hidden1, args.d_hidden2, args.slope, num_times).to(device) # survival
    if args.xavier_uniform:
        for name, p in model.named_parameters():
            if ('embed' not in name) & (p.dim() > 1):
                nn.init.xavier_uniform_(p)
    
#     criterion = nn.MSELoss() # regression
#     criterion = nn.CrossEntropyLoss() # classification
    
    if args.noam:
        optimizer = get_std_opt(args.d_model, args.noam_factor, args.noam_warmup, model)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    Triangle = torch.FloatTensor(np.tri(num_times, num_times + 1, dtype=np.float32)).to(device) # survival
    
    def train():
        model.train()
        total_loss = 0
#         for x, y in train_dataloader: # regression & classification
        for x, y, T, E in train_dataloader: # survival
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
#             loss = criterion(output, y) # regression & classification
            loss = SurvivalLoss(output, y, E, Triangle) # survival
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        return total_loss / len(train_dataloader.dataset)
    
    @torch.no_grad()
    def test():
        model.eval()
        total_loss = 0
#         for x, y in val_dataloader: # regression & classification
        for x, y, T, E in val_dataloader: # survival
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
#             loss = criterion(output, y) # regression & classification
            loss = SurvivalLoss(output, y, E, Triangle) # survival
            
            total_loss += loss.item() * x.size(0)
        return total_loss / len(val_dataloader.dataset)
    
    model_file = f'{args.result_dir}/model.pt'
    log_file = f'{args.result_dir}/log.txt'
    f = open(log_file, 'w')
    early_stopping = EarlyStopping(patience=args.patience, path=model_file)

    for epoch in range(1, args.max_epoch+1):
        train_loss = train()
        
        if args.val_ratio > 0:
            val_loss = test()
            print(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            f.write(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
        else:
            print(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}')
            f.write(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}\n')
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f'Stop training because the training loss or validation loss is nan.')
            f.write(f'Stop training because the training loss or validation loss is nan.\n')
            break
    
    print(f'\nLog file is saved to {log_file}.')
    print(f'Model is saved to {model_file}.\n')
    f.write('\n')
    model = load_best_model(model, model_file, device)
    
    # regression
#     print_save_result(model, train_dataloader, device, f, 'Training')
#     print_save_result(model, val_dataloader, device, f, 'Validation')
#     print_save_result(model, test_dataloader, device, f, 'Test')
    
    # classification
#     print_save_result(model, train_dataloader, device, f, 'Training', n_classes)
#     print_save_result(model, val_dataloader, device, f, 'Validation', n_classes)
#     print_save_result(model, test_dataloader, device, f, 'Test', n_classes)
    
    # survival
    print_save_result(model, train_dataloader, device, f, 'Training', num_times, times)
    if args.val_ratio > 0:
        print_save_result(model, val_dataloader, device, f, 'Validation', num_times, times)
    print_save_result(model, test_dataloader, device, f, 'Test', num_times, times)
    
    f.close()

def main():
    args = parse_args()
    save_args(args)    
    run(args)

if __name__ == '__main__':
    main()
