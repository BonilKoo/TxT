import argparse
import copy
import math
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

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
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--xavier_uniform', action='store_true')
    parser.add_argument('--noam', action='store_true')
    parser.add_argument('--noam_factor', type=int, default=2)
    parser.add_argument('--noam_warmup', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=100)
    
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

class CustomDataset(Dataset):
    def __init__(self, input_file, output_file):
        input_df = pd.read_csv(input_file, index_col=0)
        print(f'\nInput file {input_file} is loaded.')
        output_df = pd.read_table(output_file, index_col=0)
        print(f'Output file {output_file} is loaded.')
        df = pd.merge(input_df, output_df, left_index=True, right_index=True)
        print(f'\n# of samples = {len(df)}')
        print(f'# of genes = {df.shape[1] - output_df.shape[1]}\n')
        
        self.gene_list = input_df.columns.to_list()
        
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values
        
        self.length = len(df)
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = torch.FloatTensor([self.y[index]])
        return x, y
    
    def __len__(self):
        return self.length

def load_dataset(input_file, output_file, val_ratio, test_ratio, scaler, batch_size):
    dataset = CustomDataset(input_file, output_file)
    
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
    def __init__(self, n_genes, d_embed, dropout=0.1, aggfunc='Flatten', d_hidden1=128, d_hidden2=64, slope=0.2):
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
        
        self.linear_3 = nn.Linear(d_hidden2, 1)
    
    def forward(self, x):
        x = self.dropout_1(self.activation_1(self.batch_norm_1(self.linear_1(x))))
        x = self.dropout_2(self.activation_2(self.batch_norm_2(self.linear_2(x))))
        x = self.linear_3(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, embed_file, gene_list, device,
                 n_heads=8, d_model=512, dropout=0.1, d_ff=2048, norm_first=False, n_layers=6,
                 aggfunc='Flatten', d_hidden1=128, d_hidden2=64, slope=0.2):
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
                                                     aggfunc, d_hidden1, d_hidden2, slope).to(device)
    
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

class EarlyStopping:
    """
    Code based on: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if self.path is not None:
            torch.save(model.state_dict(), self.path)
        
        self.val_loss_min = val_loss

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
    
    def step(self):
        "Update parameters and rate."
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        "Implement `lrate` above."
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        "Reset gradient."
        self.optimizer.zero_grad()

def get_std_opt(d_model, factor, warmup, model):
    return NoamOpt(d_model, factor, warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def eval_result(model, dataloader, device):
    y_true = np.empty(0)
    y_pred = torch.empty((0,1))
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            y_true = np.append(y_true, y)
            x = x.to(device)
            
            output = model(x)
            
            y_pred = torch.cat((y_pred, output.detach().cpu()))
    
    y_true = torch.Tensor(y_true)
    y_pred = y_pred.squeeze(-1)
    
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    PCC = np.corrcoef(y_true, y_pred)[0][1]
    SCC = spearmanr(y_true, y_pred)[0]
    
    return MAE, RMSE, PCC, SCC

def print_save_result(model, dataloader, device, log, dataset_type):
    MAE, RMSE, PCC, SCC = eval_result(model, dataloader, device)
    log.write(f'[{dataset_type:^10s}] MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SCC: {SCC:.4f}\n')
    print(f'[{dataset_type:^10s}] MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SCC: {SCC:.4f}')

def run(args):
    set_seed(args.seed)
    
    train_dataloader, val_dataloader, test_dataloader, gene_list = load_dataset(args.input_file, args.output_file, args.val_ratio, args.test_ratio, args.scaler, args.batch_size)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    model = CustomModel(args.embed_file, gene_list, device,
                        args.n_heads, args.d_model, args.dropout, args.d_ff, args.norm_first, args.n_layers,
                        args.aggfunc, args.d_hidden1, args.d_hidden2, args.slope).to(device)
    if args.xavier_uniform:
        for name, p in model.named_parameters():
            if ('embed' not in name) & (p.dim() > 1):
                nn.init.xavier_uniform_(p)
    
    criterion = nn.MSELoss()
    
    if args.noam:
        optimizer = get_std_opt(args.d_model, args.noam_factor, args.noam_warmup, model)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    def train():
        model.train()
        total_loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        return total_loss / len(train_dataloader.dataset)
    
    @torch.no_grad()
    def test():
        model.eval()
        total_loss = 0
        #for x, y in val_dataloader:
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            loss = criterion(output, y)
            
            total_loss += loss.item() * x.size(0)
        #return total_loss / len(val_dataloader.dataset)
        return total_loss / len(test_dataloader.dataset)
    
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
    
    print_save_result(model, train_dataloader, device, f, 'Training')
    if args.val_ratio > 0:
        print_save_result(model, val_dataloader, device, f, 'Validation')
    print_save_result(model, test_dataloader, device, f, 'Test')    
    f.close()

def main():
    args = parse_args()
    save_args(args)
    run(args)

if __name__ == '__main__':
    main()
