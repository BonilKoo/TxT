import argparse
from collections import OrderedDict
import copy
from itertools import accumulate # PCGrad
import math
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sksurv.metrics import concordance_index_censored, integrated_brier_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--task_file', required=True)
    parser.add_argument('--embed_file', required=True)
    parser.add_argument('--result_dir', required=True)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--scaler', choices=[None, 'MinMax', 'Standard'], default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_time_intervals', type=int, default=64)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--xavier_uniform', action='store_true')
    parser.add_argument('--noam', action='store_true')
    parser.add_argument('--noam_factor', type=int, default=2)
    parser.add_argument('--noam_warmup', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=100)

    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--n_layers', type=int, default=6)

    parser.add_argument('--aggfunc', choices=['Flatten', 'Avgpool'], default='Flatten')
    parser.add_argument('--d_hidden1', type=int, default=128)
    parser.add_argument('--d_hidden2', type=int, default=64)
    parser.add_argument('--slope', type=float, default=0.2)

    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, input_file, output_file, task_file, n_time_intervals, is_min_time_zero=True, extra_pct_time=0.1):
        input_df = pd.read_csv(input_file, index_col=0)
        print(f'\nInput file {input_file} is loaded.')
        output_df = pd.read_table(output_file, index_col=0)
        print(f'Output file {output_file} is loaded.')

        task_df = pd.read_table(task_file)
        print(f'Task file {task_file} is loaded.')
        task_df_regression = task_df[task_df['task'] == 'regression']
        task_df_classification = task_df[task_df['task'] == 'classification']
        task_df_survival = task_df[task_df['task'].str.contains('survival')]
        task_df = pd.concat([task_df_regression, task_df_classification, task_df_survival], ignore_index=True)
        output_df = output_df[task_df['name']]

        df = pd.merge(input_df, output_df, left_index=True, right_index=True)
        print(f'\n# of samples = {len(df)}')
        print(f'# of genes = {df.shape[1] - output_df.shape[1]}')


        if ('survival_time' in task_df['task'].to_list()) and ('survival_event' in task_df['task'].to_list()):
            self.flag_survival = 1
        elif ('survival_time' not in task_df['task'].to_list()) and ('survival_event' not in task_df['task'].to_list()):
            self.flag_survival = 0
        else:
            raise Exception('"survival_time" and "survival_event" must be included together.')
        self.n_tasks = task_df.shape[0] - self.flag_survival
        print(f'# of tasks = {self.n_tasks}\n')

        self.name_task_dict = OrderedDict({name: task for name, task in task_df.values})

        self.task_name_dict = OrderedDict()
        for name, task in task_df.values:
            if task in self.task_name_dict.keys():
                self.task_name_dict[task].append(name)
            else:
                if 'survival' in task:
                    self.task_name_dict[task] = name
                else:
                    self.task_name_dict[task] = [name]
        
        # PCGrad
        if 'regression' in self.task_name_dict.keys():
            flag_regression = True
        else:
            flag_regression = False
        if 'classification' in self.task_name_dict.keys():
            flag_classification = True
        else:
            flag_classification = False
        
        self.idx_task_dict = OrderedDict()
        if flag_regression:
            for i in range(len(self.task_name_dict['regression'])):
                self.idx_task_dict[len(self.idx_task_dict)] = 'regression'
        if flag_classification:
            for i in range(len(self.task_name_dict['classification'])):
                self.idx_task_dict[len(self.idx_task_dict)] = 'classification'
        if self.flag_survival:
            self.idx_task_dict[len(self.idx_task_dict)] = 'survival'
        # PCGrad

        self.gene_list = input_df.columns.to_list()

        self.x = df.iloc[:, :-output_df.shape[1]].values

        self.y_df = df.iloc[:, -output_df.shape[1]:]
        self.classification_name_label_dict = OrderedDict()
        self.d_output_dict = OrderedDict()
        for task in self.task_name_dict.keys():
            if task == 'regression':
                for name in self.task_name_dict[task]:
                    self.d_output_dict[name] = 1
            elif task == 'classification':
                for name in self.task_name_dict[task]:
                    self.classification_name_label_dict[name] = {label: idx for idx, label in enumerate(np.unique(self.y_df[name]))}
                    self.tmp_dict = self.classification_name_label_dict[name]
                    self.y_df[name] = list(map(self.label_to_vector, self.y_df[name]))
                    del self.tmp_dict
                    self.d_output_dict[name] = len(self.classification_name_label_dict[name])
            elif task == 'survival_time':
                self.T = self.y_df[self.task_name_dict['survival_time']].values
                self.E = self.y_df[self.task_name_dict['survival_event']].values
                self.y_df.drop(columns=[self.task_name_dict['survival_time'], self.task_name_dict['survival_event']], inplace=True)
                self.n_time_intervals = n_time_intervals
                self.y_survival = self.compute_Y_survival(self.T, self.E, is_min_time_zero, extra_pct_time)
                self.d_output_dict['survival_time'] = self.num_times
            else:
                continue

        self.length = len(df)

    def label_to_vector(self, value):
        return self.tmp_dict.get(value, None)

    def get_time_buckets(self): # survival
        return [(self.times[i], self.times[i+1]) for i in range(len(self.times) - 1)]

    def get_times(self, T, is_min_time_zero, extra_pct_time): # survival
        max_time = max(T)
        if is_min_time_zero:
            min_time = 0
        else:
            min_time = min(T)

        if 0 <= extra_pct_time <= 1:
            p = extra_pct_time
        else:
            raise Exception('"extra_pct_time" has to be between [0,1].')

        self.times = np.linspace(min_time, max_time * (1 + p), self.n_time_intervals)
        self.time_buckets = self.get_time_buckets()
        self.num_times = len(self.time_buckets)

    def compute_Y_survival(self, T, E, is_min_time_zero, extra_pct_time): # survival
        self.get_times(T, is_min_time_zero, extra_pct_time)

        Y = []

        for t, e in zip(T, E):
            y = np.zeros(self.num_times + 1)
            min_abs_value = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_abs_value)

            if e == 1:
                y[index] = 1
                Y.append(y.tolist())
            else:
                y[index:] = 1
                Y.append(y.tolist())

        return torch.FloatTensor(Y)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y_list = []
        for task in self.task_name_dict.keys():
            if task == 'regression':
                for name in self.task_name_dict[task]:
                    y_list.append(torch.FloatTensor([self.y_df[name].values[index]]))
            elif task == 'classification':
                for name in self.task_name_dict[task]:
                    y_list.append(torch.LongTensor(self.y_df[name].values)[index])
        if self.flag_survival:
            y_list.append(torch.FloatTensor(self.y_survival[index]))
            T = torch.FloatTensor(self.T)[index]
            E = torch.FloatTensor(self.E)[index]
        else:
            T = []
            E = []
        return x, *y_list, T, E

    def __len__(self):
        return self.length

def load_dataset(input_file, output_file, task_file, val_ratio, test_ratio, scaler, batch_size, n_time_intervals):
    dataset = CustomDataset(input_file, output_file, task_file, n_time_intervals)

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
    def __init__(self, n_genes, d_embed, dropout=0.1, aggfunc='Flatten', d_hidden1=128, d_hidden2=64, slope=0.2, d_output=1):
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

        self.linear_3 = nn.Linear(d_hidden2, d_output)

    def forward(self, x):
        x = self.dropout_1(self.activation_1(self.batch_norm_1(self.linear_1(x))))
        x = self.dropout_2(self.activation_2(self.batch_norm_2(self.linear_2(x))))
        x = self.linear_3(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, embed_file, gene_list, device,
                 n_heads=8, d_model=512, dropout=0.1, d_ff=2048, norm_first=False, n_layers=6,
                 aggfunc='Flatten', d_hidden1=128, d_hidden2=64, slope=0.2, d_output_dict=None):
        super(CustomModel, self).__init__()

        self.transformer = Transformer(embed_file, gene_list, device,
                                       n_heads, d_model, dropout, d_ff, norm_first, n_layers)

        self.d_embed = self.transformer.encoder.d_embed

        self.aggfunc = aggfunc
        if aggfunc == 'Flatten':
            self.flatten = nn.Flatten(start_dim=1)
            self.dropout = nn.Dropout(dropout)
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1)

        self.task_specific_layers = nn.ModuleList([TaskSpecificLayer(len(gene_list), self.d_embed, dropout,
                                                                     aggfunc, d_hidden1, d_hidden2, slope,
                                                                     d_output) for d_output in d_output_dict.values()])

    def forward(self, x, mask=None):
        # x: (batch_size, n_genes)

        x = self.transformer(x, mask)
        # x: (batch_size, n_genes, d_embed)

        if self.aggfunc == 'Flatten':
            x = self.dropout(self.flatten(x))
            # x: (batch_size, n_genes*d_embed)
        else:
            x = self.pooling(x).squeeze(-1)
        # x: (batch_size, n_genes)

        outputs = [layer(x) for layer in self.task_specific_layers]

        return outputs

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
        self.param_groups = self.optimizer.param_groups # PCGrad

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

def PCGrad_backward(optimizer, outputs, y_list, n_tasks, idx_task_dict, flag_survival, E, Triangle, device): # PCGrad
    '''Code based on: https://github.com/wgchang/PCGrad-pytorch-example/blob/master/pcgrad-example.py'''
    criterion_regression = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()
    
    grads_task = []
    grad_shapes = [p.shape if p.requires_grad is True else None
                   for group in optimizer.param_groups for p in group['params']]
    grad_numel = [p.numel() if p.requires_grad is True else 0
                  for group in optimizer.param_groups for p in group['params']] # total number of elements
    
    loss_list = []
    optimizer.zero_grad()
    
    # calculate gradients for each task
    for idx in range(n_tasks):
        if idx_task_dict[idx] == 'regression':
            loss = criterion_regression(outputs[idx], y_list[idx])
        elif idx_task_dict[idx] == 'classification':
            loss = criterion_classification(outputs[idx], y_list[idx])
        else:
            loss = SurvivalLoss(outputs[idx], y_list[idx], E, Triangle)
        loss_list.append(loss)
        # 1
        loss.backward(retain_graph=True)

        grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                else None for group in optimizer.param_groups for p in group['params']]

        # fill zero grad if grad is None but requires_grad is true
        grads_task.append(torch.cat([g if g is not None else torch.zeros(
            grad_numel[i], device=device) for i, g in enumerate(grad)]))
        optimizer.zero_grad()
        
    # shuffle gradient order
    # 3 & 4
    random.shuffle(grads_task)

    # gradient projection
    grads_task = torch.stack(grads_task, dim=0)  # (T, # of params)
#     grads_task = torch.stack(grads_task, dim=0).detach().cpu()  # (T, # of params)
    # 2
    proj_grad = grads_task.clone()

    # 5 & 6 & 7
    def _proj_grad(grad_task):
        for k in range(n_tasks):
            inner_product = torch.sum(grad_task * grads_task[k])
            proj_direction = inner_product / (torch.sum(grads_task[k] * grads_task[k]) + 1e-12)
            grad_task = grad_task - torch.min(proj_direction, torch.zeros_like(proj_direction)) * grads_task[k]
        return grad_task

    proj_grad = torch.sum(torch.stack(list(map(_proj_grad, list(proj_grad)))), dim=0)  # (of params, )

    indices = [0, ] + [v for v in accumulate(grad_numel)]
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len(grad_shapes) == len(indices[:-1])
    for param, grad_shape, start_idx, end_idx in zip(params, grad_shapes, indices[:-1], indices[1:]):
        if grad_shape is not None:
            param.grad[...] = proj_grad[start_idx:end_idx].view(grad_shape)  # copy proj grad

    return loss_list

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

def criterion(outputs, y_list, E, Triangle, task_name_dict, flag_survival): # naive (sum)
    criterion_regression = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()
    loss_list = []
    idx = 0
    if 'regression' in task_name_dict.keys():
        for name in task_name_dict['regression']:
            loss_list.append(criterion_regression(outputs[idx], y_list[idx]))
            idx += 1
    if 'classification' in task_name_dict.keys():
        for name in task_name_dict['classification']:
            loss_list.append(criterion_classification(outputs[idx], y_list[idx]))
            idx += 1
    if flag_survival:
        loss_list.append(SurvivalLoss(outputs[idx], y_list[idx], E, Triangle))

    return sum(loss_list)

def eval_result_regression(model, dataloader, device, index): # regression
    y_true = np.empty(0)
    y_pred = torch.empty((0,1))

    model.eval()
    with torch.no_grad():
        for x, *y_list, T, E in dataloader:
            y_true = np.append(y_true, y_list[index])

            x = x.to(device)

            output = model(x)[index].detach().cpu()

            y_pred = torch.cat((y_pred, output))

    y_true = torch.Tensor(y_true)
    y_pred = y_pred.squeeze(-1)

    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    PCC = np.corrcoef(y_true, y_pred)[0][1]
    SCC = spearmanr(y_true, y_pred)[0]

    return MAE, RMSE, PCC, SCC

def eval_result_classification(model, dataloader, device, n_classes, index): # classification
    y_true = np.empty(0)
    y_score = torch.empty((0, n_classes))
    y_pred = np.empty(0)

    model.eval()
    with torch.no_grad():
        for x, *y_list, T, E in dataloader:
            y_true = np.append(y_true, y_list[index])

            x = x.to(device)

            output = model(x)[index].detach().cpu()

            y_score = torch.cat((y_score, output))
            output = F.softmax(output, dim=1).argmax(1)

            y_pred = np.append(y_pred, output.numpy())

    y_true = torch.IntTensor(y_true)
    y_score = F.softmax(y_score, dim=1)
    y_pred = torch.IntTensor(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    if n_classes == 2:
        auroc = roc_auc_score(y_true, y_score[:, 1])
    else:
        auroc = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')

    return accuracy, precision, recall, f1, auroc

def predict(score, num_times): # survival
    Triangle1 = np.tri(num_times, num_times + 1)
    Triangle2 = np.tri(num_times + 1, num_times + 1)

    phi = np.exp(np.dot(score, Triangle1))
    div = np.repeat(np.sum(phi, 1).reshape(-1, 1), phi.shape[1], axis=1)
    density = phi / div
    Survival = np.dot(density, Triangle2)
    hazard = density[:, :-1] / Survival[:, 1:]

    return hazard, density, Survival

def predict_hazard(score, num_times): # survival
    hazard, _, _ = predict(score, num_times)

    return hazard

def predict_survival(score, num_times): # survival
    _, _, survival = predict(score, num_times)
    return survival

def predict_cumulative_hazard(score, num_times): # survival
    hazard = predict_hazard(score, num_times)
    cumulative_hazard = np.cumsum(hazard, 1)
    return cumulative_hazard

def predict_risk(score, num_times, use_log=False): # survival
    cumulative_hazard = predict_cumulative_hazard(score, num_times)
    risk_score = np.sum(cumulative_hazard, 1)
    if use_log:
        return np.log(risk_score)
    else:
        return risk_score

def c_index(score, T, E, num_times): # survival
    risk = predict_risk(score, num_times)

    result = concordance_index_censored(E.astype(bool), T, risk)[0]

    return result

def ibs(score, T, E, num_times, times): # survival
    Survival = predict_survival(score, num_times)

    E_bool = E.astype(bool)
    true = np.array([(E_bool[i], T[i]) for i in range(len(E))], dtype=[('event', np.bool_), ('time', np.float32)])

    max_time = max(T)
    min_time = min(T)

    valid_index = [i for i in range(len(times)) if min_time <= times[i] <= max_time]
    times = times[valid_index]
    Survival = Survival[:, valid_index]

    result = integrated_brier_score(true, true, Survival, times)

    return result

def eval_result_survival(model, dataloader, device, num_times, times, index):
    y_true = np.empty((0, num_times + 1))
    T_true = np.empty(0)
    E_true = np.empty(0)
    score = np.empty((0, num_times))

    model.eval()
    with torch.no_grad():
        for x, *y_list, T, E in dataloader:
            y_true = np.append(y_true, y_list[index])
            T_true = np.append(T_true, T)
            E_true = np.append(E_true, E)

            x = x.to(device)

            output = model(x)[index].detach().cpu().numpy()

            score = np.append(score, output, axis=0)

    C_Index = c_index(score, T_true, E_true, num_times)
    IBS = ibs(score, T_true, E_true, num_times, times)

    return C_Index, IBS

def print_save_result(model, dataloader, device, log, dataset_type,
                      task_name_dict, d_output_dict, num_times, times, flag_survival):
    idx = 0
    if 'regression' in task_name_dict.keys():
        for name in task_name_dict['regression']:
            MAE, RMSE, PCC, SCC = eval_result_regression(model, dataloader, device, idx)
            log.write(f'[{name:^17s}] [{dataset_type:^10s}] MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SCC: {SCC:.4f}\n')
            print(f'[{name:^17s}] [{dataset_type:^10s}] MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SCC: {SCC:.4f}')
            idx += 1

    if 'classification' in task_name_dict.keys():
        for name in task_name_dict['classification']:
            n_classes = d_output_dict[name]
            accuracy, precision, recall, f1, auroc = eval_result_classification(model, dataloader, device, n_classes, idx) # classification
            log.write(f'[{name:^17s}] [{dataset_type:^10s}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}\n') # classification
            print(f'[{name:^17s}] [{dataset_type:^10s}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}') # classification
            idx += 1

    if flag_survival:
        name = 'survival'
        C_Index, IBS = eval_result_survival(model, dataloader, device, num_times, times, idx) # survival
        log.write(f'[{name:^17s}] [{dataset_type:^10s}] C-Index: {C_Index:.4f}, IBS: {IBS:.4f}\n') # survival
        print(f'[{name:^17s}] [{dataset_type:^10s}] C-Index: {C_Index:.4f}, IBS: {IBS:.4f}') # survival

    print('\n')

def run(args):
    set_seed(args.seed)

    train_dataloader, val_dataloader, test_dataloader, gene_list = load_dataset(args.input_file, args.output_file, args.task_file, args.val_ratio, args.test_ratio, args.scaler, args.batch_size, args.n_time_intervals)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    task_name_dict = train_dataloader.dataset.dataset.task_name_dict
    d_output_dict = train_dataloader.dataset.dataset.d_output_dict
    flag_survival = train_dataloader.dataset.dataset.flag_survival
    n_tasks = train_dataloader.dataset.dataset.n_tasks # PCGrad
    idx_task_dict = train_dataloader.dataset.dataset.idx_task_dict # PCGrad
    
    if flag_survival:
        num_times = train_dataloader.dataset.dataset.num_times
        times = train_dataloader.dataset.dataset.times
        Triangle = torch.FloatTensor(np.tri(num_times, num_times + 1, dtype=np.float32)).to(device)
    else:
        num_times = None
        times = None
        Triangle = None

    model = CustomModel(args.embed_file, gene_list, device,
                        args.n_heads, args.d_model, args.dropout, args.d_ff, args.norm_first, args.n_layers,
                        args.aggfunc, args.d_hidden1, args.d_hidden2, args.slope, d_output_dict).to(device)
    if args.xavier_uniform:
        for name, p in model.named_parameters():
            if ('embed' not in name) & (p.dim() > 1):
                nn.init.xavier_uniform_(p)

    if args.noam:
        optimizer = get_std_opt(args.d_model, args.noam_factor, args.noam_warmup, model)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train():
        model.train()
        total_loss = 0
        for x, *y_list, T, E in train_dataloader:
            x = x.to(device)
            y_list = [y.to(device) for y in y_list]

            outputs = model(x)
#             loss = criterion(outputs, y_list, E, Triangle, task_name_dict, flag_survival)
            loss_list = PCGrad_backward(optimizer, outputs, y_list, n_tasks, idx_task_dict, flag_survival, E, Triangle, device) # PCGrad
            loss = sum(loss_list)

#             optimizer.zero_grad()
#             loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
        return total_loss / len(train_dataloader.dataset)

    @torch.no_grad()
    def test():
        model.eval()
        total_loss = 0
        #for x, *y_list, T, E in val_dataloader:
        for x, *y_list, T, E in test_dataloader:
            x = x.to(device)
            y_list = [y.to(device) for y in y_list]

            outputs = model(x)
            loss = criterion(outputs, y_list, E, Triangle, task_name_dict, flag_survival)

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

    print_save_result(model, train_dataloader, device, f, 'Training', task_name_dict, d_output_dict, num_times, times, flag_survival)
    if args.val_ratio > 0:
        print_save_result(model, val_dataloader, device, f, 'Validation', task_name_dict, d_output_dict, num_times, times, flag_survival)
    print_save_result(model, test_dataloader, device, f, 'Test', task_name_dict, d_output_dict, num_times, times, flag_survival)

    f.close()

def main():
    args = parse_args()
    save_args(args)
    run(args)

if __name__ == '__main__':
    main()
