import math

import pandas as pd

import torch
import torch.nn as nn

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