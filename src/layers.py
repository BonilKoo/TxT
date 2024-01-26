import math

import torch
import torch.nn.functional as F

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