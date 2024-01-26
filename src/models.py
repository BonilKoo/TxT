from utils import get_clones

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