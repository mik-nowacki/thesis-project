from .layers import EmbeddingLayer, ContextTokenModule, PositionalEncoding, SequentialTransformerEncoder, RegressionHead
from torch import nn
import torch

class DoATransformer(nn.Module):
    def __init__(self, config):

        super(DoATransformer, self).__init__()
        sequential_input_size = config.sequential_input_size   # number of sequential features (19)
        static_input_size = config.static_input_size       # number of static features (10-20)
        d_model = config.d_model                 # embedding dimension
        n_heads = config.n_heads                 # attention heads
        n_layers = config.n_layers               # transformer encoder layers
        dropout = config.dropout
        
        ffn_multiplier = getattr(config, 'ffn_multiplier', 2)         #FF layer size multiplicator
        use_intermediate_prediction = getattr(config, 'use_intermediate_prediction', True) #Use intermediate layer for the regression head
        self.has_context = getattr(config, 'has_context', True)
        
        self.embedding = EmbeddingLayer(sequential_input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = SequentialTransformerEncoder(d_model, n_heads, n_layers, dropout, ffn_multiplier)
        self.head = RegressionHead(d_model, dropout, use_intermediate_prediction)

        if self.has_context:
            self.context = ContextTokenModule(static_input_size, d_model)
    
    def forward(self, x_seq, x_static=None):
        # x_seq:    [batch, k, sequential_input_size]
        # x_static: [batch, static_input_size]

        if self.has_context and x_static is None:
            raise ValueError("Model was initialised with has_context=True but x_static was not provided.")
        
        # 1. embed sequential features
        x = self.embedding(x_seq)              # [batch, k, d_model]
        
        # 2. create patient context token
        ctx = self.context(x_static) if self.has_context else None
        
        # 3. add positional encoding to EEG tokens and prepend context token
        x = self.positional_encoding(x, ctx)   # [batch, k+1, d_model]
        
        # 4. pass through causal transformer encoder
        x = self.encoder(x)                    # [batch, k+1, d_model]
        
        # 5. regression head discards context token output and predicts BIS
        x = self.head(x)                       # [batch]
        
        return x