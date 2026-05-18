import torch

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer

import numpy as np

class EmbeddingLayer(nn.Module):
    
    def __init__(self, input_size: int, embed_size: int):
        super().__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size

        self.input_embedding = nn.Linear(input_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        
        return self.layer_norm(self.input_embedding(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        # Build the full encoding matrix upfront for all positions up to max_len
        # Shape: [1, max_len, d_model] — batch dim of 1 allows broadcasting across any batch size
        pe = torch.zeros(1, max_len, d_model)
        
        # Column vector [0, 1, 2, ..., max_len-1] — one index per position
        # Shape: [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)

        # One frequency per pair of dimensions
        # Frequencies decrease exponentially: high freq for early dims, low freq for later dims
        # Shape: [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        # Even dimensions get sine waves, odd dimensions get cosine waves
        # Sin/cos pair at the same frequency = unique position fingerprint
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # register_buffer: pe is saved with the model (e.g. when you call torch.save)
        # but is NOT updated by the optimizer — it's a fixed mathematical constant
        self.register_buffer('pe', pe)

    
    def forward(self, x_seq, context_token=None):
        # x_seq:        [batch, k, d_model]  — the EEG temporal tokens
        # context_token: [batch, 1, d_model]  — the patient context, no time meaning
        
        # Add positional fingerprint to EEG tokens only
        # pe[:, :k, :] slices out exactly k position encodings regardless of max_len
        x_seq = x_seq + self.pe[:, :x_seq.size(1), :]
        x_seq = self.dropout(x_seq)
        
        if context_token is not None:
            # Concatenate context token at front — it sits at position 0
            # but carries no positional encoding, only patient identity
            # Final shape: [batch, k+1, d_model]
            return torch.cat([context_token, x_seq], dim=1)
        return x_seq  # [batch, k, d_model] if no context token provided
         

class ContextTokenModule(nn.Module):
    def __init__(self, static_feature_size, d_model):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(static_feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, static_features):
        # static_features: [batch, static_feature_size]
        x = self.mlp(static_features)
        return x.unsqueeze(1)  # [batch, 1, d_model]
    
    
class SequentialTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1, ffn_multiplier=2):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=dropout,
            batch_first=True,      # expects [batch, seq, d_model] not [seq, batch, d_model]
            norm_first=True        # pre-norm for training stability
        )
        
        #Stacks this n_layer times to build the full encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
    
    def make_sequential_mask(self, seq_len, device):
        # Upper triangular matrix of -inf, diagonal is 0
        # Shape: [seq_len, seq_len]
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask
    
    def forward(self, x):
        # x: [batch, k+1, d_model]  (k EEG tokens + 1 context token)
        seq_len = x.size(1)
        mask = self.make_sequential_mask(seq_len, x.device)
        return self.encoder(x, mask=mask)  # [batch, k+1, d_model]
    

class RegressionHead(nn.Module):
    def __init__(self, d_model, dropout=0.1, use_intermediate=True):
        super().__init__()
        
        if use_intermediate:
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
        else:
            self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: [batch, k+1, d_model] — full encoder output including context token
        x = x[:, -1, :]           # just keep the last token's output
        x = self.head(x)          # [batch, k, 1]
        x = torch.sigmoid(x)      # [batch, k, 1] in (0, 1)
        return x.squeeze(-1)      # [batch, k]