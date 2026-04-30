import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe will hold the positional encoding matrix
        # shape: (max_len, 1, d_model) - the middle 1 is for batch broadcasting
        pe = torch.zeros(max_len, 1, d_model)
        
        # position: column vector of indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(max_len).unsqueeze(1)

        # div_term: the frequency scaling - shape (d_model/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        
        # fill even dims with sin, odd dims with cos
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # register_buffer means pe is part of the module but NOT a trainable parameter
        pe = pe.transpose(0,1)
        self.register_buffer('pe',pe)

    def forward(self, x):
        # shape: (batch, seq_len, d_model)
        #               1,  max_len, d_model (use ':' instead of '0' to preserve the dimensions (3D instead of 2D))
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)