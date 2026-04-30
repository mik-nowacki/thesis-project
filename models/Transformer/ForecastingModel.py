from torch import nn

from models.Transformer.EmbeddingLayer import EmbeddingLayer
from models.Transformer.PositionalEncoding import PositionalEncoding
from models.Transformer.TransformerBlock import TransformerBlock
from models.Transformer.RegressionHead import RegressionHead

class ForecastingModel(nn.Module):
    def __init__(self, 
                input_size,
                seq_len, 
                embed_size, 
                nhead, 
                dim_feedforward, 
                dropout, 
                conv1d_emb=True,
                kernel_size=3,
                device='cpu'):
        super().__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conv1d_emb = conv1d_emb
        self.kernel_size = kernel_size
        self.device = device

        self.embedding_layer = EmbeddingLayer(
            input_size=input_size,
            embed_size=embed_size,
            conv1d_emb=conv1d_emb,
            kernel_size=kernel_size
        )

        self.pos_encoding = PositionalEncoding(embed_size, dropout)

        self.transformer_block = TransformerBlock(
            embed_size=embed_size, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_len=seq_len,
            device=device
        )

        self.regression_head = RegressionHead(
            seq_len=seq_len,
            embed_size=embed_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )


    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.pos_encoding(x)
        x = self.transformer_block(x)
        x = self.regression_head(x)
        return x