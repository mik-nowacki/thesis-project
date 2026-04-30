import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoderLayer


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, nhead, dim_feedforward, dropout, seq_len, device):
        super().__init__()
        self.seq_len = seq_len
        self.device = device

        # PyTorch provides the full encoder layer
        # batch_first=True means input is (batch, seq, features) not (seq, batch, features)
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        # Generate causal mask — upper triangular matrix of -inf
        # shape must be (seq_len, seq_len)
        mask = torch.triu(
            torch.full((x.size(1),x.size(1)), float('-inf'), device=self.device),
            diagonal = 1 # leave the main diagonal as 0
        )

        x = self.encoder_layer(x, src_mask=mask)
        return x