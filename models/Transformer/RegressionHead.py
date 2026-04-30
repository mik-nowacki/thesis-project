import torch
from torch import nn

class RegressionHead(nn.Module):
    def __init__(self, seq_len, embed_size, dim_feedforward, dropout):
        super().__init__()
        # after flattening, input size is seq_len * embed_size
        flat_size = seq_len * embed_size
        
        # stack of linear layers squeezing down
        # original model goes: flat -> ff -> ff/2 -> ff/4 -> ff/16 -> ff/64 -> 1
        self.linear1 = nn.Linear(flat_size, dim_feedforward)
        self.linear2 = nn.Linear(int(dim_feedforward), int(dim_feedforward/2))
        self.linear3 = nn.Linear(int(dim_feedforward/2), int(dim_feedforward/4))
        self.linear4 = nn.Linear(int(dim_feedforward/4), int(dim_feedforward/16))
        self.linear5 = nn.Linear(int(dim_feedforward/16), int(dim_feedforward/64))
        self.outlayer = nn.Linear(int(dim_feedforward/64), 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # shape (batch, seq_len, embed_size)

        # flatten everything except batch dim
        x = x.reshape(x.size(0), x.size(1)*x.size(2))

        # pass through each linear layer with relu + dropout
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear5(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.outlayer(x)
