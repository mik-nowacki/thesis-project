from torch import nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, input_size=1, embed_size=8, conv1d_emb=False, kernel_size=3):
        super().__init__()
        # kernel_size must be odd to make padding math clean
        # padding needed = kernel_size - 1
        self.input_size = input_size
        self.embed_size = embed_size
        self.conv1d_emb = conv1d_emb
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        # Conv1d(in_channels, out_channels, kernel_size)
        # in_channels = 1 because each timestep is a scalar
        if conv1d_emb:
            self.conv = nn.Conv1d(in_channels=input_size, out_channels=embed_size, kernel_size=kernel_size)
        else:
            self.input_embedding = nn.Linear(input_size, embed_size)
    
    def forward(self, x):
        # shape: (batch, 200, 1)
        if self.conv1d_emb:
            # pad the START of the sequence (dimension 1) with self.padding zeros
            # F.pad(tensor, (left, right, top, bottom), value) — but for 3D tensors
            # the pad tuple goes from last dim backwards: (dim2_left, dim2_right, dim1_left, dim1_right)
            # torch.nn.functional.pad(input, pad, mode='constant', value=None)
            x = F.pad(x, (0, 0, self.padding, 0), "constant", 0)
            # Conv1d expects (batch, channels, length) — transpose dims 1 and 2
            x = x.transpose(2,1)
            # apply convolution
            x = self.conv(x)
            # transpose back to (batch, seq_len, embed_size)
            x = x.transpose(2,1)
        else:
            x = self.input_embedding(x)
        
        return x