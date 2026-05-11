import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2, bidirectional=False):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True means input shape should be (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional
                            )
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        out, _ = self.lstm(x)
        
        # only the output from the final time step in the sequence
        # out shape becomes: (batch_size, hidden_size)
        out = out[:, -1, :] 
        
        # Pass through the linear layer to get the final BIS prediction
        out = self.fc(out)
        return out