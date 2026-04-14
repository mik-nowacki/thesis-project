import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from models import ForecastingModel
from dataset import load_train_data


# Hyperparameters ---------------------------------------------------
EPOCHS = 30 # 15
BATCH_SIZE = 1
SEQ_LEN = 200
LEARNING_RATE = 2.2e-6
NUMBER_OF_PATIENTS = 10 # doesn't do anything for now
DEVICE = 'cuda'
PATH = 'data/raw/physionet.org/files/vitaldb/1.0.0/vital_files/'
# --------------------------------------------------------------------

# Build the model
# model = ForecastingModel(
#     input_size = 2, seq_len=SEQ_LEN, embed_size=128, nhead=32,
#     dim_feedforward=1024, dropout=0, device=DEVICE
# )

model = ForecastingModel( # architecture for 1 model
    input_size = 2, seq_len=SEQ_LEN, embed_size=16, nhead=4,
    dim_feedforward=1024, dropout=0, device=DEVICE
)
model.to(DEVICE)
model.train() # puts model in training mode - enables dropout

# Loss, optimizer, scheduler
criterion = torch.nn.HuberLoss() # standard MSE for small errors, linear penalty for outliers
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=0.9)

dataloader, stats = load_train_data(PATH, BATCH_SIZE, SEQ_LEN, DEVICE)

# Training Loop
for epoch in range(EPOCHS):
    for batch_x, batch_y in dataloader:
        # clear old gradients
        optimizer.zero_grad()
        # forward pass - model has a look a the data
        y_hat = model(batch_x)
        # compute loss
        loss = criterion(y_hat, batch_y)
        # backward pass - compute gradients
        loss.backward()
        # update weights using gradients - model is learning
        optimizer.step()

    # decay learning rate once per epoch
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss.item():.6f}")

save_path = "models/saved_weights/forecasting_model_weights.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    **stats
    }, save_path)
print(f"Training complete. Model weights saved to {save_path}")