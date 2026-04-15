import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import ForecastingModel
from dataset import load_training_eeg_samples, load_normalized_eeg_train

# Hyperparameters ---------------------------------------------------
EPOCHS = 30 # 15
BATCH_SIZE = 1
SEQ_LEN = 200
LEARNING_RATE = 2.2e-6
NUMBER_OF_PATIENTS = 10 # doesn't do anything for now
SPLIT = 0.8 # training split
FEATURES = ['delta','theta','alpha','beta'] # waves to read ['delta','theta','alpha','beta']
DEVICE = 'cuda'
PATH = 'data/processed/eeg_sample.csv'
# --------------------------------------------------------------------

# Build the model
# model = ForecastingModel(
#     input_size = 2, seq_len=SEQ_LEN, embed_size=128, nhead=32,
#     dim_feedforward=1024, dropout=0, device=DEVICE
# )
embed_size = 128
nhead = 32

model = ForecastingModel( # architecture for 1 model
    input_size = len(FEATURES), seq_len=SEQ_LEN, embed_size=embed_size, nhead=nhead,
    dim_feedforward=1024, dropout=0, device=DEVICE
)
model.to(DEVICE)
model.train() # puts model in training mode - enables dropout

# Loss, optimizer, scheduler
criterion = torch.nn.HuberLoss() # standard MSE for small errors, linear penalty for outliers
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# X_train, Y_train = load_training_eeg_samples(PATH, SPLIT, SEQ_LEN, FEATURES)
X_train, Y_train, x_means, x_stds = load_normalized_eeg_train(PATH, SPLIT, SEQ_LEN, FEATURES)

# --------- TRAINING ---------------
dataset_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(DEVICE), 
                            torch.tensor(Y_train, dtype=torch.float32).to(DEVICE))
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE)

# Training Loop
start = time.time()
for epoch in range(EPOCHS):
    for batch_x, batch_y in dataloader_train:
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
end = time.time()
print(f"Total training time: {end-start}")
print(f"Features: {FEATURES}, embed_size: {embed_size}, nhead: {nhead}")
# save_path = "models/saved_weights/eeg_sample_model_weights.pth"
# torch.save({
#     'model_state_dict': model.state_dict()
#     }, save_path)
# print(f"Training complete. Model weights saved to {save_path}")