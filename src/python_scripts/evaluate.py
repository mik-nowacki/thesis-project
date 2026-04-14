import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import ForecastingModel

# Hyperparameters ---------------------------------------------------
EPOCHS = 30 # 15
BATCH_SIZE = 1
SEQ_LEN = 200
LEARNING_RATE = 2.2e-6
NUMBER_OF_PATIENTS = 10 # doesn't do anything for now
DEVICE = 'cuda'
PATH = 'data/raw/physionet.org/files/vitaldb/1.0.0/vital_files/'
WEIGHTS_PATH = 'models/saved_weights/'
# --------------------------------------------------------------------

# Initialize the architecture (must match training)
model = ForecastingModel(
    seq_len=SEQ_LEN, embed_size=16, nhead=4,
    dim_feedforward=1024, dropout=0, device=DEVICE
)

# 3. Load the trained weights
model.load_state_dict(torch.load(f"{WEIGHTS_PATH}forecasting_model_weights.pth", weights_only=True))
model.eval() # Set model to evaluation mode (disables dropout, etc.)
print("Model loaded successfully!")

preds, actuals = [], []
