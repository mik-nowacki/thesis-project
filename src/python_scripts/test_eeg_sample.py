import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

from models import ForecastingModel
from dataset import load_test_eeg_samples

# Hyperparameters ---------------------------------------------------
EPOCHS = 15 # 15
BATCH_SIZE = 1
SEQ_LEN = 50
LEARNING_RATE = 2.2e-6
NUMBER_OF_PATIENTS = 10 # doesn't do anything for now
DEVICE = 'cuda'
PATH = 'data/processed/eeg_sample.csv'
WEIGHTS_PATH = 'models/saved_weights/'
# --------------------------------------------------------------------

# Initialize the architecture (must match training)
model = ForecastingModel(
    seq_len=SEQ_LEN, embed_size=16, nhead=4,
    dim_feedforward=1024, dropout=0, device=DEVICE
)

# Load the trained weights
checkpoint = torch.load(f"{WEIGHTS_PATH}eeg_sample_model_weights.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set model to evaluation mode (disables dropout, etc.)
print("Model loaded successfully!")

# ----------- TESTING -----------
model.eval()
preds,actuals = [], []

X_test, Y_test = load_test_eeg_samples()

dataset_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(DEVICE),
                             torch.tensor(Y_test, dtype=torch.float32).to(DEVICE))
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE)

with torch.no_grad():
    for batch_x, batch_y in dataloader_test:
        y_hat = model(batch_x)
        preds.extend(y_hat.cpu().numpy())
        actuals.extend(batch_y.cpu().numpy())

print(f"MAE:  {mean_absolute_error(actuals, preds):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actuals, preds)):.2f}")
print(f"R²:   {r2_score(actuals, preds):.4f}")