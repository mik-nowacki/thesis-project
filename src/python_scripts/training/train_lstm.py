import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import wandb


class EEGWindowDataset(Dataset):
    def __init__(self, input_dir, case_ids, seq_len, scaler=None, is_training=True):
        self.seq_len = seq_len
        self.patient_X = []
        self.patient_Y = []
        self.index_map = []  # Stores tuples of (patient_index, timestep_start)
        
        all_X_for_scaler = []
        
        print(f"Loading raw data into memory (Training={is_training})...")
        patient_idx = 0
        
        for cid in case_ids:
            sample_path = os.path.join(input_dir, f'case_{cid}.pt')
            if not os.path.exists(sample_path):
                continue
                
            data = torch.load(sample_path, weights_only=False)
            
            # Neural networks output NaN instantly if inputs have NaNs.
            # We replace any missing EEG features with 0.0 safely.
            x = np.nan_to_num(data['features'].numpy(), nan=0.0) 
            y = data['bis'].numpy()
            
            if x.shape[0] <= seq_len:
                continue
            
            self.patient_X.append(x)
            self.patient_Y.append(y)
            
            if is_training:
                all_X_for_scaler.append(x)
            
            # Pre-calculate valid indices! 
            # We only record the start index if the target Y at the END of the window is NOT NaN
            for start_t in range(x.shape[0] - seq_len):
                target_y = y[start_t + seq_len]
                if not np.isnan(target_y):
                    self.index_map.append((patient_idx, start_t))
                    
            patient_idx += 1

        # Fit and Apply Scaler on flat 2D data (super fast and memory efficient)
        if is_training and scaler is not None:
            print("Fitting standard scaler...")
            stacked_X = np.vstack(all_X_for_scaler)
            scaler.fit(stacked_X)
            
        if scaler is not None:
            print("Applying scaler...")
            for i in range(len(self.patient_X)):
                self.patient_X[i] = scaler.transform(self.patient_X[i])
                
        # Convert to tensors once to save CPU time during training
        self.patient_X = [torch.tensor(arr, dtype=torch.float32) for arr in self.patient_X]
        self.patient_Y = [torch.tensor(arr, dtype=torch.float32) for arr in self.patient_Y]
        print(f"Dataset ready. Total valid 60s windows: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. Look up which patient and which timestamp this index corresponds to
        p_idx, start_t = self.index_map[idx]
        
        # 2. Slice the 60-step window from the raw data ON THE FLY
        X_window = self.patient_X[p_idx][start_t : start_t + self.seq_len]
        
        # 3. Grab the target BIS value at the end of the window
        Y_target = self.patient_Y[p_idx][start_t + self.seq_len]
        
        # Return X and Y (unsqueeze Y to make it shape [1] instead of a scalar)
        return X_window, Y_target.unsqueeze(0)


# =====================================================================
# 1. DATA LOADING FUNCTION (Unchanged)
# =====================================================================
# def load_pt_samples(input_dir, case_ids, seq_len):
#     """Loads 3D sliding windows directly from .pt files for specific patients."""
#     X_list = []
#     Y_list = []

#     for cid in case_ids:
#         sample_path = os.path.join(input_dir, f'case_{cid}.pt')
#         if not os.path.exists(sample_path):
#             continue
            
#         data = torch.load(sample_path, weights_only=False)
#         x = data['features'].numpy()
#         y = data['bis'].numpy()
            
#         num_samples = x.shape[0]
#         if num_samples <= seq_len:
#             continue
            
#         # Create sliding windows per patient
#         X_case = np.array([x[i:i+seq_len] for i in range(num_samples - seq_len)])
#         Y_case = np.array([y[i+seq_len] for i in range(num_samples - seq_len)]).reshape(-1, 1)
        
#         X_list.append(X_case)
#         Y_list.append(Y_case)

#     # Combine all patients
#     X = np.concatenate(X_list, axis=0)
#     Y = np.concatenate(Y_list, axis=0)
    
#     return X, Y

# =====================================================================
# 2. LSTM MODEL DEFINITION
# =====================================================================
class BISPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(BISPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True means input shape should be (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        out, _ = self.lstm(x)
        
        # We only want the output from the final time step in the sequence
        # out shape becomes: (batch_size, hidden_size)
        out = out[:, -1, :] 
        
        # Pass through the linear layer to get the final BIS prediction
        out = self.fc(out)
        return out

# =====================================================================
# 3. MAIN EXECUTION PIPELINE
# =====================================================================
# Configuration
INPUT_DIR = 'data/processed/eeg'
CASES_FILE = 'data/processed/cases_data.csv'
SEQ_LEN = 60 # 60-second window

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# A. Load Patient IDs and Split
cases_master = pd.read_csv(CASES_FILE)
all_ids = cases_master['caseid'].tolist()
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

# B. Initialize Scaler and PyTorch Datasets
scaler = StandardScaler()

# The training dataset will FIT the scaler
train_dataset = EEGWindowDataset(
    input_dir=INPUT_DIR, 
    case_ids=train_ids, 
    seq_len=SEQ_LEN, 
    scaler=scaler, 
    is_training=True
)

# The testing dataset will USE the fitted scaler
test_dataset = EEGWindowDataset(
    input_dir=INPUT_DIR, 
    case_ids=test_ids, 
    seq_len=SEQ_LEN, 
    scaler=scaler, 
    is_training=False
)

# C. Prepare DataLoaders
BATCH_SIZE = 256
# num_workers=4 will use multiple CPU cores to slice the windows in parallel
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)

# D. Define Model Parameters & Initialize W&B
params = {
    'input_size': train_dataset.patient_X[0].shape[-1], # Grabs the feature count dynamically
    'hidden_size': 64,                     
    'num_layers': 2,                       
    'learning_rate': 0.001,
    'epochs': 30,
    'batch_size': BATCH_SIZE
}

wandb.init(project="eeg-bis-prediction", config=params, name="lstm-baseline")

model = BISPredictorLSTM(
    input_size=params['input_size'], 
    hidden_size=params['hidden_size'], 
    num_layers=params['num_layers']
).to(device)

criterion = nn.MSELoss() # Same as 'reg:squarederror' in XGBoost
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

# E. Train the Model
print("\nTraining LSTM Model...")
for epoch in range(params['epochs']):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        
    train_loss = train_loss / len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)
            val_loss += loss.item() * batch_X.size(0)
            
    val_loss = val_loss / len(test_loader.dataset)
    val_rmse = np.sqrt(val_loss)
    
    print(f"Epoch [{epoch+1}/{params['epochs']}] | Train MSE: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")
    
    # Log to W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_rmse": val_rmse
    })

# F. Final Evaluation
print("\n--- FINAL RESULTS ---")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X = batch_X.to(device)
        preds = model(batch_X).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_Y.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

print(f"Mean Absolute Error (MAE): {mae:.2f} BIS points")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} BIS points")

wandb.log({
    "final_test_mae": mae,
    "final_test_rmse": rmse
})

wandb.finish()