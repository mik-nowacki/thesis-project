import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import wandb

from models.iTransformer.iTransformer import Model
from src.python_scripts.datasets.eeg_window_dataset import EEGWindowDataset

# =====================================================================
#  HYPERPARAMETER CONFIGURATION
# =====================================================================
@dataclass
class Configs:
    seq_len: int = 120          # window (in seconds)
    pred_len: int = 1          # Predicting 1 step ahead (the target BIS)
    output_attention: bool = False
    use_norm: bool = True      # Non-stationary normalization (great for EEG)
    d_model: int = 512         # Dimension of the transformer embeddings
    embed: str = 'fixed'
    freq: str = 'h'
    dropout: float = 0.1
    class_strategy: str = 'projection'
    factor: int = 1            # Attention factor
    n_heads: int = 8           # Multi-head attention
    d_ff: int = 512            # Feed-forward network dimension
    activation: str = 'gelu'
    e_layers: int = 4          # Number of Encoder layers


# =====================================================================
#  MAIN EXECUTION PIPELINE
# =====================================================================
def main():
    # Paths (Running from the thesis-project root on Minerva)
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/cases_data.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Patient IDs and Split
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()
    train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=2026)

    # Initialize Configs (Do this early so we can pass seq_len to the dataset)
    configs = Configs()

    # Initialize Scaler and Datasets
    scaler = StandardScaler()
    
    # We now pass `configs.seq_len` dynamically to prevent mismatches!
    train_dataset = EEGWindowDataset(INPUT_DIR, train_ids, seq_len=configs.seq_len, scaler=scaler, is_training=True)
    test_dataset = EEGWindowDataset(INPUT_DIR, test_ids, seq_len=configs.seq_len, scaler=scaler, is_training=False)

    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Initialize iTransformer Model
    model = Model(configs).to(device)
    
    num_features = train_dataset.num_features
    target_projection = nn.Linear(num_features, 1).to(device) # apply regression layers instead?

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(target_projection.parameters()), lr=0.001)

    # Initialize Weights & Biases
    run_name = f"iTrans_seq{configs.seq_len}_dm{configs.d_model}_elay{configs.e_layers}_bsize{BATCH_SIZE}"
    wandb.init(
        project="eeg-bis-prediction", 
        config=configs.__dict__, 
        name=run_name,
        group="iTransformer-Config-Search"
    )

    # F. Training Loop
    epochs = 30
    print("\nTraining iTransformer Model...")
    for epoch in range(epochs):
        model.train()
        target_projection.train()
        train_loss = 0.0
        
        # Iterating directly over the DataLoader (tqdm removed)
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            dummy_dec = torch.zeros(batch_X.size(0), configs.pred_len, batch_X.size(2)).to(device)
            
            optimizer.zero_grad()
            
            outputs = model(x_enc=batch_X, x_mark_enc=None, x_dec=dummy_dec, x_mark_dec=None)
            predictions = target_projection(outputs)
            predictions = predictions.squeeze(1) 
            
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        target_projection.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                dummy_dec = torch.zeros(batch_X.size(0), configs.pred_len, batch_X.size(2)).to(device)
                
                outputs = model(x_enc=batch_X, x_mark_enc=None, x_dec=dummy_dec, x_mark_dec=None)
                predictions = target_projection(outputs).squeeze(1)
                
                loss = criterion(predictions, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss = val_loss / len(test_loader.dataset)
        val_rmse = np.sqrt(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_rmse": val_rmse})

    # G. Final Evaluation
    print("\n--- FINAL RESULTS ---")
    model.eval()
    target_projection.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            dummy_dec = torch.zeros(batch_X.size(0), configs.pred_len, batch_X.size(2)).to(device)
            
            outputs = model(x_enc=batch_X, x_mark_enc=None, x_dec=dummy_dec, x_mark_dec=None)
            preds = target_projection(outputs).squeeze(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(batch_Y.numpy())

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    print(f"Mean Absolute Error (MAE): {mae:.2f} BIS points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} BIS points")

    wandb.log({"final_test_mae": mae, "final_test_rmse": rmse})
    wandb.finish()

if __name__ == "__main__":
    main()