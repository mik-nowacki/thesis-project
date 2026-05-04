import os
import gc
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.python_scripts.datasets.eeg_window_dataset import EEGWindowDataset

import wandb
import optuna

from src.python_scripts.datasets.eeg_window_dataset import EEGWindowDataset


# --- LSTM ---
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
        
        # only the output from the final time step in the sequence
        # out shape becomes: (batch_size, hidden_size)
        out = out[:, -1, :] 
        
        # Pass through the linear layer to get the final BIS prediction
        out = self.fc(out)
        return out

# --- Optuna ---
def objective(trial, train_dataset, test_dataset, device):
    # Suggest Hyperparameters
    params = {
        'input_size': train_dataset.patient_X[0].shape[-1],
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        'epochs': 30
    }

    # Initialize W&B
    run = wandb.init(
        project="eeg-bis-prediction",
        group="optuna-tuning-lstm-2",
        name=f"trial_{trial.number}",
        config=params,
        reinit=True
    )

    # Create DataLoaders dynamically for this trial's batch_size
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params['batch_size'], num_workers=4)
    test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=params['batch_size'], num_workers=4)

    # Initialize Model, Loss, Optimizer
    model = BISPredictorLSTM(
        input_size=params['input_size'], 
        hidden_size=params['hidden_size'], 
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    best_val_rmse = float('inf')

    try:
        # training loop
        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0.0

            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_Y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()* batch_X.size(0)

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

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_rmse": val_rmse
            })

            # --- OPTUNA EARLY STOPPING ---
            # Report the intermediate metric to Optuna
            trial.report(val_rmse, epoch)
            # Check if Optuna thinks this trial is unpromising
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}.")
                raise optuna.TrialPruned()

        return best_val_rmse

    except RuntimeError as e:
        # Catch PyTorch GPU Out of Memory errors
        if "out of memory" in str(e).lower():
            print(f"\n[Warning] Trial {trial.number} failed due to GPU OOM. Pruning...")
            wandb.log({"error": "GPU Out of Memory"})
            raise optuna.TrialPruned()
        else:
            raise e

    finally:
        wandb.finish()
        
        # Strict memory cleanup for PyTorch
        del model, optimizer, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()



# --- Main execution pipeline
def main():
    # Configuration
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/cases_data.csv'
    SEQ_LEN = 60 # 60-second window

    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Patient IDs and Split
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()
    train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

    # Initialize Scaler and PyTorch Datasets
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

    # --- OPTUNA OPTIMIZATION ---
    print("\nStarting PyTorch LSTM Hyperparameter Tuning...")
    study = optuna.create_study(direction="minimize")
    
    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, train_dataset, test_dataset, device), 
        n_trials=30
    )

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best Val RMSE: {study.best_trial.value:.4f} BIS points")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()