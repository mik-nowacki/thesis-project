import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

import json
import argparse
import os
import gc

import wandb
import optuna

from models.LSTM.lstm import Model
from src.python_scripts.datasets.eeg_window_dataset import EEGWindowDataset


# --- Optuna ---
def objective(trial, train_dataset, val_dataset, seq_len, device):
    # --- HYPERPARAMETER SPACE ---
    params = {
        # LSTM
        'input_size': train_dataset.patient_X[0].shape[-1],
        'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        # Optimizer
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        # Dataloader
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'epochs': 30
    }

    # custom optuna early stopping
    early_stopping_rounds = 10
    early_stopping_counter = 0

    run_config = params.copy()
    run_config['seq_len'] = seq_len

    # directory to save best model
    save_dir = f'checkpoints/lstm/seq{seq_len}'
    os.makedirs(save_dir, exist_ok=True)

    # Initialize W&B
    run = wandb.init(
        project="bis-prediction",
        group=f"lstm-seq-{seq_len}-v2",
        name=f"trial_{trial.number}",
        config=run_config,
        reinit=True
    )

    # Create DataLoaders dynamically for this trial's batch_size
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params['batch_size'], num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=params['batch_size'], num_workers=4, pin_memory=True)

    # Initialize Model, Loss, Optimizer
    model = Model(
        input_size=params['input_size'], 
        hidden_size=params['hidden_size'], 
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        bidirectional=params['bidirectional']
    ).to(device)

    criterion = nn.MSELoss() # we want to penalize large outliers
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    best_val_rmse = float('inf')

    try:
        for epoch in range(params['epochs']):
            # --- Training ---
            model.train()
            train_sse = 0.0

            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_Y)
                loss.backward()
                optimizer.step()

                train_sse += loss.item() * batch_X.size(0)

            train_mse = train_sse / len(train_loader.dataset)
            train_rmse = np.sqrt(train_mse)

            # --- Validation ---
            model.eval()
            val_sse = 0.0
            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                    
                    preds = model(batch_X)
                    loss = criterion(preds, batch_Y)
                    val_sse += loss.item() * batch_X.size(0)
            
            val_mse = val_sse / len(val_loader.dataset)
            val_rmse = np.sqrt(val_mse)
            # log training/validation progress in W&B
            wandb.log({
                "train_mse": train_mse, 
                "train_rmse": train_rmse,
                "val_mse": val_mse, 
                "val_rmse": val_rmse,
                "epoch": epoch
            })

            # Optuna early stopping / saving best model
            if val_rmse < best_val_rmse: # check if the model still is improving
                best_val_rmse = val_rmse
                early_stopping_counter = 0
                trial.set_user_attr('best_epoch', epoch)
                best_model_path = f'{save_dir}/trial_{trial.number}_best.pt'
                torch.save(model.state_dict(), best_model_path)
            else:
                early_stopping_counter += 1
            
            # Report the intermediate metric to Optuna
            trial.report(val_rmse, epoch)
            # Check if Optuna thinks this trial is unpromising
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}.")
                raise optuna.TrialPruned()
            
            if early_stopping_counter >= early_stopping_rounds:
                print(f"Early stopping triggered at epoch {epoch}")
                break # Exit the epoch loop

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



def main():
    # Parse cmd arguments
    parser = argparse.ArgumentParser(description="Run LSTM Tuning for a specific Sequence Length")
    parser.add_argument('--seq_len', type=int, default=100, help="Length of the sliding window (Default: 100)")
    args = parser.parse_args()

    # Configuration
    SEQ_LEN = args.seq_len
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/train_cases.csv'

    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Patient IDs and Split
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()
    train_ids, val_ids = train_test_split(all_ids, test_size=0.176, random_state=2026)

    # Initialize Scaler and PyTorch Datasets
    # scaler = StandardScaler()
    scaler = RobustScaler() # less sensitive to outliers

    # The training dataset will FIT the scaler
    train_dataset = EEGWindowDataset(
        input_dir=INPUT_DIR, 
        case_ids=train_ids, 
        seq_len=SEQ_LEN, 
        scaler=scaler, 
        is_training=True
    )

    # The validating dataset will USE the fitted scaler
    val_dataset = EEGWindowDataset(
        input_dir=INPUT_DIR,
        case_ids=val_ids,
        seq_len=SEQ_LEN,
        scaler=scaler,
        is_training=False
    )

    # --- OPTUNA OPTIMIZATION ---
    print("\nStarting PyTorch LSTM Hyperparameter Tuning...")
    study = optuna.create_study(direction="minimize", 
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    
    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, SEQ_LEN, device), 
        n_trials=50
    )

    best_trial = study.best_trial
    save_dir = f'checkpoints/lstm/seq{SEQ_LEN}'
    os.makedirs(save_dir, exist_ok=True)

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best RMSE: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save study metadata
    study_results = {
        'best_trial': best_trial.number,
        'best_val_rmse': best_trial.value,
        'best_epoch': best_trial.user_attrs['best_epoch'],
        'params': best_trial.params,
        'seq_len': SEQ_LEN,
    }
    with open(f'{save_dir}/study_results.json', 'w') as f:
        json.dump(study_results, f, indent=2)

    # clean up former best trials
    for trial in study.trials:
        path = f'{save_dir}/trial_{trial.number}_best.pt'
        if trial.number != best_trial.number and os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    main()