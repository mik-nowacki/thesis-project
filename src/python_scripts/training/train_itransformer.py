import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

from dataclasses import dataclass
import argparse
import gc
import os
import json

import wandb
import optuna

from models.iTransformer.iTransformer import Model
from src.python_scripts.datasets.eeg_window_dataset import EEGWindowDataset

@dataclass
class iTransformerConfig:
    """Configuration parameters for the iTransofmer constructor"""
    num_features: int
    seq_len: int 
    pred_len: int 
    use_norm: bool
    mask: bool
    activation: str
    d_model: int
    e_layers: int
    dropout: float
    n_heads: int
    d_ff: int       


def objective(trial, train_dataset, val_dataset, seq_len, device):
    # --- HYPERPARAMETER SPACE ---
    d_model = trial.suggest_categorical('d_model', [128, 256, 512]) # The paper tested [256, 512]
    ff_multiplier = trial.suggest_categorical('ff_multiplier', [2, 4])
    model_params = {
        # iTransformer
        'num_features': train_dataset.num_features,
        'seq_len': seq_len, 
        'pred_len': 1, 
        'use_norm': True,
        'mask': True,
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
        'd_model': d_model,
        'e_layers': trial.suggest_categorical('e_layers', [2,3,4]), # The paper found optimal values between 2, 3, and 4.
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'n_heads': trial.suggest_categorical('n_heads', [4, 6, 8]),
        'd_ff': d_model * ff_multiplier, # Calculate d_ff dynamically
    }
    training_params = {
        # Optimizer
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        # Dataloader
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'epochs': 30
    }

    # custom optuna early stopping
    early_stopping_rounds = 10
    early_stopping_counter = 0

    # directory to save best model
    save_dir = f'checkpoints/itransformer/seq{seq_len}'
    os.makedirs(save_dir, exist_ok=True)

    run = wandb.init(
        project="eeg-bis-prediction",
        group=f"itrans-seq-{seq_len}-v1",
        name=f"trial_{trial.number}",
        config={**model_params, **training_params},
        reinit=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=training_params["batch_size"], num_workers=4, pin_memory=True)
    val_loader  = DataLoader(val_dataset, shuffle=False, batch_size=training_params["batch_size"], num_workers=4, pin_memory=True)

    # Initialize iTransformer Model
    configs = iTransformerConfig(**model_params)
    model = Model(configs).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    best_val_rmse = float('inf')

    try:
        for epoch in range(training_params['epochs']):
            # --- Training ---
            model.train()
            train_sse = 0.0
            
            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                                
                optimizer.zero_grad()
                predictions = model(x_enc=batch_X, x_mark_enc=None) # maybe use x_mark_enc for patient context?
                
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
                    
                    predictions = model(x_enc=batch_X, x_mark_enc=None)
                    loss = criterion(predictions, batch_Y)
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
    parser.add_argument('--seq_len', type=int, default=200, help="Length of the sliding window (Default: 200)")
    args = parser.parse_args() 

    # Configuration
    SEQ_LEN = args.seq_len    
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/train_cases.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Patient IDs and Split
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()
    train_ids, val_ids = train_test_split(all_ids, test_size=0.176, random_state=2026)
    
    # Initialize Scaler and Datasets
    # scaler = StandardScaler()
    scaler = RobustScaler()
    
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
    save_dir = f'checkpoints/itransformer/seq{SEQ_LEN}'
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