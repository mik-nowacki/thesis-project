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

import wandb
import optuna

from models.iTransformer.iTransformer import Model
from src.python_scripts.datasets.eeg_window_dataset import EEGWindowDataset

@dataclass
class iTransformerConfig:
    """Configuration parameters for the iTransofmer constructor"""
    seq_len: int 
    pred_len: int 
    output_attention: bool
    use_norm: bool
    mask: bool
    activation: str
    d_model: int
    e_layers: int
    dropout: float
    n_heads: int
    d_ff: int       


def objective(trial, train_dataset, val_dataset, test_dataset, seq_len, device):
    # --- HYPERPARAMETER SPACE ---
    params = {
        # iTransformer
        'seq_len': seq_len, 
        'pred_len': 1, 
        'output_attention': False,
        'use_norm': True,
        'mask': True,
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
        'd_model': trial.suggest_categorical('d_model', [128, 256, 512]), # The paper tested [256, 512]
        'e_layers': trial.suggest_categorical('e_layers', [2,3,4]), # The paper found optimal values between 2, 3, and 4.
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'n_heads': trial.suggest_categorical('n_heads', [4, 6, 8]),
        'd_ff': trial.suggest_categorical('d_ff', [256, 512, 1024, 2048]), # usually 2x or 4x d_model
        # Optimizer
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        # Dataloader
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'epochs': 30
    }

    # custom optuna early stopping
    early_stopping_rounds = 10
    early_stopping_counter = 0

    run = wandb.init(
        project="eeg-bis-prediction",
        group=f"itrans-seq-{seq_len}-v1",
        name=f"trial_{trial.number}",
        config=params,
        reinit=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params["batch_size"], num_workers=4, pin_memory=True)
    val_loader  = DataLoader(val_dataset, shuffle=False, batch_size=params["batch_size"], num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=params["batch_size"], num_workers=4, pin_memory=True)

    # Initialize iTransformer Model
    configs = iTransformerConfig(**params)
    model = Model(configs).to(device)
    
    num_features = train_dataset.num_features
    target_projection = nn.Linear(num_features, 1).to(device) # apply regression layers instead?

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(target_projection.parameters()), lr=params['learning_rate'])

    best_val_rmse = float('inf')

    try:
        # --- Training ---
        for epoch in range(params['epochs']):
            model.train()
            target_projection.train()
            train_sse = 0.0
            
            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                                
                optimizer.zero_grad()
                outputs = model(x_enc=batch_X, x_mark_enc=None) # maybe use this for patient context?
                predictions = target_projection(outputs)
                predictions = predictions.squeeze(1) 
                
                loss = criterion(predictions, batch_Y)
                loss.backward()
                optimizer.step()
                
                train_sse += loss.item() * batch_X.size(0)
                
            train_mse = train_sse / len(train_loader.dataset)
            train_rmse = np.sqrt(train_mse)
            
            # --- Validation phase ---
            model.eval()
            target_projection.eval()
            val_sse = 0.0

            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                    
                    outputs = model(x_enc=batch_X, x_mark_enc=None)
                    predictions = target_projection(outputs).squeeze(1)
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
                "epoch": epoch+1
            })

            if best_val_rmse - val_rmse > 0.005:
                best_val_rmse = val_rmse
                early_stopping_counter = 0
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

        # --- Testing ---
        model.eval()
        target_projection.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = batch_X.to(device)

                outputs = model(x_enc=batch_X, x_mark_enc=None)
                preds = target_projection(outputs).squeeze(1).cpu().numpy()
                
                all_preds.extend(preds.flatten())
                all_targets.extend(batch_Y.numpy())

        test_mae = mean_absolute_error(all_targets, all_preds)
        test_mse = mean_squared_error(all_targets, all_preds)
        test_rmse = root_mean_squared_error(all_targets, all_preds)
        test_r2 = r2_score(all_targets, all_preds)

        abs_error = np.abs(all_targets - all_preds)

        within_2_5 = abs_error <= 2.5
        tolerance_accuracy_2_5 = np.mean(within_2_5) * 100

        within_5 = abs_error <= 5
        tolerance_accuracy_5 = np.mean(within_5) * 100

        within_10 = abs_error <= 10
        tolerance_accuracy_10 = np.mean(within_10) * 100

        # log final tests in W&B
        wandb.log({
            "test_mae": test_mae, 
            "test_mse": test_mse, 
            "test_rmse": test_rmse,
            "test_r2_score": test_r2,
            "test_tolerance_accuracy_2_5": tolerance_accuracy_2_5,
            "test_tolerance_accuracy_5": tolerance_accuracy_5,
            "test_tolerance_accuracy_10": tolerance_accuracy_10,
        })

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
    TRAINING_SIZE = 0.7
    SEQ_LEN = args.seq_len    
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/cases_data.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Patient IDs and Split
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()
    train_ids, val_test_ids = train_test_split(all_ids, test_size=1-TRAINING_SIZE, random_state=2026)
    val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=2026)
    
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

    # The testing dataset will USE the fitted scaler
    test_dataset = EEGWindowDataset(
        input_dir=INPUT_DIR, 
        case_ids=test_ids, 
        seq_len=SEQ_LEN, 
        scaler=scaler, 
        is_training=False
    )

    # --- OPTUNA OPTIMIZATION ---
    print("\nStarting iTransformer Hyperparameter Tuning...")
    study = optuna.create_study(direction="minimize")
    
    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, test_dataset, SEQ_LEN, device), 
        n_trials=50
    )

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best Val RMSE: {study.best_trial.value:.4f} BIS points")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()