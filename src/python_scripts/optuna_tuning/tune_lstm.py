import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os

import wandb
import optuna

from models.LSTM.lstm import Model
from src.python_scripts.utils.data_utils import parse_common_args, prepare_datasets
from src.python_scripts.utils.train_utils import run_pytorch_training
from src.python_scripts.utils.optuna_utils import process_optuna_results


# --- Optuna ---
def objective(trial, train_set, val_set, seq_len, has_context, device, save_dir):
    # --- HYPERPARAMETER SPACE ---
    # ======= CHANGE HERE ========
    params = {
        # LSTM
        'input_size': train_set.num_features + (train_set.num_context_features if has_context else 0),
        'has_context': has_context,
        'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        # Optimizer
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        # Dataloader
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'seq_len':seq_len,
        'epochs': 30
    }
    # ============================

    # Initialize W&B
    run = wandb.init(
        project="bis-prediction-optuna-tuning",
        group=f"lstm-seq-{seq_len}-pc-{has_context}",
        name=f"trial_{trial.number}",
        config=params,
        reinit=True
    )

    # Create DataLoaders dynamically for this trial's batch_size
    train_loader = DataLoader(train_set, shuffle=True, batch_size=params['batch_size'], num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=params['batch_size'], num_workers=4, pin_memory=True)

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

    try:
        best_rmse = run_pytorch_training(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, criterion=criterion, device=device,
            epochs=params['epochs'], has_context=has_context, 
            model_type='lstm', trial=trial, save_dir=save_dir
        )
        return best_rmse
    finally:
        wandb.finish()


def main():
    args = parse_common_args(description="Run PyTorch Model Tuning")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_set, val_set = prepare_datasets(
        seq_len=args.seq_len, 
        has_context=args.p_context, 
        is_pytorch=True
    )

    # --- OPTUNA OPTIMIZATION ---
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    save_dir = f'checkpoints/lstm/seq{args.seq_len}_pc{args.p_context}'
    os.makedirs(save_dir, exist_ok=True)

    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, train_set, val_set, args.seq_len, args.p_context, device, save_dir), 
        n_trials=50
    )

    process_optuna_results(
        study=study, 
        save_dir=save_dir, 
        seq_len=args.seq_len, 
        has_context=args.p_context, 
        file_extension='.pt'
    )

if __name__ == "__main__":
    main()