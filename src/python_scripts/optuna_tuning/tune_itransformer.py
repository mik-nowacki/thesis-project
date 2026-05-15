import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataclasses import dataclass

import os
import json

import wandb
import optuna

from models.iTransformer.iTransformer import Model
from src.python_scripts.utils.data_utils import parse_common_args, prepare_datasets
from src.python_scripts.utils.train_utils import run_pytorch_training
from src.python_scripts.utils.optuna_utils import process_optuna_results

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


def objective(trial, train_dataset, val_dataset, seq_len, has_context, device, save_dir):
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
        'has_context': has_context,
        # Optimizer
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        # Dataloader
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'epochs': 30
    }

    run = wandb.init(
        project="bis-prediction-optuna-tuning",
        group=f"itrans-seq-{seq_len}-pc-{has_context}",
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

    try:
        best_rmse = run_pytorch_training(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, criterion=criterion, device=device,
            epochs=training_params['epochs'], has_context=has_context, 
            model_type='itransformer', save_dir=save_dir
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
    save_dir = f'checkpoints/itransformer/seq{args.seq_len}_pc{args.p_context}'
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