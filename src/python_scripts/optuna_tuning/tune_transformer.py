import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataclasses import dataclass

import sys
sys.path.append('C:/GU/Thesis/code/thesis-project')
import os
import json

import wandb
import optuna

from models.vanTransformer.transformer_eeg import DoATransformer
from src.python_scripts.utils.data_utils import parse_common_args, prepare_datasets_trans
from src.python_scripts.utils.train_utils import run_pytorch_training
from src.python_scripts.utils.optuna_utils import process_optuna_results

@dataclass
class TransformerConfig:
    """Configuration parameters for the Transformer constructor"""
    sequential_input_size : int   # number of sequential features (19)
    static_input_size: int       # number of static features (10-20)
    d_model: int              # embedding dimension
    n_heads: int                 # attention heads
    n_layers: int                # transformer encoder layers
    dropout: float              # Dropout ratio
    ffn_multiplier: int = 2        #FF layer size multiplicator
    use_intermediate_prediction: bool = True #Use intermediate layer for the regression head
    has_context: bool = True          #Whether to include patient context in the model


def objective(trial, train_dataset, val_dataset, seq_len, has_context, device, save_dir):
    # --- HYPERPARAMETER SPACE ---
    
    model_params = {
        # Transformer
        'sequential_input_size': train_dataset.num_features,
        'has_context': has_context,
        'static_input_size': train_dataset.size_context, 
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128, 256]),
        'n_layers': trial.suggest_categorical('n_layers', [3, 4, 5, 6]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
        'ffn_multiplier': trial.suggest_categorical('ffn_multiplier', [2, 4]),
        'use_intermediate_prediction': trial.suggest_categorical('use_intermediate_prediction', [True, False])
        
    }
    training_params = {
        'has_context': has_context,
        # Optimizer
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        # Dataloader
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'epochs': 20
    }

    run = wandb.init(
        project="bis-prediction-optuna-tuning",
        group=f"trans-seq-{seq_len}",
        name=f"trans-trial_{trial.number}",
        config={**model_params, **training_params},
        reinit=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=training_params["batch_size"], num_workers=4, pin_memory=True)
    val_loader  = DataLoader(val_dataset, shuffle=False, batch_size=training_params["batch_size"], num_workers=4, pin_memory=True)

    # Initialize Transformer Model
    configs = TransformerConfig(**model_params)
    model = DoATransformer(configs).to(device)

    criterion = nn.MSELoss()
    #Build in a learning rate schedule
    optimizer = optim.AdamW(model.parameters(), lr=training_params['learning_rate'], weight_decay=training_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params['epochs'], eta_min=training_params['learning_rate'] * 0.01)

    try:
        best_rmse = run_pytorch_training(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, criterion=criterion, device=device,
            epochs=training_params['epochs'], has_context=has_context, 
            model_type='transformer', scheduler=scheduler, save_dir=save_dir
        )
        return best_rmse
    finally:
        wandb.finish()


def main():
    args = parse_common_args(description="Run PyTorch Model Tuning")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_set, val_set = prepare_datasets_trans(
        seq_len=args.seq_len, 
        has_context=args.p_context,
    )

    # --- OPTUNA OPTIMIZATION ---
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    save_dir = f'checkpoints/transformer/seq{args.seq_len}_pc{args.p_context}'
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

