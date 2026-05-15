import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from models.LSTM.lstm import Model
from src.python_scripts.utils.data_utils import parse_common_args, prepare_datasets
from src.python_scripts.utils.train_utils import run_pytorch_training

# --- Optuna ---
def train_lstm(train_dataset, val_dataset, seq_len, has_context, device, save_dir):
    # --- HYPERPARAMETER SPACE ---
    # ======= CHANGE HERE ========
    params = {
        # LSTM
        'input_size': train_dataset.num_features + (train_dataset.num_context_features if has_context else 0),
        'has_context': has_context,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.07521665103217301,
        'bidirectional': False,
        # Optimizer
        'learning_rate': 0.0001851924021756676,
        # Dataloader
        'batch_size': 256,
        'seq_len':seq_len,
        'epochs': 30
    }
    # ============================

    # Initialize W&B
    run = wandb.init(
        project="bis-prediction-training",
        group=f"lstm",
        name=f'lstm_batch_{params["batch_size"]}_{seq_len}_h_{params["hidden_size"]}_pc_{params["has_context"]}',
        config=params,
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

    try:
        best_rmse = run_pytorch_training(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, criterion=criterion, device=device,
            epochs=params['epochs'], has_context=has_context, 
            model_type='lstm', save_dir=save_dir
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

    save_dir = None
    
    train_lstm(train_set, val_set, args.seq_len, args.p_context, device, save_dir)

if __name__ == "__main__":
    main()