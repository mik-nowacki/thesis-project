import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataclasses import dataclass
import wandb

from models.iTransformer.iTransformer import Model
from src.python_scripts.utils.data_utils import parse_common_args, prepare_datasets
from src.python_scripts.utils.train_utils import run_pytorch_training

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


def train_itransformer(train_dataset, val_dataset, seq_len, has_context, device, save_dir):
    # --- HYPERPARAMETER SPACE ---
    d_model = 512 # The paper tested [256, 512]
    ff_multiplier = 4
    model_params = {
        # iTransformer
        'num_features': train_dataset.num_features,
        'seq_len': seq_len, 
        'pred_len': 1, 
        'use_norm': True,
        'mask': True,
        'activation': 'gelu',
        'd_model': d_model,
        'e_layers': 2, # The paper found optimal values between 2, 3, and 4.
        'dropout': 0.1344449062328291,
        'n_heads': 6,
        'd_ff': d_model * ff_multiplier, # Calculate d_ff dynamically   
    }
    training_params = {
        'has_context': has_context,
        # Optimizer
        'learning_rate': 0.004364780954369978,
        # Dataloader
        'batch_size': 128,
        'epochs': 30
    }

    run = wandb.init(
        project="bis-prediction-training",
        group=f"itrans",
        name=f'itrans_batch_{training_params["batch_size"]}_{seq_len}_d_{model_params["d_model"]}_pc_{has_context}',
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

    save_dir = None

    train_itransformer(train_set, val_set, args.seq_len, args.p_context, device, save_dir)


if __name__ == "__main__":
    main()