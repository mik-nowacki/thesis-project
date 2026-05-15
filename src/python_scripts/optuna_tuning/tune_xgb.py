import xgboost as xgb

import gc
import os

import wandb

import optuna

from src.python_scripts.utils.data_utils import parse_common_args, prepare_datasets
from src.python_scripts.utils.train_utils import run_xgboost_training
from src.python_scripts.utils.optuna_utils import process_optuna_results

def objective(trial, ds_train, ds_val, Y_val, seq_len, has_context, save_dir):
    # --- HYPERPARAMETER SPACE ---
    params = {
        'objective': 'reg:squarederror',    
        'eval_metric': 'rmse',  # metric_name
        'tree_method': 'hist',  # Required for GPU
        'device': 'cuda',       # Minerva L4 GPU
        'max_bin': 192,         # Optimize VRAM usage

        # --- Optuna Parameter Suggestions ---
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True), # min_split_loss
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # 'max_delta_step' # only when patient context is included
        }
    
    # Initialize W&B
    run_config = params.copy()
    run_config.update({'seq_len': seq_len, 'has_context': has_context})

    run = wandb.init(
        project='bis-prediction-optuna-tuning',
        group=f'xgb-sq-{seq_len}-b-{params["max_bin"]}-pc-{has_context}',
        name=f'trial_{trial.number}',
        config=run_config,
        reinit=True # Allows multiple runs in the same script
    )

    try:
        best_rmse = run_xgboost_training(
            params=params, ds_train=ds_train, ds_val=ds_val, Y_val=Y_val,
            trial=trial, save_dir=save_dir
        )
        return best_rmse
    finally:
        wandb.finish()


def main():
    args = parse_common_args(description="Run XGBoost Tuning")
    
    # Set is_pytorch=False to get numpy arrays instead of Dataset objects
    X_train, Y_train, X_val, Y_val = prepare_datasets(
        seq_len=args.seq_len, 
        has_context=args.p_context, 
        is_pytorch=False
    )
    
    ds_train = xgb.DMatrix(X_train, label=Y_train)
    ds_val = xgb.DMatrix(X_val, label=Y_val)

    del X_train, Y_train, X_val
    gc.collect()

    # --- OPTUNA OPTIMIZATION ---
    study = optuna.create_study(direction="minimize")
    save_dir = f'checkpoints/xgb/seq{args.seq_len}_pc{args.p_context}'
    os.makedirs(save_dir, exist_ok=True)

    study.optimize(
        lambda trial: objective(trial, ds_train, ds_val, Y_val, args.seq_len, args.p_context, save_dir),
        n_trials=50,
    )

    process_optuna_results(
        study=study, 
        save_dir=save_dir, 
        seq_len=args.seq_len, 
        has_context=args.p_context, 
        file_extension='.ubj'
    )

if __name__ == "__main__":
    main()