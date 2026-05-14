import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

import json
import argparse
import gc
import os

import wandb
from wandb.integration.xgboost import WandbCallback

import optuna
from optuna.integration import XGBoostPruningCallback

from src.python_scripts.datasets.dataset_xgb import load_pt_samples

def objective(trial, ds_train, ds_val, Y_val_clean, seq_len, save_dir):
    # --- HYPERPARAMETER SPACE ---
    params = {
        'objective': 'reg:squarederror',    
        'eval_metric': 'rmse',  # metric_name
        'tree_method': 'hist',  # Required for GPU
        'device': 'cuda',       # Minerva L4 GPU
        'max_bin': 128,         # Optimize VRAM usage

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
    run_config['seq_len'] = seq_len

    run = wandb.init(
        project='bis-prediction',
        group=f'xgb-seq-{seq_len}-bin-128',
        name=f'trial_{trial.number}',
        config=run_config,
        reinit=True # Allows multiple runs in the same script
    )
    wandb_callback = WandbCallback()
    pruning_callback = XGBoostPruningCallback(trial, "val-rmse") # "dataset_name-metric_name"

    evals_list = [(ds_train, 'train'), (ds_val, 'val')] # dataset_name is 'val' here 

    try:
        # --- TRAINING ---
        xgb_model = xgb.train(
                params=params, 
                dtrain=ds_train, 
                num_boost_round=600, 
                evals=evals_list,
                early_stopping_rounds=20,
                verbose_eval=0, # messages are turned off
                callbacks=[wandb_callback, pruning_callback] 
            )
        
        # Evaluate the best iteration
        predictions = xgb_model.predict(ds_val)
        val_rmse = root_mean_squared_error(Y_val_clean.flatten(), predictions)

        xgb_model.save_model(f'{save_dir}/trial_{trial.number}_best.ubj')
        trial.set_user_attr('best_iteration', xgb_model.best_iteration)

        # Free memory before next trial
        # del xgb_model, predictions, abs_errors    
        gc.collect()

        # Optuna optimizes based on the returned value
        return val_rmse
    

    except RuntimeError as e:
        # Catch PyTorch GPU Out of Memory errors
        if "out of memory" in str(e).lower():
            print(f"\n[Warning] Trial {trial.number} failed due to GPU OOM. Pruning...")
            wandb.log({"error": "GPU Out of Memory"})
            raise optuna.TrialPruned()
        else:
            raise e

    finally:
        if 'xgb_model' in locals():
            del xgb_model
        if 'predictions' in locals():
            del predictions
        wandb.finish()
        gc.collect()


def main():
    # Parse cmd arguments
    parser = argparse.ArgumentParser(description="Run XGBoost Tuning for a specific Sequence Length")
    parser.add_argument('--seq_len', type=int, default=60, help="Length of the sliding window (Default: 60)")
    args = parser.parse_args()
    
    # Configuration
    SEQ_LEN = args.seq_len
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/train_cases.csv'

    # --- INDEX LOADING ---
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()

    train_ids, val_ids = train_test_split(all_ids, test_size=0.176, random_state=2026) # 0.176 so it keeps 70/15/15

    # --- DATA LOADING ---
    X_train_3d, Y_train = load_pt_samples(INPUT_DIR, train_ids, SEQ_LEN)
    X_val_3d, Y_val = load_pt_samples(INPUT_DIR, val_ids, SEQ_LEN)

    X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_val_2d = X_val_3d.reshape(X_val_3d.shape[0], -1)

    # Free up RAM
    del X_train_3d, X_val_3d, train_ids, val_ids
    gc.collect()

    # CLEAN TARGET VARIABLES (Remove NaNs from Labels)
    valid_train_mask = ~np.isnan(Y_train).flatten()
    valid_val_mask = ~np.isnan(Y_val).flatten()

    X_train_clean = X_train_2d[valid_train_mask]
    Y_train_clean = Y_train[valid_train_mask]

    X_val_clean = X_val_2d[valid_val_mask]
    Y_val_clean = Y_val[valid_val_mask]

    del X_train_2d, X_val_2d, valid_train_mask, valid_val_mask
    gc.collect()

    print(f"Clean Training Matrix Shape: {X_train_clean.shape}")
    # Prepare XGBoost Data Structures
    ds_train = xgb.DMatrix(X_train_clean, label=Y_train_clean)
    ds_val = xgb.DMatrix(X_val_clean, label=Y_val_clean)

    # --- OPTUNA OPTIMIZATION ---
    print(f"=== STARTING OPTUNA STUDY FOR SEQ_LEN: {SEQ_LEN} ===")
    study = optuna.create_study(direction="minimize")
    save_dir = f'checkpoints/xgb/seq{SEQ_LEN}'
    os.makedirs(save_dir, exist_ok=True)

    study.optimize(
        lambda trial: objective(trial, ds_train, ds_val, Y_val_clean, SEQ_LEN, save_dir),
        n_trials=50,
    )

    best_trial = study.best_trial

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
        'best_iteration': best_trial.user_attrs['best_iteration'],
        'params': best_trial.params,
        'seq_len': SEQ_LEN,
    }
    with open(f'{save_dir}/study_results.json', 'w') as f:
        json.dump(study_results, f, indent=2)

    # clean up former best trials
    for trial in study.trials:
        path = f'{save_dir}/trial_{trial.number}_best.ubj'
        if trial.number != best_trial.number and os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    main()