import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

import gc

import wandb
from wandb.integration.xgboost import WandbCallback

import optuna
from optuna.integration import XGBoostPruningCallback

from src.python_scripts.datasets.dataset_xgb import load_pt_samples

def objective(trial, dtrain, dtest, Y_test_clean, seq_len):

    # --- HYPERPARAMETER SPACE ---
    params = {
        'objective': 'reg:squarederror',    
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Required for GPU
        'device': 'cuda',       # Minerva L4 GPU
        'max_bin': 64,          # Optimize VRAM usage

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
        project='eeg-bis-prediction',
        group=f'xgb-seq-{seq_len}-tuning-1',
        name=f'trial_{trial.number}',
        config=params,
        reinit=True # Allows multiple runs in the same script
    )
    wandb_callback = WandbCallback()
    pruning_callback = XGBoostPruningCallback(trial, "eval-rmse")

    # --- TRAINING ---
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    xgb_model = xgb.train(
            params=params, 
            dtrain=dtrain, 
            num_boost_round=600, 
            evals=evals, 
            early_stopping_rounds=20,
            verbose_eval=0, # messages are turned off
            callbacks=[wandb_callback, pruning_callback] 
        )
    
    # Evaluate the best iteration
    Y_test_flat = Y_test_clean.flatten() # so it matches predictions shape ( (367597) instead of (367597, 1))
    predictions = xgb_model.predict(dtest)
    mae = mean_absolute_error(Y_test_flat, predictions)
    mse = mean_squared_error(Y_test_flat, predictions)
    rmse = root_mean_squared_error(Y_test_flat, predictions)
    r2 = r2_score(Y_test_flat, predictions)
    
    # custom tolerance accurarcy metric
    abs_errors = np.abs(Y_test_flat - predictions)
    within_tolerance_25 = abs_errors <= 2.5
    tolerance_accuracy_25 = np.mean(within_tolerance_25) * 100

    within_tolerance_5 = abs_errors <= 5.0
    tolerance_accuracy_5 = np.mean(within_tolerance_5) * 100

    within_tolerance_10 = abs_errors <= 10.0
    tolerance_accuracy_10 = np.mean(within_tolerance_10) * 100


    # Log final metrics for this trial
    wandb.log({"test_mae": mae, 
               "test_mse": mse, 
               "test_rmse": rmse,
               "test_r2_score": r2,
               "test_tolerance_accuracy_2_5": tolerance_accuracy_25,
               "test_tolerance_accuracy_5": tolerance_accuracy_5,
               "test_tolerance_accuracy_10": tolerance_accuracy_10,
               })
    wandb.finish()

    # Free memory before next trial
    del xgb_model, predictions, abs_errors    
    gc.collect()

    # Optuna optimizes based on the returned value
    return rmse

def main():
    # Configuration
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/cases_data.csv'
    SEQ_LEN = 120

    # --- INDEX LOADING ---
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()

    train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=2026)


    # --- DATA LOADING ---
    # seq_len = trial.suggest_categorical('seq_len', [1, 30, 60, 90, 120])

    X_train_3d, Y_train = load_pt_samples(INPUT_DIR, train_ids, SEQ_LEN)

    X_test_3d, Y_test = load_pt_samples(INPUT_DIR, test_ids, SEQ_LEN)

    # simply reshaping the data yields better results
    X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)

    # Free up RAM
    del X_train_3d, X_test_3d
    gc.collect()

    # CLEAN TARGET VARIABLES (Remove NaNs from Labels)
    valid_train_mask = ~np.isnan(Y_train).flatten()
    valid_test_mask  = ~np.isnan(Y_test).flatten()

    X_train_clean = X_train_2d[valid_train_mask]
    Y_train_clean = Y_train[valid_train_mask]

    X_test_clean = X_test_2d[valid_test_mask]
    Y_test_clean = Y_test[valid_test_mask]

    del X_train_2d, X_test_2d, valid_train_mask, valid_test_mask
    gc.collect()

    print(f"Clean Training Matrix Shape: {X_train_clean.shape}")
    # Prepare XGBoost Data Structures
    dtrain = xgb.DMatrix(X_train_clean, label=Y_train_clean)
    dtest  = xgb.DMatrix(X_test_clean,  label=Y_test_clean)

    # --- OPTUNA OPTIMIZATION ---
    print("\nStarting Hyperparameter Tuning...")

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(trial, dtrain, dtest, Y_test_clean, SEQ_LEN),
        n_trials=50,
    )

    # --- RESULTS ---
    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best RMSE: {study.best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()