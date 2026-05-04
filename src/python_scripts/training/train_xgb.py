import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import wandb
from wandb.integration.xgboost import WandbCallback

import optuna
from optuna.integration import XGBoostPruningCallback

from src.python_scripts.datasets.dataset_xgb import load_pt_samples, extract_xgboost_features

def objective(trial, dtrain, dtest, Y_test_clean):
    # Define the hyperparameter search space
    params = {
        'objective': 'reg:squarederror',    
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Required for GPU
        'device': 'cuda',       # Minerva L4 GPU

        # --- Optuna Parameter Suggestions ---
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        }
    
    # Initialize W&B
    run = wandb.init(
        project='eeg-bis-prediction',
        group='optuna-tuning-2',
        name=f'trial_{trial.number}',
        config=params,
        reinit=True # Allows multiple runs in the same script
    )

    pruning_callback = XGBoostPruningCallback(trial, "eval-rmse")
    wandb_callback = WandbCallback()

    # Train the model
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    xgb_model = xgb.train(
            params=params, 
            dtrain=dtrain, 
            num_boost_round=500, 
            evals=evals, 
            early_stopping_rounds=20,
            verbose_eval=False, # Turn off console spam for multiple trials
            callbacks=[wandb_callback, pruning_callback] 
        )
    
    # Evaluate the best iteration
    predictions = xgb_model.predict(dtest)
    mae = mean_absolute_error(Y_test_clean, predictions)
    rmse = np.sqrt(mean_squared_error(Y_test_clean, predictions))

    # Log final metrics for this trial
    wandb.log({"final_test_mae": mae, "final_test_rmse": rmse})
    wandb.finish()

    # Optuna optimizes based on the returned value
    return rmse

def main():
    # Configuration
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/cases_data.csv'
    SEQ_LEN = 96 

    # --- DATA LOADING ---
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()

    train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    print(f"Training on {len(train_ids)} patients, Testing on {len(test_ids)} patients.\n")

    print("Loading Training Data...")
    X_train_3d, Y_train = load_pt_samples(INPUT_DIR, train_ids, SEQ_LEN)

    print("Loading Testing Data...")
    X_test_3d, Y_test = load_pt_samples(INPUT_DIR, test_ids, SEQ_LEN)

    # Convert to 2D Statistical Features
    # X_train_2d = extract_xgboost_features(X_train_3d)
    # X_test_2d  = extract_xgboost_features(X_test_3d)

    # simply reshaping the data yields better results
    X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)

    # Free up RAM
    del X_train_3d
    del X_test_3d

    # CLEAN TARGET VARIABLES (Remove NaNs from Labels)
    valid_train_mask = ~np.isnan(Y_train).flatten()
    valid_test_mask  = ~np.isnan(Y_test).flatten()

    X_train_clean = X_train_2d[valid_train_mask]
    Y_train_clean = Y_train[valid_train_mask]

    X_test_clean = X_test_2d[valid_test_mask]
    Y_test_clean = Y_test[valid_test_mask]

    print(f"Clean Training Matrix Shape: {X_train_clean.shape}")

    # Prepare XGBoost Data Structures
    dtrain = xgb.DMatrix(X_train_clean, label=Y_train_clean)
    dtest  = xgb.DMatrix(X_test_clean,  label=Y_test_clean)

    # --- OPTUNA OPTIMIZATION ---
    print("\nStarting Hyperparameter Tuning...")

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(trial, dtrain, dtest, Y_test_clean),
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