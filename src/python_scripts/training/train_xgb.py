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

from src.python_scripts.datasets.patient_window_dataset import PatientWindowDataset

def train_xgb(ds_train, ds_val, Y_val, seq_len, save_dir):
    # --- HYPERPARAMETER SPACE ---
    params = {
        'objective': 'reg:squarederror',    
        'eval_metric': 'rmse',  # metric_name
        'tree_method': 'hist',  # Required for GPU
        'device': 'cuda',       # Minerva L4 GPU
        'max_bin': 256,         # Optimize VRAM usage
        'learning_rate': 0.11220455829120202,
        'gamma': 0.000309994165063507, # min_split_loss
        'max_depth': 5,
        'min_child_weight': 10,
        'subsample': 0.7124655669610912,
        'colsample_bytree': 0.6825427370519438,
        # 'max_delta_step' # only when patient context is included
        }
    
    # Initialize W&B
    run = wandb.init(
        project='bis-prediction-training',
        group=f'xgb',
        name=f'xgb_{seq_len}_d_{params["max_depth"]}_bin_{params["max_bin"]}', 
        config=params,
        reinit=True # Allows multiple runs in the same script
    )
    wandb_callback = WandbCallback()

    evals_list = [(ds_train, 'train'), (ds_val, 'val')] # dataset_name is 'val' here 

    # try:
    # --- TRAINING ---
    xgb_model = xgb.train(
            params=params, 
            dtrain=ds_train, 
            num_boost_round=600, 
            evals=evals_list,
            early_stopping_rounds=20,
            verbose_eval=0, # messages are turned off
            callbacks=[wandb_callback] 
        )
    
    # Evaluate the best iteration
    y_pred = xgb_model.predict(ds_val)
    val_mse = mean_squared_error(Y_val, y_pred)
    val_rmse = root_mean_squared_error(Y_val, y_pred)
    val_r2 = r2_score(Y_val, y_pred)
    wandb.log({
            "val_mse": val_mse, 
            "val_rmse": val_rmse,
            "val_r2": val_r2
            })
    wandb.finish()
    
    # xgb_model.save_model(f'{save_dir}/trial_{trial.number}_best.ubj')
    # trial.set_user_attr('best_iteration', xgb_model.best_iteration)

    # Free memory before next trial
    # del xgb_model, predictions, abs_errors    
    # gc.collect()

    # Optuna optimizes based on the returned value


    # except RuntimeError as e:
    #     # Catch PyTorch GPU Out of Memory errors
    #     if "out of memory" in str(e).lower():
    #         print(f"\n[Warning] Trial {trial.number} failed due to GPU OOM. Pruning...")
    #         wandb.log({"error": "GPU Out of Memory"})
    #         raise e
    #     else:
    #         raise e

    # finally:
    #     if 'xgb_model' in locals():
    #         del xgb_model
    #     if 'predictions' in locals():
    #         del predictions
    #     wandb.finish()
    #     gc.collect()


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
    cases_df = pd.read_csv(CASES_FILE)
    all_ids = cases_df['caseid'].tolist()

    train_ids, val_ids = train_test_split(all_ids, test_size=0.176, random_state=2026) # 0.176 so it keeps 70/15/15

    # --- DATA LOADING ---
    train_set = PatientWindowDataset(
        INPUT_DIR,
        train_ids,
        SEQ_LEN,
        is_training=True,
        # context_df=cases_df
    )
    X_train, Y_train = train_set.to_numpy()

    val_set = PatientWindowDataset(
        INPUT_DIR,
        val_ids,
        SEQ_LEN,
        is_training=False,
        # context_df=cases_df
    )
    X_val, Y_val = val_set.to_numpy()

    print(f"Clean Training Matrix Shape: {X_train.shape}")
    # Prepare XGBoost Data Structures
    ds_train = xgb.DMatrix(X_train, label=Y_train)
    ds_val = xgb.DMatrix(X_val, label=Y_val)

    save_dir = f'checkpoints/xgb/seq{SEQ_LEN}'
    os.makedirs(save_dir, exist_ok=True)

    train_xgb(ds_train, ds_val, Y_val, SEQ_LEN, save_dir)



    # print("\n--- OPTIMIZATION FINISHED ---")
    # print(f"Number of finished trials: {len(study.trials)}")
    # print(f"Best RMSE: {best_trial.value:.4f}")
    # print("Best hyperparameters:")
    # for key, value in best_trial.params.items():
    #     print(f"    {key}: {value}")

    # # Save study metadata
    # study_results = {
    #     'best_trial': best_trial.number,
    #     'best_val_rmse': best_trial.value,
    #     'best_iteration': best_trial.user_attrs['best_iteration'],
    #     'params': best_trial.params,
    #     'seq_len': SEQ_LEN,
    # }
    # with open(f'{save_dir}/study_results.json', 'w') as f:
    #     json.dump(study_results, f, indent=2)

if __name__ == "__main__":
    main()