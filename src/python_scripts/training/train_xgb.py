import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import wandb
from wandb.integration.xgboost import WandbCallback

# Import your custom functions from the new dataset module
from python_scripts.datasets.dataset_xgb import load_pt_samples, extract_xgboost_features

def main():
    # Configuration
    INPUT_DIR = 'data/processed/eeg'
    CASES_FILE = 'data/processed/cases_data.csv'
    SEQ_LEN = 96 # Updated to 96 to match your transformer config!

    # A. Load Patient IDs and Split
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()

    train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    print(f"Training on {len(train_ids)} patients, Testing on {len(test_ids)} patients.\n")

    # B. Load 3D Sequences
    print("Loading Training Data...")
    X_train_3d, Y_train = load_pt_samples(INPUT_DIR, train_ids, SEQ_LEN)

    print("Loading Testing Data...")
    X_test_3d, Y_test = load_pt_samples(INPUT_DIR, test_ids, SEQ_LEN)

    # C. Convert to 2D Statistical Features
    X_train_2d = extract_xgboost_features(X_train_3d)
    X_test_2d  = extract_xgboost_features(X_test_3d)

    # Free up RAM by deleting the massive 3D arrays immediately
    del X_train_3d
    del X_test_3d

    # C.5 CLEAN TARGET VARIABLES (Remove NaNs from Labels)
    valid_train_mask = ~np.isnan(Y_train).flatten()
    valid_test_mask  = ~np.isnan(Y_test).flatten()

    X_train_clean = X_train_2d[valid_train_mask]
    Y_train_clean = Y_train[valid_train_mask]

    X_test_clean = X_test_2d[valid_test_mask]
    Y_test_clean = Y_test[valid_test_mask]

    print(f"Clean Training Matrix Shape: {X_train_clean.shape}")

    # D. Prepare XGBoost Data Structures
    dtrain = xgb.DMatrix(X_train_clean, label=Y_train_clean)
    dtest  = xgb.DMatrix(X_test_clean,  label=Y_test_clean)

    # E. Define Model Parameters (NOW WITH GPU ACCELERATION)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',  # Required for GPU training
        'device': 'cuda'        # Tells XGBoost to use the Minerva L4 GPU
    }

    # F. Initialize W&B
    wandb.init(
        project="eeg-bis-prediction",
        config=params,
        name="xgb-gpu-baseline"
    )

    # G. Train the Model
    print("Training XGBoost Model on GPU...")
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    xgb_model = xgb.train(
        params=params, 
        dtrain=dtrain, 
        num_boost_round=500, 
        evals=evals, 
        early_stopping_rounds=20,
        verbose_eval=50,
        callbacks=[WandbCallback()] 
    )

    # H. Final Evaluation
    predictions = xgb_model.predict(dtest)
    mae = mean_absolute_error(Y_test_clean, predictions)
    rmse = np.sqrt(mean_squared_error(Y_test_clean, predictions))

    print("\n--- FINAL RESULTS ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} BIS points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} BIS points")

    wandb.log({
        "final_test_mae": mae,
        "final_test_rmse": rmse
    })

    wandb.finish()

if __name__ == "__main__":
    main()