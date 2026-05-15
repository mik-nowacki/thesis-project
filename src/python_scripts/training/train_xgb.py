import xgboost as xgb
import gc
import wandb

from src.python_scripts.utils.data_utils import parse_common_args, prepare_datasets
from src.python_scripts.utils.train_utils import run_xgboost_training

def train_xgb(ds_train, ds_val, Y_val, seq_len, has_context):
    # --- HYPERPARAMETER SPACE ---
    params = {
        'objective': 'reg:squarederror',    
        'eval_metric': 'rmse',  # metric_name
        'tree_method': 'hist',  # Required for GPU
        'device': 'cuda',       # Minerva L4 GPU
        'has_context': has_context,
        'max_bin': 192,         # Optimize VRAM usage
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
        name=f'xgb_{seq_len}_d_{params["max_depth"]}_bin_{params["max_bin"]}_pc_{params["has_context"]}', 
        config=params,
        reinit=True # Allows multiple runs in the same script
    )
    
    try:
        run_xgboost_training(
            params=params, ds_train=ds_train, ds_val=ds_val, Y_val=Y_val
        )
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

    print(f"Clean Training Matrix Shape: {X_train.shape}")
    
    ds_train = xgb.DMatrix(X_train, label=Y_train)
    ds_val = xgb.DMatrix(X_val, label=Y_val)

    del X_train, Y_train, X_val
    gc.collect()

    train_xgb(ds_train, ds_val, Y_val, args.seq_len, args.p_context)


if __name__ == "__main__":
    main()