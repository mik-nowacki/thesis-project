import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.python_scripts.datasets.patient_window_dataset import PatientWindowDataset

def parse_common_args(description="Run Model Training/Tuning"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seq_len', type=int, default=100, help="Length of the sliding window")
    parser.add_argument('--p_context', action='store_true', help="Include patient context")
    return parser.parse_args()

def prepare_datasets(seq_len, has_context, is_pytorch=True):
    """
    Loads metadata, splits IDs, handles scaling, and initializes datasets.
    Set is_pytorch=False for XGBoost (skips scaling and returns numpy arrays).
    """
    INPUT_DIR = 'data/processed/patient_dataset'
    CASES_FILE = 'data/processed/train_cases.csv'

    # --- METADATA LOADING ---
    cases_df = pd.read_csv(CASES_FILE, index_col='caseid')
    all_ids = cases_df.index.tolist()
    train_ids, val_ids = train_test_split(all_ids, test_size=0.176, random_state=2026)

    # --- SCALER LOGIC ---
    scaler = RobustScaler() if is_pytorch else None

    if has_context:
        ctx_scaler = RobustScaler() if is_pytorch else None
        
        # Split context by train/val ids before scaling
        train_ctx = PatientWindowDataset.preprocess_context(
            cases_df.loc[train_ids], ctx_scaler, is_training=True
        )
        val_ctx = PatientWindowDataset.preprocess_context(
            cases_df.loc[val_ids], ctx_scaler, is_training=False
        )
        val_ctx = val_ctx.reindex(columns=train_ctx.columns, fill_value=0.0)
    else:
        train_ctx = val_ctx = None

    # --- DATASET INITIALIZATION ---
    train_set = PatientWindowDataset(
        input_dir=INPUT_DIR, case_ids=train_ids, seq_len=seq_len,
        scaler=scaler, is_training=True, context_df=train_ctx
    )

    val_set = PatientWindowDataset(
        input_dir=INPUT_DIR, case_ids=val_ids, seq_len=seq_len,
        scaler=scaler, is_training=False, context_df=val_ctx
    )

    # --- RETURN LOGIC ---
    if not is_pytorch:
        # XGBoost requires raw numpy arrays
        X_train, Y_train = train_set.to_numpy()
        X_val, Y_val = val_set.to_numpy()
        return X_train, Y_train, X_val, Y_val

    # PyTorch requires the Dataset objects
    return train_set, val_set