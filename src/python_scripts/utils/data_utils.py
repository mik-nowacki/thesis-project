import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.python_scripts.datasets.patient_window_dataset import CatContextDataset, SeparateContextDataset

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
        train_ctx = CatContextDataset.preprocess_context(
            cases_df.loc[train_ids], ctx_scaler, is_training=True
        )
        val_ctx = CatContextDataset.preprocess_context(
            cases_df.loc[val_ids], ctx_scaler, is_training=False
        )
        val_ctx = val_ctx.reindex(columns=train_ctx.columns, fill_value=0.0)
    else:
        train_ctx = val_ctx = None

    # --- DATASET INITIALIZATION ---
    train_set = CatContextDataset(
        input_dir=INPUT_DIR, case_ids=train_ids, seq_len=seq_len,
        scaler=scaler, is_training=True, context_df=train_ctx
    )

    val_set = CatContextDataset(
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


def prepare_datasets_trans(seq_len, has_context):
    """
    Loads metadata, splits IDs, handles scaling, and initializes datasets.
    """

    INPUT_DIR = 'data/processed/patient_dataset'
    CASES_FILE = 'data/processed/train_cases.csv'

    # --- METADATA LOADING ---
    cases_df = pd.read_csv(CASES_FILE, index_col='caseid')
    all_ids = cases_df.index.tolist()
    train_ids, val_ids = train_test_split(all_ids, test_size=0.176, random_state=2026)

    # --- SCALER LOGIC ---
    scaler = RobustScaler()

    if has_context:
        ctx_scaler = RobustScaler()
        
        # Split context by train/val ids before scaling
        train_ctx, fill_values = preprocess_context_transformer(
            cases_df.loc[train_ids], ctx_scaler, is_training=True
        )
        val_ctx, _ = preprocess_context_transformer(
            cases_df.loc[val_ids], ctx_scaler, is_training=False, fill_values=fill_values
        )
    else:
        train_ctx = val_ctx = None

    # --- DATASET INITIALIZATION ---
    train_set = SeparateContextDataset(
        input_dir=INPUT_DIR, case_ids=train_ids, seq_len=seq_len,
        scaler=scaler, is_training=True, context_df=train_ctx
    )

    val_set = SeparateContextDataset(
        input_dir=INPUT_DIR, case_ids=val_ids, seq_len=seq_len,
        scaler=scaler, is_training=False, context_df=val_ctx
    )

    # PyTorch requires the Dataset objects
    return train_set, val_set


def preprocess_context_transformer(
    context_df: pd.DataFrame,
    scaler=None,
    is_training: bool = True,
    fill_values: pd.Series = None
    ) -> pd.DataFrame:
    """
    Encodes and normalises the patient-level context CSV so every column
    is numeric and ready to be appended to the feature matrix.
    Expects caseid to be the index.
    """

    df = context_df.copy()

    # Binary encode sex
    df['sex'] = df['sex'].map({'F': 0, 'M': 1})

    # One-hot encode surgery type
    known_categories = ['Colorectal', 'Biliary/Pancreas', 'Breast', 'Minor resection', 'Thyroid', 'Other']

    df['optype'] = pd.Categorical(df['optype'], categories=known_categories)
    df = pd.get_dummies(df, columns=['optype'], drop_first=False, dtype=float)

    # Boolean columns -> float
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(float)

    # Fill any remaining NaNs with median
    if is_training:
        fill_values = df.median(numeric_only=True)
    df = df.fillna(fill_values)

    # Scale numerical columns
    numerical_cols = ['age', 'bmi', 'asa', 'preop_hb', 'preop_k',
                   'preop_na', 'preop_gluc', 'preop_alb']
    
    # Only scale columns that exist
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    if scaler is not None:
        if is_training:
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df, fill_values