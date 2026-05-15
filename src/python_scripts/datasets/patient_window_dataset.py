import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PatientWindowDataset(Dataset):
    """
    Unified sliding-window dataset for intraoperative monitoring data.

    Works as a standard PyTorch Dataset for LSTM / iTransformer, and can
    materialise all windows as flat numpy arrays for XGBoost via .to_numpy().

    Parameters
    ----------
    input_dir    : directory containing case_<id>.pt files
    case_ids     : list of integer case IDs to load
    seq_len      : sliding-window length (number of timesteps)
    scaler       : sklearn scaler instance (optional)
    is_training  : if True, fits the scaler; otherwise only transforms
    context_df   : preprocessed patient-context DataFrame, indexed by caseid.
                   When provided, context features are available per window.
    """
    def __init__(
        self,
        input_dir: str,
        case_ids: list,
        seq_len: int,
        scaler = None,
        is_training: bool = True,
        context_df: pd.DataFrame | None = None
    ):

        self.seq_len = seq_len
        self.has_context = context_df is not None

        self.patient_X: list[np.ndarray] = [] # raw unscaled features
        self.patient_Y: list[np.ndarray] = [] # bis values
        self.patient_ctx: list[np.ndarray] = [] # per-patient context
        self.index_map: list[tuple[int,int]] = [] # (patient_idx, window_start_t)

        all_X_for_scaler: list[np.ndarray] = []
        patient_idx = 0

        for cid in case_ids:
            sample_path = os.path.join(input_dir, f'case_{cid}.pt')
            if not os.path.exists(sample_path):
                  continue
            
            data = torch.load(sample_path, weights_only=False)
            x = np.nan_to_num(data['features'].numpy(), nan=0.0) # (T, n_features)
            y = data['bis'].numpy()                              # (T,)

            if x.shape[0] <= seq_len:
                  continue
            
            self.patient_X.append(x)
            self.patient_Y.append(y)

            # store content for the patient (empty if not provided)
            if self.has_context and cid in context_df.index:
                  self.patient_ctx.append(context_df.loc[cid].values.astype(np.float32))
            else:
                 self.patient_ctx.append(np.array([], dtype=np.float32))

            if is_training and scaler is not None:
                 all_X_for_scaler.append(x)
            
            # Only index whose target Y is not NaN
            for start_t in range(x.shape[0] - seq_len):
                 if not np.isnan(y[start_t + seq_len]):
                      self.index_map.append((patient_idx, start_t))

            patient_idx+=1

        # Fit scaler on all data 
        if is_training and scaler is not None and all_X_for_scaler:
                scaler.fit(np.vstack(all_X_for_scaler))
        
        if scaler is not None:
                self.patient_X = [scaler.transform(x) for x in self.patient_X]
            
        # Convert time-series to tensors once to save CPU time per batch
        self.patient_X = [torch.tensor(x, dtype=torch.float32) for x in self.patient_X]
        self.patient_Y = [torch.tensor(y, dtype=torch.float32) for y in self.patient_Y]

        self.num_features = self.patient_X[0].shape[-1] if self.patient_X else 0
        self.num_context_features = self.patient_ctx[0].shape[0] if self.patient_ctx else 0

        print(
            f"{'Training set' if is_training else 'Validation set'} ready | "
            f"windows: {len(self.index_map)} | "
            f"ts_features: {self.num_features} | "
            f"ctx_features: {self.num_context_features}"
        )
    
    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
         return len(self.index_map)
    

    def __getitem__(self, idx: int):
        """
        Returns a single window of patient timeseres
        -------
        X_window : (seq_len, n_features) float32 tensor
        context  : (n_context_features,) float32 tensor  — only if context_df was provided
        Y_target : scalar float32 tensor
        """
        p_idx, start_t = self.index_map[idx]
        X_window = self.patient_X[p_idx][start_t : start_t + self.seq_len]
        Y_target = self.patient_Y[p_idx][start_t + self.seq_len]

        if self.has_context:
            # (num_context_features,)
            C_window = torch.tensor(self.patient_ctx[p_idx], dtype=torch.float32) 

            # (seq_len, num_context_features)
            C_window = C_window.unsqueeze(0).expand(self.seq_len, -1) # repeat the static data across the entire window
            
            return X_window, C_window, Y_target

        return X_window, Y_target
    
    
    # ------------------------------------------------------------------
    # XGBoost export
    # ------------------------------------------------------------------

    def to_numpy(self, flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Materialises all valid windows as numpy arrays.

        Parameters
        ----------
        flatten : if True, each window is flattened to 1D and context (if any)
                  is appended → shape (n_windows, seq_len * n_features [+ n_ctx])
                  if False, windows keep shape (n_windows, seq_len, n_features)

        Returns
        -------
        X : numpy array
        Y : (n_windows, 1) numpy array
        """
        X_list, Y_list = [], []

        for p_idx, start_t in self.index_map:
            window = self.patient_X[p_idx][start_t : start_t + self.seq_len].numpy()

            if flatten:
                window = window.flatten()
                if self.has_context and self.patient_ctx[p_idx].shape[0] > 0:
                    window = np.concatenate([window, self.patient_ctx[p_idx]])

            X_list.append(window)
            Y_list.append(self.patient_Y[p_idx][start_t + self.seq_len].item())

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32).reshape(-1, 1)
        return X, Y
    
    @staticmethod
    def preprocess_context(
         context_df: pd.DataFrame,
         scaler=None,
         is_training: bool = True,
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
        df = pd.get_dummies(df, columns=['optype'], drop_first=True, dtype=float)

        # Boolean columns -> float
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(float)

        # Fill any remaining NaNs with median
        df = df.fillna(df.median(numeric_only=True))

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

        return df