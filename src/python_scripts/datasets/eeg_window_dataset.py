import numpy as np
import torch
from torch.utils.data import Dataset

import os

class EEGWindowDataset(Dataset):
    def __init__(self, input_dir, case_ids, seq_len, scaler=None, is_training=True):
        self.seq_len = seq_len
        self.patient_X = []
        self.patient_Y = []
        self.index_map = []  # Stores tuples of (patient_index, timestep_start)
        
        all_X_for_scaler = []

        patient_idx = 0
        
        for cid in case_ids:
            sample_path = os.path.join(input_dir, f'case_{cid}.pt')
            if not os.path.exists(sample_path):
                continue
                
            data = torch.load(sample_path, weights_only=False)
            
            # Replace NaN in EEG features with 0.0
            x = np.nan_to_num(data['features'].numpy(), nan=0.0) 
            y = data['bis'].numpy()
            
            if x.shape[0] <= seq_len:
                continue
            
            self.patient_X.append(x)
            self.patient_Y.append(y)
            
            if is_training:
                all_X_for_scaler.append(x)
            
            # Pre-calculate valid indices
            # only record the start index if the target Y at the END of the window is NOT NaN
            for start_t in range(x.shape[0] - seq_len):
                target_y = y[start_t + seq_len]
                if not np.isnan(target_y):
                    self.index_map.append((patient_idx, start_t))
                    
            patient_idx += 1

        # Fit and Apply Scaler on flat 2D data
        if is_training and scaler is not None:
            stacked_X = np.vstack(all_X_for_scaler)
            scaler.fit(stacked_X)
            
        if scaler is not None:
            for i in range(len(self.patient_X)):
                self.patient_X[i] = scaler.transform(self.patient_X[i])
                
        # Convert to tensors once to save CPU time during training
        self.patient_X = [torch.tensor(arr, dtype=torch.float32) for arr in self.patient_X]
        self.patient_Y = [torch.tensor(arr, dtype=torch.float32) for arr in self.patient_Y]
        print(f"Dataset ready. Total valid windows: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # Look up which patient and which timestamp this index corresponds to
        p_idx, start_t = self.index_map[idx]
        
        # Slice the window from the raw data
        X_window = self.patient_X[p_idx][start_t : start_t + self.seq_len]
        
        # Grab the target BIS value at the end of the window
        Y_target = self.patient_Y[p_idx][start_t + self.seq_len]
        
        # Return X and Y (unsqueeze Y to make it shape [1] instead of a scalar)
        return X_window, Y_target.unsqueeze(0)

