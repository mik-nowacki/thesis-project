import os
import torch
import numpy as np

def load_pt_samples(input_dir, case_ids, seq_len):
    """Loads 3D sliding windows directly from .pt files for specific patients."""
    X_list = []
    Y_list = []

    for cid in case_ids:
        sample_path = os.path.join(input_dir, f'case_{cid}.pt')
        if not os.path.exists(sample_path):
            continue
            
        data = torch.load(sample_path, weights_only=False)
        x = data['features'].numpy()
        y = data['bis'].numpy()
            
        num_samples = x.shape[0]
        if num_samples <= seq_len:
            continue
            
        # Create sliding windows per patient
        X_case = np.array([x[i:i+seq_len] for i in range(num_samples - seq_len)])
        Y_case = np.array([y[i+seq_len] for i in range(num_samples - seq_len)]).reshape(-1, 1)
        
        X_list.append(X_case)
        Y_list.append(Y_case)

    if not X_list:
        return np.array([]), np.array([])

    # Combine all patients
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    
    return X, Y