import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import vitaldb as v

def load_train_data(path, batch_size, seq_len, device):
    # read patients
    track_names = [
                "SNUADC/ART",        # Arterial pressure wave  | W/500 | mmHg
                "SNUADC/ECG_II",     # ECG lead II wave        | W/500 | mV
                "SNUADC/ECG_V5",     # ECG lead V5 wave        | W/500 | mV
                "BIS/EEG1_WAV",      # EEG wave from channel 1 | W/128 | uV
                "BIS/EEG2_WAV",      # EEG wave from channel 2 | W/128 | uV
                "Solar8000/RR_CO2",  # Respiratory rate based on capnography | N | /min
                "Primus/CO2",        # Capnography wave        | W/62.5 | mmHg
                "BIS/BIS",           # Bispectral index value  |    N   | unitless
                ]

    patients = []

    # train only on 1 patients for now
    p = v.read_vital(path+f'0001.vital', track_names) # ?????
    p = p.to_pandas(track_names=track_names, interval=5)
    p.columns = ['arterial_pres', 'ecg1', 'ecg2', 'eeg1', 'eeg2', 'resp_rate', 'capnography', 'bis']
    p = p[['eeg1', 'eeg2', 'bis']].dropna()
    p = p[p['bis'] > 0]
    patients.append(p)

    # normalize data
    x1 = patients[0].eeg1
    x1_mean = np.mean(x1)
    x1_std = np.std(x1)
    x1_z = [(x1_i - x1_mean)/x1_std for x1_i in x1]

    x2 = patients[0].eeg2
    x2_mean = np.mean(x2)
    x2_std = np.std(x2)
    x2_z = [(x2_i - x2_mean)/x2_std for x2_i in x2]

    x_z = np.stack([x1_z, x2_z], axis=1) # [[x1[0],x2[0]], [x1[1], x2[1]]...]
    X = np.array([x_z[i:i+seq_len] for i in range(x_z.shape[0]-seq_len)]).reshape(-1, seq_len, 2)

    y = patients[0].bis
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_z = [(y_i - y_mean)/y_std for y_i in y]
    Y = np.array([y_z[i+seq_len] for i in range(len(y_z)-seq_len)]).reshape(-1, 1)

    print(f"x_z: mean {np.mean(x_z)} std {np.std(x_z)}  y_z: mean {np.mean(y_z)} std{np.std(y_z)}")
    print(f"X: {X.shape}, Y: {Y.shape}")

    stats = {
        "y_mean": y_mean,
        "y_std": y_std
    }

    # Dataloader for easy data manipulation
    # TensorDataset just pairs X and Y together
    # DataLoader handles batching and shuffling
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).to(device), 
                            torch.tensor(Y, dtype=torch.float32).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader, stats 

def load_test_data(path, batch_size, seq_len, device, stats):
    pass


# -------------- EEG PREPROCESSED SPECTRA --------------
def load_eeg_samples(path, seq_len, features):
    df = pd.read_csv(path)
    x = df[features].to_numpy()
    X = np.array([x[i:i+seq_len] for i in range(x.shape[0] - seq_len)])
    X = X.reshape(-1, seq_len, len(features))

    y = df.bis
    Y = np.array([y[i+seq_len] for i in range(len(y)-seq_len)]).reshape(-1, 1)

    return X, Y

def load_training_eeg_samples(path, split, seq_len, features):
    X_train, Y_train = load_eeg_samples(path, seq_len, features)
    return X_train[:split], Y_train[:split]

def load_test_eeg_samples(path, split, seq_len, features):
    X_test, Y_test = load_eeg_samples(path, seq_len, features)
    return X_test[split:], Y_test[split:]

