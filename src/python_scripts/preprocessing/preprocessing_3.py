import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

from scipy import signal
from scipy import ndimage
from scipy.signal import butter, sosfiltfilt

from tqdm import tqdm
import os

import vitaldb
vitaldb.login("login", "password")

OUTPUT_DIR = 'preprocessed/eeg'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLING_RATE = 128
PADDING = 4 #Padding of 4 second so that the 8 second window can be computed for the start/end indices. 
N_FEATURES = 19 #15 EEG features and 4 vital features (MBP, HR, SpO2, ETCO2)
TRACKS = ['BIS/EEG1_WAV', 'BIS/BIS', 'Solar8000/ART_MBP', 'Solar8000/NIBP_MBP', 'Solar8000/HR', 'Solar8000/PLETH_SPO2', 'Solar8000/ETCO2']

#Create the Frequency Bin Edges
LOW_EDGES  = np.logspace(np.log10(0.5),  np.log10(4),  num=6)
MID_EDGES  = np.logspace(np.log10(4),    np.log10(13), num=6)
HIGH_EDGES = np.logspace(np.log10(13),   np.log10(50), num=6)

EDGES = {
        'low': LOW_EDGES,
        'mid': MID_EDGES,
        'high': HIGH_EDGES
    }

##################################################################################################
################################# Functions EEG Transformation ###################################
##################################################################################################

def preprocess_case(caseid, start, end):

    eeg, vitals, bis = load_crop_eeg(caseid, start, end)

    # Interpolate NaNs before filtering
    eeg = interpolate_nans(eeg)

    # Extract the three EEG branches (low, mid, high) based on the defined frequency bins
    eeg_low = bandpass_filter(eeg, low=0.5, high=4, fs=SAMPLING_RATE)
    eeg_mid = bandpass_filter(eeg, low=4, high=13, fs=SAMPLING_RATE)
    eeg_high = bandpass_filter(eeg, low=13, high=50, fs=SAMPLING_RATE)

    # Create Feature Matrix 
    eeg_features = np.zeros((end-start, N_FEATURES))  #15 EEG features + 4 vitals
    
    for t in range(end-start): #Create one feature vector per second of valid data
        t_eeg = (t+PADDING)*SAMPLING_RATE

        eeg_features[t, 0:5] = compute_branch_power(eeg_low, 'low', t_eeg, EDGES)
        eeg_features[t, 5:10] = compute_branch_power(eeg_mid, 'mid', t_eeg, EDGES)
        eeg_features[t, 10:15] = compute_branch_power(eeg_high, 'high', t_eeg, EDGES)

    #Add vital features (MBP, HR, SpO2, ETCO2)
    eeg_features[:, 15:19] = vitals

    return eeg_features, bis


def load_crop_eeg(caseid, start, end):
    data = vitaldb.load_case(caseid, TRACKS, interval=1/SAMPLING_RATE)
    
    #Divide into EEg, BIS and vitals 
    eeg = data[:, 0]
    bis = data[:, 1]

    start_128 = start*SAMPLING_RATE
    end_128   = end*SAMPLING_RATE

    art_invasive = data[start_128:end_128, 2]
    art_nibp     = data[start_128:end_128, 3]
    cov_inv  = np.mean(~np.isnan(art_invasive))
    cov_nibp = np.mean(~np.isnan(art_nibp))
    
    #
    art_col = 2 if cov_inv >= cov_nibp else 3
    vital_cols = [art_col, 4, 5, 6]
    vitals = data[:, vital_cols]

    #Define cropping indices with padding
    l_crop = (start - PADDING)*SAMPLING_RATE
    r_crop = (end + PADDING)*SAMPLING_RATE

    #Crop the EEG signal
    eeg_cropped = eeg[l_crop:r_crop]

    #Crop BIS signal and resample to 1Hz by taking the first valid value in each second
    bis_cropped = np.zeros(end - start)
    
    for s in range(end - start):
        block = bis[(s + start) * SAMPLING_RATE : (s + start + 1) * SAMPLING_RATE]
        valid = block[~np.isnan(block)]
        bis_cropped[s] = valid[0] if len(valid) > 0 else np.nan

    #Crop vitals and resample to 1Hz by taking a valid value in the next second or interpolating otherwise
    vitals_cropped = np.zeros((end - start, 4))

    for s in range(end - start):
        block = vitals[(s + start) * SAMPLING_RATE : (s + start + 1) * SAMPLING_RATE, :]

        for v in range(4):
            valid = block[~np.isnan(block[:, v]), v]
            vitals_cropped[s, v] = valid[0] if len(valid) > 0 else np.nan

    # Interpolate those NaNs in the vitals that appear due to the 2 second sampling value
    for v in range(4):
        vitals_cropped[:, v] = interpolate_nans(vitals_cropped[:, v], max_gap=3)    

    return eeg_cropped, vitals_cropped, bis_cropped


def interpolate_nans(signal, max_gap=None):
    nans = np.isnan(signal)
    if not nans.any():
        return signal.copy()

    filled  = signal.copy()
    indices = np.arange(len(signal))

    if max_gap is None:
        filled[nans] = np.interp(indices[nans], indices[~nans], signal[~nans])
    else:
        labeled, n_gaps = ndimage.label(nans)
        for i in range(1, n_gaps + 1):
            gap_indices = np.where(labeled == i)[0]
            if len(gap_indices) <= max_gap:
                filled[gap_indices] = np.interp(
                    gap_indices, indices[~nans], signal[~nans]
                )

    return filled


def bandpass_filter(signal, low, high, fs=SAMPLING_RATE, order=4):
    nyquist = fs / 2
    sos = butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def compute_branch_power(signal, branch, t_eeg, edges, fs=SAMPLING_RATE):
    if branch == 'low':
        step = 4 * fs
        l_bound = t_eeg - step
        r_bound = t_eeg + step
        segment = signal[l_bound:r_bound]
        freqs, power = compute_frequency_domain(segment, fs)
        return np.log1p(compute_bin_frequency_domain(freqs, power, edges['low']))
    
    elif branch == 'mid':
        step = 1 * fs
        l_bound = t_eeg - step
        r_bound = t_eeg + step
        segment = signal[l_bound:r_bound]
        freqs, power = compute_frequency_domain(segment, fs)
        return np.log1p(compute_bin_frequency_domain(freqs, power, edges['mid']))
    
    elif branch == 'high':
        step =  fs // 4
        segments = [
            signal[t_eeg - 2*step :t_eeg ],
            signal[t_eeg - step   :t_eeg + step],
            signal[t_eeg          :t_eeg + 2*step]
        ]

        spectrums = []
        for s in segments:
            freqs, power = compute_frequency_domain(s, fs)
            spectrums.append(np.log1p(compute_bin_frequency_domain(freqs, power, edges['high'])))
        
        return np.mean(spectrums, axis=0)
    

def compute_frequency_domain(window, fs=SAMPLING_RATE):
    n = len(window)
    hann = np.hanning(n)
    tap_window = window * hann
    
    fft_values = np.fft.rfft(tap_window)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    power = (np.abs(fft_values)**2)/n  
    #power = (np.abs(fft_values)**2) / (fs * (hann**2).sum()) Maybe more useful? 
    return freqs, power


def compute_bin_frequency_domain(freqs, power, edges):
    n_bins = len(edges) - 1
    bin_power = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = (freqs >= edges[i]) & (freqs < edges[i+1])
        if bin_mask.sum() > 0:
            bin_power[i] = np.mean(power[bin_mask])
        else:
            bin_power[i] = 0.0
    return bin_power

##################################################################################################
###################################### Transform EEG Data ########################################
##################################################################################################

failed_cases = []
skipped_cases = []
successful_cases = []

high_quality_cases = pd.read_csv('high_quality_cases.csv')

for row in tqdm(high_quality_cases.itertuples(), total=len(high_quality_cases), desc="Preprocessing Cases"):
    
    caseid = row.caseid
    start  = row.start
    end    = row.end
    
    output_path = os.path.join(OUTPUT_DIR, f'case_{caseid}.pt')
    
    # Skip if already processed
    if os.path.exists(output_path):
        skipped_cases.append(caseid)
        continue
    
    try:
        features, bis = preprocess_case(caseid, start, end)
        
        # Validate output before saving
        if features is None or bis is None:
            raise ValueError("preprocess_case returned None")
        
        if features.shape[1] != N_FEATURES:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        nan_fraction = np.isnan(features).mean()
        if nan_fraction > 0.1:
            raise ValueError(f"Excessive NaNs in features: {nan_fraction:.1%}")
        
        torch.save({
            'features': torch.tensor(features, dtype=torch.float32),
            'bis':      torch.tensor(bis,      dtype=torch.float32),
            'caseid':   caseid,
        }, output_path)
        
        successful_cases.append(caseid)
        
    except Exception as e:
        failed_cases.append({'caseid': caseid, 'error': str(e)})

# Save failure log
if failed_cases:
    pd.DataFrame(failed_cases).to_csv('preprocessed/failed_cases.csv', index=False)
    print(f"Failed: {len(failed_cases)} cases — see preprocessed/failed_cases.csv")

print(f"Done — successful: {len(successful_cases)}, "
      f"skipped: {len(skipped_cases)}, "
      f"failed: {len(failed_cases)}")


##################################################################################################
########################## Step 4: Create Patient Context Master File ############################
##################################################################################################

def classify_anesthetic(caseids, df_trks):
    # Group once outside the loop
    trks_grouped = df_trks.groupby('caseid')['tname'].apply(set).to_dict()
    
    propofol_list, ether_list, remi_list = [], [], []

    for caseid in caseids:
        case_tracks = trks_grouped.get(caseid, set())
        propofol_list.append(bool(case_tracks & PROPOFOL_TRACKS))
        ether_list.append(bool(case_tracks & VOLATILE_TRACKS))
        remi_list.append(bool(case_tracks & REMI_TRACKS))

    return propofol_list, ether_list, remi_list


PROPOFOL_TRACKS = set([
    'Orchestra/PPF20_VOL', 
])
VOLATILE_TRACKS = set([
    'Primus/INSP_SEVO',
    'Primus/INSP_DES',
])
REMI_TRACKS = set([
    'Orchestra/RFTN20_VOL',
    'Orchestra/RFTN50_VOL',
])


all_cases = skipped_cases + successful_cases
columns = ['caseid', 'optype', 'sex', 'age', 'asa', 'bmi', 'preop_hb', 
           'preop_k', 'preop_na', 'preop_gluc', 'preop_cr', 'preop_alb']

df_cases = pd.read_csv("https://api.vitaldb.net/cases")
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

df_cases = df_cases[df_cases['caseid'].isin(all_cases)]
df_cases = df_cases.loc[:, columns]

propofol, ether, remi = classify_anesthetic(df_cases['caseid'], df_trks)

print("\n=== Anesthetic agent distribution ===")
print(f"  Propofol (TIVA):  {sum(propofol)}")
print(f"  Volatile:         {sum(ether)}")
print(f"  Remifentanil:     {sum(remi)}")
print(f"  TIVA only:        {sum(p and not e for p, e in zip(propofol, ether))}")
print(f"  Volatile only:    {sum(e and not p for p, e in zip(propofol, ether))}")
print(f"  Combined:         {sum(p and e for p, e in zip(propofol, ether))}")
print(f"  Unknown:          {sum(not p and not e for p, e in zip(propofol, ether))}")

df_cases['propofol'] = propofol
df_cases['volatile'] = ether
df_cases['remifentanil'] = remi

df_cases.to_csv('cases_data.csv', index=False)