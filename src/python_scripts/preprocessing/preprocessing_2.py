import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json

from tqdm import tqdm
import os

import vitaldb
###### Include Login for Vital #####

SAMPLING_RATE = 128 
PADDING = 4
TRACKS = ['BIS/BIS', 'BIS/SQI', 'BIS/EMG']
VITAL_TRACKS = ['Solar8000/ART_MBP', 'Solar8000/NIBP_MBP', 'Solar8000/HR', 'Solar8000/PLETH_SPO2', 'Solar8000/ETCO2']

########################################################################################
#################################### Functions Filter ##################################
########################################################################################


def find_valid_sqi_range(sqi, min_sqi=20, min_run=60):
    """
    Find start and end index where SQI is consistently above threshold.
    min_run: minimum consecutive seconds of valid SQI to define start/end
    """
    valid = sqi > min_sqi

    # Find first index where SQI stays valid for min_run consecutive seconds
    start = None
    for i in range(PADDING, len(valid) - min_run):
        if valid[i:i + min_run].all():
            start = i
            break

    # Find last index where SQI stays valid for min_run consecutive seconds
    end = None
    for i in range(len(valid) - min_run - PADDING, 0, -1):
        if valid[i:i + min_run].all():
            end = i + min_run
            break

    return start, end


def filter_quality_cases(eeg_cases):
    '''
    Filter the EEG based on quality measures. This includes:
    - Start: Start of valid SQI range is not later than 5 minutes after case start to ensure we capture the induction phase
    - BIS: Bispectral Index coverage of at least 95% in the valid range
    - BIS: Range of at least 60 points ensuring unconscious and awake states are represented
    - SQI: Signal Quality Index of at least 75 on average in the valid range
    - SQI: Worst 10% of SQI values above 40 to exclude cases with long periods of poor quality
    - EMG: Electromyography mean value of less than 31 in the valid range to ensure low muscle artifact 
    '''

    excluded = {'no_valid_range': 0, 'error': 0, 'late_start': 0, 'BIS_coverage': 0,
                'BIS_range': 0, 'SQI_mean': 0, 'SQI_worst_10': 0, 'EMG_mean': 0,
                'art_coverage': 0, 'hr_coverage': 0, 'spo2_coverage': 0, 'etco2_coverage': 0}
    
    tresholds = {
        'start': 5*60,
        'bis_cov': 0.95,
        'bis_range': 60,
        'sqi_mean': 75,
        'sqi_p10': 40,
        'emg_mean': 31,
        'art_cov': 0.95,
        'hr_cov': 0.95,
        'spo2_cov': 0.95,
        'etco2_cov': 0.88
    }

    quality_cases = {}
    

    for caseid in tqdm(eeg_cases, desc="Filtering quality cases"):

        try: 
            data = vitaldb.load_case(caseid, TRACKS, interval=1)
        

            if data is None or len(data) == 0:
                excluded['error'] += 1
                continue
        
            start, end = find_valid_sqi_range(data[:,1])

            if start is None or end is None:
                excluded['no_valid_range'] += 1
                continue

            # Exclude cases where valid SQI range starts later than 5 minutes after case start
            if start > tresholds['start']:
                excluded['late_start'] += 1
                continue

            bis  = data[start:end, 0]
            sqi  = data[start:end, 1]
            emg  = data[start:end, 2]

            # Check BIS coverage
            bis_coverage = np.mean(~np.isnan(bis))
            if bis_coverage < tresholds['bis_cov']:
                excluded['BIS_coverage'] += 1
                continue

            # Check BIS range
            bis_range = np.nanmax(bis) - np.nanmin(bis)
            if bis_range < tresholds['bis_range']:
                excluded['BIS_range'] += 1
                continue

            # Check SQI average
            sqi_mean = np.nanmean(sqi)
            if sqi_mean < tresholds['sqi_mean']:
                excluded['SQI_mean'] += 1
                continue

            #Check SQI worst 10% to exclude cases with long periods of very poor quality
            sqi_p10 = np.nanpercentile(sqi, 10)
            if sqi_p10 < tresholds['sqi_p10']:
                excluded['SQI_worst_10'] += 1
                continue

            # Check EMG average
            emg_mean = np.nanmean(emg)
            if emg_mean > tresholds['emg_mean']:
                excluded['EMG_mean'] += 1
                continue
        
        except Exception as e:
            excluded['error'] += 1
            continue
        
        try: 
            data_vitals = vitaldb.load_case(caseid, VITAL_TRACKS, interval=2)

            if data_vitals is None or len(data_vitals) == 0:
                excluded['error'] += 1
                continue

            # Check arterial pressure coverage
            art_invasive = data_vitals[start//2:end//2, 0]
            art_nibp     = data_vitals[start//2:end//2, 1]
            cov_inv  = np.mean(~np.isnan(art_invasive))
            cov_nibp = np.mean(~np.isnan(art_nibp))
            art_coverage  = max(cov_inv, cov_nibp)

            if art_coverage < tresholds['art_cov']:
                excluded['art_coverage'] += 1
                continue

            # Check other vital signs coverage
            hr_coverage   = np.mean(~np.isnan(data_vitals[start//2:end//2, 2]))
            if hr_coverage < tresholds['hr_cov']:
                excluded['hr_coverage'] += 1
                continue
            
            spo2_coverage = np.mean(~np.isnan(data_vitals[start//2:end//2, 3]))
            if spo2_coverage < tresholds['spo2_cov']:
                excluded['spo2_coverage'] += 1
                continue
            
            etco2_coverage= np.mean(~np.isnan(data_vitals[start//2:end//2, 4]))
            if etco2_coverage < tresholds['etco2_cov']:
                excluded['etco2_coverage'] += 1
                continue

            quality_cases[caseid] = (start, end)

        except Exception as e:
            excluded['error'] += 1
            continue

    df = pd.DataFrame({
        'caseid': list(quality_cases.keys()),
        'start': [quality_cases[caseid][0] for caseid in quality_cases],
        'end': [quality_cases[caseid][1] for caseid in quality_cases]
        })

    return df, excluded

##################################################################################################
################################# Step 2: Filter on EEG quality ##################################
##################################################################################################

df_trks = pd.read_csv("https://api.vitaldb.net/trks")

with open("filtered_caseids.json", "r") as f:
    filtered_caseids = json.load(f)

required_tracks = ['BIS/EEG1_WAV', 'BIS/BIS', 'BIS/SQI', 'BIS/EMG']

eeg_caseids = df_trks[df_trks['tname'].isin(required_tracks)] \
    .groupby('caseid')['tname'] \
    .apply(set) \
    .pipe(lambda x: x[x.apply(lambda tracks: set(required_tracks).issubset(tracks))]) \
    .index

eeg_caseids = eeg_caseids[eeg_caseids.isin(filtered_caseids)]

print(f'Number of cases with EEG tracks and quality measures: {len(eeg_caseids)}')

quality_cases, excluded = filter_quality_cases(eeg_caseids)

total = len(eeg_caseids)
print("\n=== Quality filtering ===")
for reason, count in excluded.items():
    print(f"  {reason}: {count} ({count/total*100:.1f}%)")
print(f"  Passed: {len(quality_cases)} / {total} ({len(quality_cases)/total*100:.1f}%)")

quality_cases.to_csv("high_quality_cases.csv", index=False)