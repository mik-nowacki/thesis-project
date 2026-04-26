import numpy as np
import pandas as pd

import json

import vitaldb
vitaldb.login("DavidWard1999", "ThisIsMyVitalDB")

########################################################################################
################################# Collect the data set #################################
########################################################################################

# Load the case metadata from VitalDB
df_cases = pd.read_csv("https://api.vitaldb.net/cases")
print(f"Total cases in VitalDB: {len(df_cases)}")

SAMPLING_RATE = 128 
PADDING = 4
MIN_MINUTES = 30
MAX_MINUTES = 120


##################################################################################################
################################# Step 1: Filter on patient data #################################
##################################################################################################

df_filtered = df_cases[
    
    # Anesthesia duration between 30 min and 2 hours. 
    ((df_cases['aneend'] - df_cases['anestart']) >= MIN_MINUTES*60) &
    ((df_cases['aneend'] - df_cases['anestart']) <= MAX_MINUTES*60) &

    # General anesthesia only
    (df_cases['ane_type'] == 'General') &

    # No emergency operations
    (df_cases['emop'] == 0) &

    # ASA 1-4 only
    (df_cases['asa'] <= 4) &
    (df_cases['asa'] >= 1) &

    # Less than 2 ICU days 
    (df_cases['icu_days'] < 2) &

    # Must have key demographics
    (df_cases['age'].notna()) &
    (df_cases['weight'].notna()) &
    (df_cases['bmi'].notna()) &
    (df_cases['sex'].notna()) &

    #Must have preoperational values
    (df_cases['preop_hb'].notna()) &
    (df_cases['preop_k'].notna()) &
    (df_cases['preop_na'].notna()) &
    (df_cases['preop_gluc'].notna()) &
    (df_cases['preop_cr'].notna()) &
    (df_cases['preop_alb'].notna()) &

    # Exclude difficult airways (Cormack grade 3-4)
    (df_cases['cormack'].isna() | df_cases['cormack'].isin(['I', 'II']))
]

total = len(df_cases)

filters = {
    "Anesthesia duration < 30 min":     ~((df_cases['aneend'] - df_cases['anestart']) >= MIN_MINUTES * 60),
    "Anesthesia duration > 120 min":    ~((df_cases['aneend'] - df_cases['anestart']) <= MAX_MINUTES * 60),
    "Not general anesthesia":           df_cases['ane_type'] != 'General',
    "Emergency operation":              df_cases['emop'] != 0,
    "ASA out of range (1-4)":           ~(df_cases['asa'].between(1, 4)),
    "ICU days > 1":                     df_cases['icu_days'] > 1,
    "Missing age/weight/BMI/Sex":        df_cases[['age', 'weight', 'bmi', 'sex']].isna().any(axis=1),
    "Cormack grade 3-4":                ~(df_cases['cormack'].isna() | df_cases['cormack'].isin(['I', 'II'])),
    "Missing Preoperational Values":    df_cases[['preop_hb', 'preop_k', 'preop_na', 'preop_gluc', 'preop_cr', 'preop_alb']].isna().any(axis=1)
}

print("=== Exclusion summary (independent, from full dataset) ===")
for reason, mask in filters.items():
    print(f"  {reason}: {mask.sum()} cases ({mask.sum()/total*100:.1f}%)")

print(f"\n  Total after all filters: {len(df_filtered)} / {total} ({len(df_filtered)/total*100:.1f}% retained)")

#Keep only the caseids of the filtered cases for the next step
filtered_caseids = df_filtered['caseid'].tolist()

with open("filtered_caseids.json", "w") as f:
    json.dump(filtered_caseids, f)