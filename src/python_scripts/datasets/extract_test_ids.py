import pandas as pd
from sklearn.model_selection import train_test_split 

def main():
    
    CASES_FILE = 'data/processed/cases_data.csv'
    SAVE_DIRECTORY = 'data/processed'
    
    cases_master = pd.read_csv(CASES_FILE)
    all_ids = cases_master['caseid'].tolist()

    train_ids, test_ids = train_test_split(all_ids, test_size=0.15, random_state=2026)

    train_df = cases_master[cases_master['caseid'].isin(train_ids)]    
    test_df = cases_master[cases_master['caseid'].isin(test_ids)]

    train_df.to_csv(f'{SAVE_DIRECTORY}/train_cases.csv', index=False)
    test_df.to_csv(f'{SAVE_DIRECTORY}/test_cases.csv', index=False)

if __name__ == "__main__":
    main()