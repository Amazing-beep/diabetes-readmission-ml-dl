"""
data_preprocessing.py
---------------------
Data loading, verification, target definition, clinical exclusions,
and patient-grouped train/validation/test splitting for the UCI Diabetes
130-US Hospitals dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def load_raw_data(data_path):
    """
    Load raw diabetic_data.csv and IDS_mapping.csv.
    Handles '?' encodings cleanly without losing clinical 'None' test markers.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    
    # keep_default_na=False prevents strings like 'None' (used in max_glu_serum and A1Cresult)
    # from being prematurely coerced to generic NaNs.
    df = pd.read_csv(data_path, keep_default_na=False)
    return df


def audit_raw_data(df):
    """
    Perform a comprehensive empirical verification of raw dataset dimensions,
    unique patient counts, target distributions, and missing value markers.
    """
    n_rows, n_cols = df.shape
    unique_pts = df['patient_nbr'].nunique()
    multi_enc_pts = (df['patient_nbr'].value_counts() > 1).sum()
    
    target_counts = df['readmitted'].value_counts().to_dict()
    target_pcts = (df['readmitted'].value_counts(normalize=True) * 100).to_dict()
    
    # Missing '?' counts
    missing_dict = {}
    for col in df.columns:
        q_cnt = (df[col] == '?').sum()
        if q_cnt > 0:
            missing_dict[col] = (q_cnt, float(q_cnt / n_rows * 100))
            
    # Expired encounter count (discharge disposition 11, 19, 20, 21)
    df_disp = df['discharge_disposition_id'].astype(int)
    expired_cnt = df_disp.isin([11, 19, 20, 21]).sum()
    
    audit_results = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'unique_patients': unique_pts,
        'repeat_patients': multi_enc_pts,
        'target_distribution': target_counts,
        'target_percentages': target_pcts,
        'missing_question_marks': missing_dict,
        'expired_encounters': expired_cnt
    }
    return audit_results


def apply_target_and_exclusions(df):
    """
    Define binary target:
      1 (Positive): readmitted within 30 days ('<30')
      0 (Negative): no readmission within 30 days ('NO' or '>30')
      
    Clinical exclusions:
      Exclude patients who expired during encounter (IDs 11, 19, 20, 21),
      as 100% have label 'NO' and deceased patients cannot be readmitted.
    """
    df_clean = df.copy()
    df_clean['discharge_disposition_id'] = df_clean['discharge_disposition_id'].astype(int)
    
    # Exclude expired
    expired_ids = [11, 19, 20, 21]
    df_clean = df_clean[~df_clean['discharge_disposition_id'].isin(expired_ids)].copy()
    
    # Define binary target
    df_clean['target'] = (df_clean['readmitted'] == '<30').astype(int)
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def patient_grouped_split(df_clean, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split dataset into Train, Validation, and Test sets strictly by patient_nbr
    using GroupShuffleSplit. Guarantees 0 patient overlap across all splits.
    
    Target proportions:
      Train: ~70%
      Validation: ~15%
      Test: ~15%
    """
    # Split 1: (Train + Val) vs Test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(df_clean, df_clean['target'], groups=df_clean['patient_nbr']))
    
    train_val_df = df_clean.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df_clean.iloc[test_idx].copy().reset_index(drop=True)
    
    # Split 2: Train vs Val from Train_Val
    # val_fraction_of_train_val = val_size / (1.0 - test_size) = 0.15 / 0.85 =~ 0.17647
    rel_val_size = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_df, train_val_df['target'], groups=train_val_df['patient_nbr']))
    
    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df
