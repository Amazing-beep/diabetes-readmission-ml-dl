"""
feature_engineering.py
----------------------
Clinical feature engineering, ICD-9 diagnostic grouping, feature selection,
and scikit-learn ColumnTransformer pipelines for traditional ML and Deep Learning.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def parse_icd9_code(code):
    """
    Map ICD-9 diagnosis codes into broad clinical diagnostic categories
    following the established methodology of Strack et al. (2014).
    """
    if pd.isna(code) or code == '?' or code == '' or code is None:
        return 'Missing'
    
    code_str = str(code).strip()
    
    # E and V codes represent external causes and supplementary factors
    if code_str.startswith('V') or code_str.startswith('E'):
        return 'Other'
    
    try:
        val = float(code_str)
    except ValueError:
        return 'Other'
    
    if (390 <= val <= 459) or val == 785:
        return 'Circulatory'
    elif (460 <= val <= 519) or val == 786:
        return 'Respiratory'
    elif (520 <= val <= 579) or val == 787:
        return 'Digestive'
    elif int(val) == 250:
        return 'Diabetes'
    elif 800 <= val <= 999:
        return 'Injury'
    elif 710 <= val <= 739:
        return 'Musculoskeletal'
    elif (580 <= val <= 629) or val == 788:
        return 'Genitourinary'
    elif 140 <= val <= 239:
        return 'Neoplasms'
    else:
        return 'Other'


def engineer_features(df):
    """
    Perform clinically grounded feature engineering, ICD-9 mapping,
    and regroup high-cardinality features.
    """
    df = df.copy()
    
    # 1. Map Age brackets to midpoint numeric values for ordinal modeling
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age_num'] = df['age'].map(age_map).fillna(65).astype(float)
    
    # 2. ICD-9 Diagnostic groupings for primary and secondary diagnoses
    df['diag_1_group'] = df['diag_1'].apply(parse_icd9_code)
    df['diag_2_group'] = df['diag_2'].apply(parse_icd9_code)
    df['diag_3_group'] = df['diag_3'].apply(parse_icd9_code)
    
    # 3. Clinical utilization history and interaction ratios
    df['total_visits_prior'] = (
        df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    )
    df['has_prior_emergency'] = (df['number_emergency'] > 0).astype(int)
    df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
    df['medication_density'] = df['num_medications'] / np.maximum(df['time_in_hospital'], 1)
    df['procedure_to_med_ratio'] = (
        (df['num_procedures'] + df['num_lab_procedures']) / np.maximum(df['num_medications'], 1)
    )
    
    # 4. Medication behavior and treatment indicators
    df['med_change_flag'] = (df['change'] == 'Ch').astype(int)
    df['diabetes_med_flag'] = (df['diabetesMed'] == 'Yes').astype(int)
    
    # 5. Group medical specialty: Top 10 + Missing + Other
    top_specialties = [
        'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
        'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
        'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology'
    ]
    df['medical_specialty_clean'] = df['medical_specialty'].apply(
        lambda x: 'Missing' if x == '?' or pd.isna(x)
        else (x if x in top_specialties else 'Other')
    )
    
    # 6. Admission / Discharge categorical IDs as string tokens
    df['admission_type_str'] = 'AdmType_' + df['admission_type_id'].astype(str)
    df['discharge_disp_str'] = 'DischDisp_' + df['discharge_disposition_id'].astype(str)
    df['admission_source_str'] = 'AdmSource_' + df['admission_source_id'].astype(str)
    
    # 7. Drop identifier, zero-variance, target, and uninformative features
    drop_cols = [
        'encounter_id', 'patient_nbr', 'weight', 'payer_code',
        'examide', 'citoglipton', 'readmitted', 'target',
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    features_df = df.drop(columns=existing_drops)
    
    return features_df


def create_preprocessor(X_train):
    """
    Build a Scikit-Learn ColumnTransformer pipeline.
    StandardScaler for continuous numeric features.
    OneHotEncoder for categorical features.
    """
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')
    
    return preprocessor, numeric_cols, categorical_cols
