import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
import tensorflow as tf

SEED = 42
data_path = "data/diabetic_data.csv"
if not os.path.exists(data_path):
    data_path = "../data/diabetic_data.csv"

df_raw = pd.read_csv(data_path, keep_default_na=False)
expired_ids = [11, 19, 20, 21]
df_clean = df_raw[~df_raw['discharge_disposition_id'].astype(int).isin(expired_ids)].copy().reset_index(drop=True)
df_clean['target'] = (df_clean['readmitted'] == '<30').astype(int)

from sklearn.model_selection import GroupShuffleSplit
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_val_idx, test_idx = next(gss1.split(df_clean, df_clean['target'], groups=df_clean['patient_nbr']))

train_val_df = df_clean.iloc[train_val_idx].copy().reset_index(drop=True)
test_df = df_clean.iloc[test_idx].copy().reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=SEED)
train_idx, val_idx = next(gss2.split(train_val_df, train_val_df['target'], groups=train_val_df['patient_nbr']))

train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

def parse_icd9_code(code):
    if pd.isna(code) or code == '?' or code == '' or code is None:
        return 'Missing'
    code_str = str(code).strip()
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
    df = df.copy()
    age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
               '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
               '[80-90)': 85, '[90-100)': 95}
    df['age_num'] = df['age'].map(age_map).fillna(65).astype(float)
    df['diag_1_group'] = df['diag_1'].apply(parse_icd9_code)
    df['diag_2_group'] = df['diag_2'].apply(parse_icd9_code)
    df['diag_3_group'] = df['diag_3'].apply(parse_icd9_code)
    
    df['total_visits_prior'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['has_prior_emergency'] = (df['number_emergency'] > 0).astype(int)
    df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
    df['medication_density'] = df['num_medications'] / np.maximum(df['time_in_hospital'], 1)
    df['procedure_to_med_ratio'] = (df['num_procedures'] + df['num_lab_procedures']) / np.maximum(df['num_medications'], 1)
    df['med_change_flag'] = (df['change'] == 'Ch').astype(int)
    df['diabetes_med_flag'] = (df['diabetesMed'] == 'Yes').astype(int)
    
    top_specialties = ['InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
                       'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
                       'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology']
    df['medical_specialty_clean'] = df['medical_specialty'].apply(
        lambda x: 'Missing' if x == '?' or pd.isna(x) else (x if x in top_specialties else 'Other')
    )
    df['admission_type_str'] = 'AdmType_' + df['admission_type_id'].astype(str)
    df['discharge_disp_str'] = 'DischDisp_' + df['discharge_disposition_id'].astype(str)
    df['admission_source_str'] = 'AdmSource_' + df['admission_source_id'].astype(str)
    
    drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'examide', 'citoglipton',
                 'readmitted', 'target', 'diag_1', 'diag_2', 'diag_3', 'medical_specialty',
                 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    return df.drop(columns=[c for c in drop_cols if c in df.columns])

X_train_raw = engineer_features(train_df)
X_test_raw = engineer_features(test_df)

numeric_cols = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols)
], remainder='drop')

X_train_proc = preprocessor.fit_transform(X_train_raw)
X_test_proc = preprocessor.transform(X_test_raw)
y_test = test_df['target'].values

print(f"Total Test Encounters: {len(y_test)}")
print(f"Test Positive: {y_test.sum()}")
print(f"Test Negative: {(y_test == 0).sum()}")

# Train LightGBM directly
import lightgbm as lgb
clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, scale_pos_weight=7.8, random_state=SEED, verbose=-1)
clf.fit(X_train_proc, train_df['target'].values)

y_test_trad_prob = clf.predict_proba(X_test_proc)[:, 1]
y_test_trad_pred = (y_test_trad_prob >= 0.5).astype(int)
cm_trad = confusion_matrix(y_test, y_test_trad_pred)
tn, fp, fn, tp = cm_trad.ravel()
print("\n--- LightGBM Test Confusion Matrix ---")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Recall: {tp/(tp+fn):.4f} (TP count: {tp})")
print(f"Specificity: {tn/(tn+fp):.4f}")
print(f"Precision: {tp/(tp+fp):.4f}")
print(f"Accuracy: {(tp+tn)/len(y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_trad_prob):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_test_trad_prob):.4f}")

if os.path.exists("models/best_deep_learning_model.keras"):
    dl_model = tf.keras.models.load_model("models/best_deep_learning_model.keras")
    y_test_dl_prob = dl_model.predict(X_test_proc, verbose=0).ravel()
    y_test_dl_pred = (y_test_dl_prob >= 0.5).astype(int)
    cm_dl = confusion_matrix(y_test, y_test_dl_pred)
    tn_d, fp_d, fn_d, tp_d = cm_dl.ravel()
    print("\n--- Deep Learning Test Confusion Matrix ---")
    print(f"TN: {tn_d}, FP: {fp_d}, FN: {fn_d}, TP: {tp_d}")
    print(f"Recall: {tp_d/(tp_d+fn_d):.4f} (TP count: {tp_d})")
    print(f"Specificity: {tn_d/(tn_d+fp_d):.4f}")
    print(f"Precision: {tp_d/(tp_d+fp_d):.4f}")
    print(f"Accuracy: {(tp_d+tn_d)/len(y_test):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_test_dl_prob):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, y_test_dl_prob):.4f}")
