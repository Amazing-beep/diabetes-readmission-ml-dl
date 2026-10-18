import pandas as pd

df_raw = pd.read_csv("data/diabetic_data.csv", keep_default_na=False)
print("Raw <30 count:", (df_raw['readmitted'] == '<30').sum())

expired = df_raw['discharge_disposition_id'].astype(int).isin([11, 19, 20, 21])
print("Expired count:", expired.sum())
print("Expired with <30:", (expired & (df_raw['readmitted'] == '<30')).sum())

# What if hospice [13, 14] or others are excluded?
hospice = df_raw['discharge_disposition_id'].astype(int).isin([13, 14])
print("Hospice count:", hospice.sum())
print("Hospice with <30:", (hospice & (df_raw['readmitted'] == '<30')).sum())

# What if gender invalid?
gender_invalid = df_raw['gender'] == 'Unknown/Invalid'
print("Gender invalid with <30:", (gender_invalid & (df_raw['readmitted'] == '<30')).sum())

print("Hospice + gender invalid with <30:", ((hospice | gender_invalid) & (df_raw['readmitted'] == '<30')).sum())
print("11357 - 43 =", 11357 - 43)

# Let's check which 43 encounters have <30 among hospice or discharge codes
for code in [13, 14, 11, 19, 20, 21]:
    sub = df_raw[df_raw['discharge_disposition_id'].astype(int) == code]
    print(f"Discharge code {code}: Total={len(sub)}, <30={(sub['readmitted'] == '<30').sum()}")
