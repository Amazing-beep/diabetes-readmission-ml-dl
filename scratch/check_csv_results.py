import pandas as pd

print("=== experiment_results.csv ===")
df_exp = pd.read_csv('results/experiment_results.csv')
print(df_exp.to_string())

print("\n=== final_test_evaluation.csv ===")
df_test = pd.read_csv('results/final_test_evaluation.csv')
print(df_test.to_string())
