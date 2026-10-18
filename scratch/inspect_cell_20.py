import json

nb_path = 'notebooks/diabetes_readmission_classification.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("CELL 20 SOURCE:")
print("".join(nb['cells'][20]['source']))
