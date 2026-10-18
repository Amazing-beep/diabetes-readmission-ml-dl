import json

nb_path = 'notebooks/diabetes_readmission_classification.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [13, 15, 17, 20, 22, 24, 26, 27, 30]:
    cell = nb['cells'][idx]
    print(f"=== CELL {idx:02d} OUTPUTS ===")
    for out in cell.get('outputs', []):
        if 'text' in out:
            print("".join(out['text']))
        elif 'data' in out and 'text/plain' in out['data']:
            print("".join(out['data']['text/plain']))
