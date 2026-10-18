import json

nb_path = 'notebooks/diabetes_readmission_classification.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = "".join(cell.get('source', []))
    print(f"=== CELL {idx:02d} [{ctype}] ===")
    print(source[:500])
    if len(source) > 500:
        print("... [TRUNCATED] ...")
    print("\n")
