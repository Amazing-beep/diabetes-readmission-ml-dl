import json

nb_path = 'notebooks/diabetes_readmission_classification.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        print(f"==================== MD CELL {idx:02d} ====================")
        print("".join(cell.get('source', [])))
