import json

nb_path = 'notebooks/diabetes_readmission_classification.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = "".join(cell.get('source', []))
    first_line = source.split('\n')[0] if source else ''
    first_line = first_line[:80]
    num_outputs = len(cell.get('outputs', []))
    print(f"Cell {idx:02d} [{ctype}]: {first_line} (outputs: {num_outputs})")
