import json

nb_path = 'notebooks/diabetes_readmission_classification.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source_first = "".join(cell.get('source', [])).split('\n')[0][:70]
    outputs = cell.get('outputs', [])
    output_types = [o.get('output_type', '') for o in outputs]
    has_image = any('image/png' in o.get('data', {}) for o in outputs)
    print(f"Cell {idx:02d} [{ctype:<8}]: {source_first:<70} | Outputs: {len(outputs):2d} {output_types} (PNG: {has_image})")

print("\nDetailed Inspection of Key Code Cells:")
for idx in [3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 22, 24, 26, 27, 29, 33]:
    cell = nb['cells'][idx]
    print(f"\n==================== CELL {idx:02d} OUTPUT ====================")
    for o in cell.get('outputs', []):
        if 'text' in o:
            print("".join(o['text']))
        elif 'data' in o:
            if 'text/plain' in o['data']:
                print("".join(o['data']['text/plain']))
            if 'image/png' in o['data']:
                print(f"[IMAGE PNG EMBEDDED: {len(o['data']['image/png'])} chars base64]")
