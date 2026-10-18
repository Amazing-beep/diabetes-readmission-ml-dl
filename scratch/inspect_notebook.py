import json

with open('notebooks/diabetes_readmission_classification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    print("=" * 60)
    print(f"CELL {idx:02d} ({cell['cell_type']})")
    print("-" * 30)
    lines = src.strip().split("\n")
    print("\n".join(lines[:6]))
    if len(lines) > 6:
        print(f"... [{len(lines)-6} more lines]")
    
    if cell['cell_type'] == 'code':
        outputs = cell.get('outputs', [])
        print(f"Output count: {len(outputs)}")
        for o_idx, out in enumerate(outputs):
            out_type = out.get('output_type', 'unknown')
            if 'text' in out:
                txt = "".join(out['text'])
                first_few = "\n".join(txt.strip().split("\n")[:3])
                print(f"  [Out {o_idx} text ({out_type})]: {first_few[:120]}")
            elif 'data' in out:
                keys = list(out['data'].keys())
                print(f"  [Out {o_idx} data ({out_type})]: keys={keys}")
            elif 'traceback' in out:
                print(f"  [Out {o_idx} TRACEBACK ERROR!]")
