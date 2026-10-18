import re

with open('scratch/execute_apply_tight_3500.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Extract sections
sec_names = ['header', 'abstract', 'sec1', 'sec2', 'sec3', 'sec4', 'sec5', 'sec6', 'sec7', 'appendix']
for name in sec_names:
    pattern = rf'{name} = """(.*?)"""'
    m = re.search(pattern, code, re.DOTALL)
    if m:
        val = m.group(1)
        w = len(re.findall(r'\b\w+\b', val))
        print(f"=== {name} ({w} words) ===")
        # split by '### '
        parts = val.split('### ')
        for p in parts:
            if p.strip():
                t = p.split('\n')[0].strip()
                pw = len(re.findall(r'\b\w+\b', p))
                print(f"  - {t[:45]:45s}: {pw:4d} words")
