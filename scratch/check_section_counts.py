import re
import importlib.util

spec = importlib.util.spec_from_file_location("m", "scratch/make_strictly_3500.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

sections = [
    ("header", m.header),
    ("abstract", m.abstract),
    ("sec1", m.sec1),
    ("sec2", m.sec2),
    ("sec3", m.sec3),
    ("sec4", m.sec4),
    ("sec5", m.sec5),
    ("sec6", m.sec6),
    ("sec7", m.sec7),
    ("appendix", m.appendix),
]

total_words = 0
total_tokens = 0
for name, content in sections:
    w = len(re.findall(r'\b\w+\b', content))
    t = len(content.split())
    total_words += w
    total_tokens += t
    print(f"{name:10s}: {w:5d} words | {t:5d} tokens")

print("-" * 35)
print(f"{'TOTAL':10s}: {total_words:5d} words | {total_tokens:5d} tokens")
