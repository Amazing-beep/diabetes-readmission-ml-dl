import re

with open('report/report_outline.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Academic prose word count (excluding tables, code blocks, reference list)
print("Total words in report_outline.md:", len(re.findall(r'\b[A-Za-z0-9_-]+\b', text)))
