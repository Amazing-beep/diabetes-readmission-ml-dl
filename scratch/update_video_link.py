import json

nb_path = 'notebooks/diabetes_readmission_classification.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 0 to include presentation video
source = nb['cells'][0]['source']
new_source = []
for line in source:
    new_source.append(line)
    if "**Deliverables Folder**" in line or "**Deliverables**" in line:
        new_source.append("**Presentation Video**: [YouTube Presentation Video](https://youtu.be/vJBLKcI8V2Q?si=3iOISiRe8vF2Nc4M)  \n")

nb['cells'][0]['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated video link in cell 0 of notebook.")
