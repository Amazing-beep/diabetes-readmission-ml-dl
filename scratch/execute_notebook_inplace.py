import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time
import sys

print("Building clean notebook first...")
import build_final_notebook
build_final_notebook.build_notebook()

nb_path = "notebooks/diabetes_readmission_classification.ipynb"
print(f"Reading notebook from {nb_path}...")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("Starting in-place notebook execution...")
ep = ExecutePreprocessor(timeout=900, kernel_name='python3')

t0 = time.time()
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    print(f"Execution successful in {time.time() - t0:.2f} seconds!")
except Exception as e:
    print(f"ERROR during notebook execution: {e}", file=sys.stderr)
    raise e

print(f"Saving executed notebook to {nb_path}...")
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook execution and saving completed successfully!")
