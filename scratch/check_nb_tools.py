import nbformat
print(f"nbformat version: {nbformat.__version__}")
try:
    import nbconvert
    print(f"nbconvert version: {nbconvert.__version__}")
except ImportError:
    print("nbconvert not installed")
try:
    import nbclient
    print(f"nbclient version: {nbclient.__version__}")
except ImportError:
    print("nbclient not installed")
