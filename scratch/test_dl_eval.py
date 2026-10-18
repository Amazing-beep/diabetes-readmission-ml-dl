import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

print("Checking if saved models exist in models/:")
print(os.path.exists("models/best_traditional_model.joblib"))
print(os.path.exists("models/best_deep_learning_model.keras"))
print(os.path.exists("models/model_metadata.json"))

if os.path.exists("models/model_metadata.json"):
    with open("models/model_metadata.json", "r") as f:
        print("Metadata:", json.load(f))
