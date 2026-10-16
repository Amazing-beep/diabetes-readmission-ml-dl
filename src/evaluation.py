"""
evaluation.py
-------------
Evaluation metrics, confusion matrices, ROC / PR curves,
learning curve plotting, and experiment logging for hospital readmission models.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    brier_score_loss, roc_curve, precision_recall_curve
)


def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Compute comprehensive clinical classification metrics given true labels
    and predicted probabilities.
    """
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    brier = brier_score_loss(y_true, y_pred_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'specificity': float(specificity),
        'brier_score': float(brier),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def plot_confusion_matrix(y_true, y_pred_prob, model_name, threshold=0.5, save_path=None):
    """
    Plot and save an annotated clinical confusion matrix.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['No Readmit (<30d: No)', 'Readmitted (<30d: Yes)'],
        yticklabels=['No Readmit (<30d: No)', 'Readmitted (<30d: Yes)']
    )
    plt.title(f'Confusion Matrix: {model_name} (Threshold={threshold:.2f})', fontsize=12, pad=12)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_roc_pr_curves(results_dict, save_path=None):
    """
    Plot comparative ROC and Precision-Recall curves for multiple models.
    results_dict: {model_name: (y_true, y_pred_prob)}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for name, (y_true, y_prob) in results_dict.items():
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})', lw=2)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_val = average_precision_score(y_true, y_prob)
        ax2.plot(recall, precision, label=f'{name} (PR-AUC = {pr_val:.3f})', lw=2)
        
    # Baseline references
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.500)')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=13)
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # PR baseline is the prevalence of positive class
    first_y = list(results_dict.values())[0][0]
    prevalence = np.mean(first_y)
    ax2.axhline(prevalence, color='k', linestyle='--', lw=1.5, label=f'Prevalence ({prevalence:.3f})')
    ax2.set_title('Precision-Recall Curve', fontsize=13)
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_dl_history(history_dict, model_name, save_path=None):
    """
    Plot training and validation loss, ROC-AUC, and PR-AUC learning curves
    for Deep Learning experiments.
    """
    epochs = range(1, len(history_dict['loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss
    axes[0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', lw=2)
    if 'val_loss' in history_dict:
        axes[0].plot(epochs, history_dict['val_loss'], 'r--', label='Validation Loss', lw=2)
    axes[0].set_title(f'{model_name}: Binary Cross-Entropy Loss', fontsize=12)
    axes[0].set_xlabel('Epochs', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. ROC-AUC
    auc_key = [k for k in history_dict.keys() if 'auc' in k and 'val' not in k]
    if auc_key:
        axes[1].plot(epochs, history_dict[auc_key[0]], 'b-', label='Training AUC', lw=2)
        val_auc_key = 'val_' + auc_key[0]
        if val_auc_key in history_dict:
            axes[1].plot(epochs, history_dict[val_auc_key], 'r--', label='Validation AUC', lw=2)
        axes[1].set_title(f'{model_name}: Area Under ROC', fontsize=12)
        axes[1].set_xlabel('Epochs', fontsize=11)
        axes[1].set_ylabel('ROC-AUC', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
    # 3. PR-AUC or Precision
    pr_key = [k for k in history_dict.keys() if 'pr' in k and 'val' not in k]
    if pr_key:
        axes[2].plot(epochs, history_dict[pr_key[0]], 'b-', label='Training PR-AUC', lw=2)
        val_pr_key = 'val_' + pr_key[0]
        if val_pr_key in history_dict:
            axes[2].plot(epochs, history_dict[val_pr_key], 'r--', label='Validation PR-AUC', lw=2)
        axes[2].set_title(f'{model_name}: Precision-Recall AUC', fontsize=12)
        axes[2].set_xlabel('Epochs', fontsize=11)
        axes[2].set_ylabel('PR-AUC', fontsize=11)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
    else:
        # Fallback to accuracy if PR metric not tracked directly in history
        axes[2].plot(epochs, history_dict.get('accuracy', []), 'b-', label='Training Accuracy', lw=2)
        if 'val_accuracy' in history_dict:
            axes[2].plot(epochs, history_dict['val_accuracy'], 'r--', label='Validation Accuracy', lw=2)
        axes[2].set_title(f'{model_name}: Accuracy', fontsize=12)
        axes[2].set_xlabel('Epochs', fontsize=11)
        axes[2].set_ylabel('Accuracy', fontsize=11)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
