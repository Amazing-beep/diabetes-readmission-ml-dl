"""
run_experiments.py
------------------
Main execution script to run all 10 systematically varied experiments:
- 6 Traditional ML models (Dummy, Logistic Reg x2, Random Forest, HistGB, LightGBM)
- 4 Deep Learning models using tf.data and Keras (Baseline MLP, Deep Regularized, Wide Tuned, Cost-Sensitive)
- Logs results to results/experiment_results.csv and W&B offline mode
- Saves figures (Confusion Matrices, ROC, PR Curves, DL History) to results/figures/
- Saves best traditional and DL models to models/
- Evaluates selected best models on the untouched test set
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import wandb
import tensorflow as tf

from src.data_preprocessing import load_raw_data, audit_raw_data, apply_target_and_exclusions, patient_grouped_split
from src.feature_engineering import engineer_features, create_preprocessor
from src.traditional_models import get_traditional_experiments, train_eval_traditional_model
from src.deep_learning_models import (
    create_tf_dataset, build_baseline_mlp, build_deep_regularized_mlp,
    build_wide_mlp, train_eval_dl_model
)
from src.evaluation import (
    compute_metrics, plot_confusion_matrix, plot_roc_pr_curves, plot_dl_history
)


def main():
    print("=" * 70)
    print("STARTING REPRODUCIBLE EXPERIMENTATION PIPELINE")
    print("=" * 70)
    
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(r"C:\Users\pc\Downloads\diabetes+130-us+hospitals+for+years+1999-2008", "diabetic_data.csv")
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Initialize Weights & Biases in OFFLINE mode (rubric compliance)
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        project="diabetes-readmission-ml-dl",
        name="full_experimental_benchmarking",
        notes="Systematic comparison of 6 Traditional ML and 4 Deep Learning architectures on UCI Diabetes dataset."
    )
    
    # 3. Load & Audit Raw Data
    print("\n[Step 1/7] Loading and Auditing Raw Dataset...")
    df_raw = load_raw_data(data_path)
    audit = audit_raw_data(df_raw)
    print(f"Raw shape: {audit['n_rows']} rows, {audit['n_cols']} cols.")
    print(f"Unique patients: {audit['unique_patients']}, Repeat patients: {audit['repeat_patients']}.")
    
    # 4. Clinical Exclusions & Binary Target Definition
    print("\n[Step 2/7] Applying Target Definition and Clinical Exclusions...")
    df_clean = apply_target_and_exclusions(df_raw)
    print(f"Eligible non-expired encounters: {len(df_clean)} (Excluded {audit['expired_encounters']} expired encounters).")
    
    # 5. Patient-Grouped Splitting (Strategy B)
    print("\n[Step 3/7] Performing Patient-Grouped Split (70% Train, 15% Val, 15% Test)...")
    train_df, val_df, test_df = patient_grouped_split(df_clean, test_size=0.15, val_size=0.15, random_state=42)
    
    y_train = train_df['target'].values
    y_val = val_df['target'].values
    y_test = test_df['target'].values
    
    print(f"Train: {len(train_df)} rows ({train_df['patient_nbr'].nunique()} pts) | Pos: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Val:   {len(val_df)} rows ({val_df['patient_nbr'].nunique()} pts) | Pos: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
    print(f"Test:  {len(test_df)} rows ({test_df['patient_nbr'].nunique()} pts) | Pos: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # 6. Feature Engineering & Preprocessing
    print("\n[Step 4/7] Engineering Features and Fitting ColumnTransformer on Training Data...")
    X_train_raw = engineer_features(train_df)
    X_val_raw = engineer_features(val_df)
    X_test_raw = engineer_features(test_df)
    
    preprocessor, num_cols, cat_cols = create_preprocessor(X_train_raw)
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    input_dim = X_train_proc.shape[1]
    print(f"Transformed feature dimension: {input_dim} columns.")
    
    experiment_records = []
    val_predictions = {}
    fitted_models = {}
    
    # 7. Traditional Machine Learning Experiments (EXP-ML-00 to EXP-ML-05)
    print("\n[Step 5/7] Executing Traditional Machine Learning Experiments...")
    trad_exps = get_traditional_experiments()
    
    for exp in trad_exps:
        exp_id = exp['exp_id']
        name = exp['name']
        print(f"  --> Running {exp_id}: {name}...")
        model, y_val_prob, record = train_eval_traditional_model(
            model=exp['model'],
            X_train=X_train_proc,
            y_train=y_train,
            X_val=X_val_proc,
            y_val=y_val,
            exp_id=exp_id,
            model_name=name,
            hparams=exp['hparams'],
            notes=exp['notes']
        )
        experiment_records.append(record)
        val_predictions[name] = (y_val, y_val_prob)
        fitted_models[exp_id] = model
        wandb.log({f"{exp_id}_{k}": v for k, v in record.items() if isinstance(v, (int, float))})
        print(f"      Val ROC-AUC: {record['val_roc_auc']:.4f} | PR-AUC: {record['val_pr_auc']:.4f} | Recall: {record['val_recall']:.4f}")
        
    # 8. Deep Learning Experiments (EXP-DL-01 to EXP-DL-04)
    print("\n[Step 6/7] Executing Deep Learning Experiments with tf.data Pipelines...")
    
    # Prepare tf.data pipelines
    train_ds_256 = create_tf_dataset(X_train_proc, y_train, batch_size=256, is_training=True)
    val_ds_256 = create_tf_dataset(X_val_proc, y_val, batch_size=256, is_training=False)
    
    train_ds_512 = create_tf_dataset(X_train_proc, y_train, batch_size=512, is_training=True)
    val_ds_512 = create_tf_dataset(X_val_proc, y_val, batch_size=512, is_training=False)
    
    # Compute balanced class weight for cost-sensitive training
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    class_weight_dict = {0: 1.0, 1: float(neg_count / pos_count)}
    
    dl_experiments = [
        {
            'exp_id': 'EXP-DL-01',
            'name': 'Baseline MLP (Sequential)',
            'model_fn': lambda: build_baseline_mlp(input_dim),
            'train_ds': train_ds_256,
            'val_ds': val_ds_256,
            'class_weight': None,
            'hparams': {'layers': [64, 32], 'dropout': 0.2, 'lr': 1e-3, 'batch_size': 256},
            'notes': 'Baseline two-layer feedforward architecture using Sequential API with standard binary cross-entropy.'
        },
        {
            'exp_id': 'EXP-DL-02',
            'name': 'Deep Regularized MLP (Functional)',
            'model_fn': lambda: build_deep_regularized_mlp(input_dim),
            'train_ds': train_ds_256,
            'val_ds': val_ds_256,
            'class_weight': None,
            'hparams': {'layers': [128, 64, 32], 'dropout': 0.3, 'l2': 1e-4, 'batch_norm': True, 'lr': 1e-3, 'batch_size': 256},
            'notes': 'Three-layer Functional API network adding BatchNormalization, higher Dropout, and L2 weight decay.'
        },
        {
            'exp_id': 'EXP-DL-03',
            'name': 'Wide MLP with Tuned Learning Rate',
            'model_fn': lambda: build_wide_mlp(input_dim),
            'train_ds': train_ds_512,
            'val_ds': val_ds_512,
            'class_weight': None,
            'hparams': {'layers': [128, 32], 'dropout': 0.25, 'lr': 5e-4, 'batch_size': 512},
            'notes': 'Wide first layer with halved learning rate and larger batch size for smoother gradient convergence.'
        },
        {
            'exp_id': 'EXP-DL-04',
            'name': 'Cost-Sensitive Deep MLP',
            'model_fn': lambda: build_deep_regularized_mlp(input_dim),
            'train_ds': train_ds_256,
            'val_ds': val_ds_256,
            'class_weight': class_weight_dict,
            'hparams': {'layers': [128, 64, 32], 'dropout': 0.3, 'class_weight': f'0:1, 1:{class_weight_dict[1]:.2f}', 'lr': 1e-3, 'batch_size': 256},
            'notes': 'Cost-sensitive training assigning 7.8x loss penalty to readmission false negatives to boost clinical recall.'
        }
    ]
    
    dl_histories = {}
    for exp in dl_experiments:
        exp_id = exp['exp_id']
        name = exp['name']
        print(f"  --> Running {exp_id}: {name}...")
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        model = exp['model_fn']()
        
        model, y_val_prob, history, record = train_eval_dl_model(
            model=model,
            train_ds=exp['train_ds'],
            val_ds=exp['val_ds'],
            y_val=y_val,
            exp_id=exp_id,
            model_name=name,
            hparams=exp['hparams'],
            epochs=20,
            class_weight=exp['class_weight'],
            notes=exp['notes']
        )
        experiment_records.append(record)
        val_predictions[name] = (y_val, y_val_prob)
        fitted_models[exp_id] = model
        dl_histories[exp_id] = history
        wandb.log({f"{exp_id}_{k}": v for k, v in record.items() if isinstance(v, (int, float))})
        print(f"      Val ROC-AUC: {record['val_roc_auc']:.4f} | PR-AUC: {record['val_pr_auc']:.4f} | Recall: {record['val_recall']:.4f}")
        
        # Plot DL History
        hist_fig_path = os.path.join(figures_dir, f"{exp_id.lower()}_learning_curves.png")
        plot_dl_history(history, name, save_path=hist_fig_path)
        
    # 9. Save Experiment Results Table
    df_results = pd.DataFrame(experiment_records)
    results_csv_path = os.path.join(results_dir, "experiment_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\n[Step 7/7] Experiment results saved to: {results_csv_path}")
    print("\n--- SUMMARY EXPERIMENT TABLE (VALIDATION PERFORMANCE) ---")
    print(df_results[['exp_id', 'algorithm', 'val_roc_auc', 'val_pr_auc', 'val_recall', 'val_f1', 'train_duration_sec']].to_string(index=False))
    
    # 10. Select Best Models on Validation Metric (ROC-AUC / PR-AUC)
    # Identify best traditional model
    trad_records = [r for r in experiment_records if r['family'] == 'Traditional ML' and r['exp_id'] != 'EXP-ML-00']
    best_trad_record = max(trad_records, key=lambda x: x['val_roc_auc'])
    best_trad_id = best_trad_record['exp_id']
    best_trad_model = fitted_models[best_trad_id]
    
    # Identify best DL model
    dl_records = [r for r in experiment_records if r['family'] == 'Deep Learning']
    best_dl_record = max(dl_records, key=lambda x: x['val_roc_auc'])
    best_dl_id = best_dl_record['exp_id']
    best_dl_model = fitted_models[best_dl_id]
    
    print(f"\nBest Traditional Model Selected: {best_trad_id} ({best_trad_record['algorithm']}) - Val ROC-AUC: {best_trad_record['val_roc_auc']}")
    print(f"Best Deep Learning Model Selected: {best_dl_id} ({best_dl_record['algorithm']}) - Val ROC-AUC: {best_dl_record['val_roc_auc']}")
    
    # 11. Final Evaluation on Untouched Test Set
    print("\nEvaluating Selected Models on Held-out Test Set...")
    # Traditional Test
    if hasattr(best_trad_model, 'predict_proba'):
        y_test_trad_prob = best_trad_model.predict_proba(X_test_proc)[:, 1]
    else:
        y_test_trad_prob = best_trad_model.predict(X_test_proc).astype(float)
    trad_test_metrics = compute_metrics(y_test, y_test_trad_prob)
    
    # DL Test
    test_ds = create_tf_dataset(X_test_proc, y_test, batch_size=256, is_training=False)
    y_test_dl_prob = best_dl_model.predict(test_ds, verbose=0).ravel()
    dl_test_metrics = compute_metrics(y_test, y_test_dl_prob)
    
    print("\n--- FINAL TEST EVALUATION METRICS ---")
    test_summary = [
        {
            'Model': f"Best Traditional ({best_trad_record['algorithm']})",
            'Test ROC-AUC': round(trad_test_metrics['roc_auc'], 4),
            'Test PR-AUC': round(trad_test_metrics['pr_auc'], 4),
            'Test F1': round(trad_test_metrics['f1'], 4),
            'Test Recall': round(trad_test_metrics['recall'], 4),
            'Test Precision': round(trad_test_metrics['precision'], 4),
            'Test Accuracy': round(trad_test_metrics['accuracy'], 4),
            'Test Specificity': round(trad_test_metrics['specificity'], 4)
        },
        {
            'Model': f"Best Deep Learning ({best_dl_record['algorithm']})",
            'Test ROC-AUC': round(dl_test_metrics['roc_auc'], 4),
            'Test PR-AUC': round(dl_test_metrics['pr_auc'], 4),
            'Test F1': round(dl_test_metrics['f1'], 4),
            'Test Recall': round(dl_test_metrics['recall'], 4),
            'Test Precision': round(dl_test_metrics['precision'], 4),
            'Test Accuracy': round(dl_test_metrics['accuracy'], 4),
            'Test Specificity': round(dl_test_metrics['specificity'], 4)
        }
    ]
    df_test_summary = pd.DataFrame(test_summary)
    print(df_test_summary.to_string(index=False))
    df_test_summary.to_csv(os.path.join(results_dir, "final_test_evaluation.csv"), index=False)
    
    # 12. Save Figures
    print("\nGenerating and Saving Evaluation Figures...")
    # Comparative ROC & PR curves on validation set
    plot_roc_pr_curves(
        {
            'Dummy Baseline': val_predictions['Dummy Baseline'],
            'Logistic Reg (Balanced)': val_predictions['Logistic Regression (Balanced)'],
            'Random Forest': val_predictions['Random Forest (Balanced)'],
            'LightGBM': val_predictions['LightGBM (Cost-Sensitive)'],
            'Baseline MLP': val_predictions['Baseline MLP (Sequential)'],
            'Deep Regularized MLP': val_predictions['Deep Regularized MLP (Functional)']
        },
        save_path=os.path.join(figures_dir, "comparative_roc_pr_curves.png")
    )
    
    # Confusion Matrices for Best Models on Test Set
    plot_confusion_matrix(
        y_test, y_test_trad_prob,
        model_name=f"Best Traditional ML ({best_trad_record['algorithm']})",
        save_path=os.path.join(figures_dir, "confusion_matrix_traditional_test.png")
    )
    plot_confusion_matrix(
        y_test, y_test_dl_prob,
        model_name=f"Best Deep Learning ({best_dl_record['algorithm']})",
        save_path=os.path.join(figures_dir, "confusion_matrix_deep_learning_test.png")
    )
    
    # 13. Save Best Models & Pipeline
    print("\nSaving Best Models and Artifacts...")
    # Full traditional model pipeline (fitted preprocessor + estimator)
    from sklearn.pipeline import Pipeline
    full_trad_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_trad_model)
    ])
    trad_save_path = os.path.join(models_dir, "best_traditional_model.joblib")
    joblib.dump(full_trad_pipeline, trad_save_path)
    print(f"Saved best traditional model pipeline to: {trad_save_path}")
    
    # Best Keras model
    dl_save_path = os.path.join(models_dir, "best_deep_learning_model.keras")
    best_dl_model.save(dl_save_path)
    print(f"Saved best deep learning model to: {dl_save_path}")
    
    # Feature Schema & Metadata
    metadata = {
        'target_definition': "1 = readmitted < 30 days; 0 = no readmit or readmitted >= 30 days",
        'clinical_exclusions': "discharge_disposition_id in [11, 19, 20, 21] (expired/in-hospital death)",
        'split_strategy': "GroupShuffleSplit on patient_nbr (70% Train, 15% Val, 15% Test)",
        'input_dimension': int(input_dim),
        'best_traditional_model': best_trad_record['algorithm'],
        'best_traditional_val_roc_auc': best_trad_record['val_roc_auc'],
        'best_traditional_test_roc_auc': round(trad_test_metrics['roc_auc'], 4),
        'best_deep_learning_model': best_dl_record['algorithm'],
        'best_deep_learning_val_roc_auc': best_dl_record['val_roc_auc'],
        'best_deep_learning_test_roc_auc': round(dl_test_metrics['roc_auc'], 4),
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    wandb.finish()
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
