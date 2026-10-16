"""
traditional_models.py
---------------------
Traditional machine learning algorithms using Scikit-Learn and LightGBM:
- DummyClassifier Baseline
- Logistic Regression (standard and cost-sensitive)
- Random Forest (tuned depth and estimators)
- HistGradientBoostingClassifier
- LightGBM Gradient Boosting
"""

import time
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
from src.evaluation import compute_metrics


def train_eval_traditional_model(model, X_train, y_train, X_val, y_val, exp_id, model_name, hparams, notes=""):
    """
    Train a traditional ML model, record execution time, compute validation metrics,
    and return an experiment record dictionary.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - start_time
    
    # Predict probabilities for positive class (readmitted within 30 days)
    if hasattr(model, 'predict_proba'):
        y_val_prob = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, 'decision_function'):
        # calibrate or scale decision function via sigmoid
        import scipy.special
        y_val_prob = scipy.special.expit(model.decision_function(X_val))
    else:
        # e.g., dummy constant
        y_val_prob = model.predict(X_val).astype(float)
        
    metrics = compute_metrics(y_val, y_val_prob)
    
    record = {
        'exp_id': exp_id,
        'family': 'Traditional ML',
        'algorithm': model_name,
        'hyperparameters': str(hparams),
        'train_duration_sec': round(train_duration, 2),
        'val_roc_auc': round(metrics['roc_auc'], 4),
        'val_pr_auc': round(metrics['pr_auc'], 4),
        'val_f1': round(metrics['f1'], 4),
        'val_precision': round(metrics['precision'], 4),
        'val_recall': round(metrics['recall'], 4),
        'val_specificity': round(metrics['specificity'], 4),
        'val_accuracy': round(metrics['accuracy'], 4),
        'val_brier_score': round(metrics['brier_score'], 4),
        'notes': notes
    }
    return model, y_val_prob, record


def get_traditional_experiments():
    """
    Define systematic traditional ML experimental setups.
    """
    experiments = [
        {
            'exp_id': 'EXP-ML-00',
            'name': 'Dummy Baseline',
            'model': DummyClassifier(strategy='prior'),
            'hparams': {'strategy': 'prior'},
            'notes': 'Baseline predictor that predicts class prevalence without learning feature relationships.'
        },
        {
            'exp_id': 'EXP-ML-01',
            'name': 'Logistic Regression (Unweighted)',
            'model': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
            'hparams': {'C': 1.0, 'class_weight': None, 'solver': 'lbfgs'},
            'notes': 'Standard linear baseline with L2 penalty, unadjusted for 1:8 class imbalance.'
        },
        {
            'exp_id': 'EXP-ML-02',
            'name': 'Logistic Regression (Balanced)',
            'model': LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', solver='lbfgs', random_state=42),
            'hparams': {'C': 0.1, 'class_weight': 'balanced', 'solver': 'lbfgs'},
            'notes': 'Cost-sensitive linear model inversely weighting classes to heavily penalize false negatives.'
        },
        {
            'exp_id': 'EXP-ML-03',
            'name': 'Random Forest (Balanced)',
            'model': RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=10,
                                           class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            'hparams': {'n_estimators': 120, 'max_depth': 12, 'min_samples_leaf': 10, 'class_weight': 'balanced_subsample'},
            'notes': 'Non-linear ensemble with depth restriction to mitigate overfitting on high-dimensional one-hot data.'
        },
        {
            'exp_id': 'EXP-ML-04',
            'name': 'HistGradientBoosting (Default)',
            'model': HistGradientBoostingClassifier(learning_rate=0.08, max_iter=100, min_samples_leaf=25, random_state=42),
            'hparams': {'learning_rate': 0.08, 'max_iter': 100, 'min_samples_leaf': 25},
            'notes': 'Fast histogram-based gradient boosting baseline utilizing binned numerical features.'
        },
        {
            'exp_id': 'EXP-ML-05',
            'name': 'LightGBM (Cost-Sensitive)',
            'model': lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                                       scale_pos_weight=7.8, random_state=42, verbose=-1),
            'hparams': {'n_estimators': 150, 'learning_rate': 0.05, 'num_leaves': 31, 'scale_pos_weight': 7.8},
            'notes': 'Optimized gradient boosted trees with positive class scale weighting (7.8) for imbalanced readmission.'
        }
    ]
    return experiments
