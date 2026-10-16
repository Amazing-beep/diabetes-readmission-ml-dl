"""
deep_learning_models.py
-----------------------
TensorFlow/Keras architectures and tf.data input pipelines for
30-day hospital readmission binary prediction.
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from src.evaluation import compute_metrics


def create_tf_dataset(X, y=None, batch_size=256, is_training=False, shuffle_buffer=10000):
    """
    Build a high-performance tf.data input pipeline from NumPy arrays.
    Leverages shuffling (training only), batching, and AUTOTUNE prefetching.
    """
    # Cast input features to float32
    X_tensor = tf.cast(X, tf.float32)
    
    if y is not None:
        y_tensor = tf.cast(y, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(X_tensor)
        
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(X), shuffle_buffer), seed=42)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_baseline_mlp(input_dim):
    """
    Experiment DL-01: Baseline 2-layer Dense Neural Network (Sequential API).
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout_1'),
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='Baseline_MLP')
    return model


def build_deep_regularized_mlp(input_dim):
    """
    Experiment DL-02: 3-layer Deep MLP with Batch Normalization,
    Dropout, and L2 Weight Regularization (Functional API).
    """
    inputs = layers.Input(shape=(input_dim,), name='input_features')
    
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='act_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)
    
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4), name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='act_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    x = layers.Dense(32, activation='relu', name='dense_3')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Deep_Regularized_MLP')
    return model


def build_wide_mlp(input_dim):
    """
    Experiment DL-03: Wide 2-layer MLP for learning rate / batch tuning.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', name='dense_wide_1'),
        layers.Dropout(0.25, name='dropout_1'),
        layers.Dense(32, activation='relu', name='dense_wide_2'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='Wide_MLP')
    return model


def train_eval_dl_model(model, train_ds, val_ds, y_val, exp_id, model_name,
                        hparams, epochs=25, class_weight=None, notes=""):
    """
    Compile, train with EarlyStopping and ReduceLROnPlateau, evaluate, and record metrics.
    """
    lr = hparams.get('learning_rate', 1e-3)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    # We compile with binary cross-entropy, tracking AUC (ROC) and PR-AUC
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='roc_auc'),
            keras.metrics.AUC(curve='PR', name='pr_auc'),
            'accuracy'
        ]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_roc_auc',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_roc_auc',
            mode='max',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    train_duration = time.time() - start_time
    
    # Predict probabilities on validation dataset
    y_val_prob = model.predict(val_ds, verbose=0).ravel()
    metrics = compute_metrics(y_val, y_val_prob)
    
    record = {
        'exp_id': exp_id,
        'family': 'Deep Learning',
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
    
    return model, y_val_prob, history.history, record
