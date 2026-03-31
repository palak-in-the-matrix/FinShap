"""
==============================================================================
train_baseline.py — Phase 1: Baseline Sentiment Classification
==============================================================================
Financial Text Sentiment Analysis with Sarcasm Detection & Explainable AI

Pipeline:
    Text → Preprocessing → TF-IDF Vectorization → Logistic Regression
    → Sentiment Prediction → Evaluation Metrics

Performance Optimizations:
    - Chunked data loading (50K rows per chunk)
    - Sparse matrix representation (TF-IDF)
    - Parallel processing via n_jobs=-1
    - Memory-efficient column selection (only 'title')

Author: Research Implementation
==============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data_chunked, preprocess_batch,
    simulate_sentiment_labels_batch, format_metrics
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'r_wallstreetbets_big.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# TF-IDF parameters
TFIDF_MAX_FEATURES = 50000       # Maximum vocabulary size
TFIDF_NGRAM_RANGE = (1, 2)       # Unigrams and bigrams
TFIDF_SUBLINEAR_TF = True        # Apply sublinear TF scaling (1 + log(tf))
TFIDF_MIN_DF = 5                 # Minimum document frequency
TFIDF_MAX_DF = 0.95              # Maximum document frequency ratio

# Logistic Regression parameters
LR_MAX_ITER = 1000               # Maximum optimization iterations
LR_SOLVER = 'saga'               # Solver for large-scale problems
LR_C = 1.0                       # Regularization parameter
LR_N_JOBS = -1                   # Use all available CPU cores

# Data split
TEST_SIZE = 0.20                 # 80/20 train-test split
RANDOM_STATE = 42                # Reproducibility seed


def main():
    """
    Main training pipeline for baseline sentiment classification.
    
    Steps:
        1. Load dataset (chunked, memory-efficient)
        2. Preprocess text data
        3. Simulate sentiment labels
        4. Train/test split (stratified, 80/20)
        5. TF-IDF vectorization
        6. Logistic Regression training
        7. Evaluation and metrics computation
        8. Save all artifacts to models/ directory
    """
    
    overall_start = time.time()
    
    print("=" * 70)
    print("PHASE 1: BASELINE SENTIMENT CLASSIFICATION")
    print("=" * 70)
    
    # ------------------------------------------------------------------
    # Step 1: Load Data
    # ------------------------------------------------------------------
    print("\n[STEP 1/7] Loading dataset...")
    step_start = time.time()
    
    df = load_data_chunked(DATA_PATH, chunksize=50000)
    load_time = time.time() - step_start
    print(f"[DONE] Data loaded in {load_time:.2f}s | Shape: {df.shape}")
    
    # ------------------------------------------------------------------
    # Step 2: Preprocess Text
    # ------------------------------------------------------------------
    print("\n[STEP 2/7] Preprocessing text...")
    step_start = time.time()
    
    df['clean_title'] = preprocess_batch(df['title'])
    
    # Remove empty texts after preprocessing
    df = df[df['clean_title'].str.len() > 0].reset_index(drop=True)
    preprocess_time = time.time() - step_start
    print(f"[DONE] Preprocessing in {preprocess_time:.2f}s | Rows after cleaning: {len(df):,}")
    
    # ------------------------------------------------------------------
    # Step 3: Simulate Sentiment Labels
    # ------------------------------------------------------------------
    print("\n[STEP 3/7] Simulating sentiment labels...")
    step_start = time.time()
    
    df['sentiment'] = simulate_sentiment_labels_batch(df['clean_title'], seed=RANDOM_STATE)
    label_time = time.time() - step_start
    
    # Display label distribution
    label_dist = df['sentiment'].value_counts()
    print(f"[DONE] Labels generated in {label_time:.2f}s")
    print(f"  Label Distribution:")
    for label, count in label_dist.items():
        pct = count / len(df) * 100
        print(f"    {label:>10}: {count:>10,} ({pct:.1f}%)")
    
    # ------------------------------------------------------------------
    # Step 4: Encode Labels and Split Data
    # ------------------------------------------------------------------
    print("\n[STEP 4/7] Splitting data (80/20 stratified)...")
    step_start = time.time()
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_title'].values,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    split_time = time.time() - step_start
    print(f"[DONE] Split in {split_time:.2f}s")
    print(f"  Training set: {len(X_train_text):,} samples")
    print(f"  Test set:     {len(X_test_text):,} samples")
    
    # ------------------------------------------------------------------
    # Step 5: TF-IDF Vectorization
    # ------------------------------------------------------------------
    print("\n[STEP 5/7] TF-IDF Vectorization...")
    step_start = time.time()
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        dtype=np.float32            # Use float32 to save memory
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    
    tfidf_time = time.time() - step_start
    vocab_size = len(tfidf_vectorizer.vocabulary_)
    print(f"[DONE] TF-IDF in {tfidf_time:.2f}s")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Train matrix shape: {X_train_tfidf.shape} (sparse)")
    print(f"  Test matrix shape:  {X_test_tfidf.shape} (sparse)")
    print(f"  Memory (train): {X_train_tfidf.data.nbytes / 1024 / 1024:.1f} MB")
    
    # ------------------------------------------------------------------
    # Step 6: Train Logistic Regression
    # ------------------------------------------------------------------
    print("\n[STEP 6/7] Training Logistic Regression...")
    step_start = time.time()
    
    model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        solver=LR_SOLVER,
        C=LR_C,
        n_jobs=LR_N_JOBS,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - step_start
    print(f"\n[DONE] Model trained in {train_time:.2f}s")
    
    # ------------------------------------------------------------------
    # Step 7: Evaluate Model
    # ------------------------------------------------------------------
    print("\n[STEP 7/7] Evaluating model...")
    step_start = time.time()
    
    y_pred = model.predict(X_test_tfidf)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    
    eval_time = time.time() - step_start
    
    # Print results
    print(f"\n{'='*60}")
    print(f"BASELINE MODEL — EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  Labels: {list(label_encoder.classes_)}")
    print(f"  {conf_matrix}")
    print(f"\n  Classification Report:")
    print(f"  {class_report}")
    
    # ------------------------------------------------------------------
    # Save All Artifacts
    # ------------------------------------------------------------------
    print("\n[SAVING] Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, os.path.join(MODEL_DIR, 'baseline_model.joblib'))
    joblib.dump(tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    
    # Save test data for Phase 2
    joblib.dump({
        'X_test_text': X_test_text,
        'X_test_tfidf': X_test_tfidf,
        'y_test': y_test,
        'y_pred_baseline': y_pred,
        'X_train_text': X_train_text,
        'y_train': y_train,
    }, os.path.join(MODEL_DIR, 'test_data.joblib'))
    
    # Compute total time
    total_time = time.time() - overall_start
    
    # Save comprehensive metrics
    baseline_metrics = {
        'baseline_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'label_classes': list(label_encoder.classes_),
        'train_size': len(X_train_text),
        'test_size': len(X_test_text),
        'vocab_size': vocab_size,
        'tfidf_max_features': TFIDF_MAX_FEATURES,
        'model_params': {
            'solver': LR_SOLVER,
            'max_iter': LR_MAX_ITER,
            'C': LR_C
        },
        # Timing breakdown
        'timing': {
            'load_time': load_time,
            'preprocess_time': preprocess_time,
            'label_time': label_time,
            'split_time': split_time,
            'tfidf_time': tfidf_time,
            'train_time': train_time,
            'eval_time': eval_time,
            'total_time': total_time
        },
        'label_distribution': label_dist.to_dict(),
    }
    
    joblib.dump(baseline_metrics, os.path.join(MODEL_DIR, 'baseline_metrics.joblib'))
    
    print(f"\n[SAVED] All artifacts saved to: {MODEL_DIR}")
    print(f"  - baseline_model.joblib")
    print(f"  - tfidf_vectorizer.joblib")
    print(f"  - label_encoder.joblib")
    print(f"  - test_data.joblib")
    print(f"  - baseline_metrics.joblib")
    
    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING COMPLETE")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Baseline Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    return baseline_metrics


if __name__ == '__main__':
    metrics = main()
