"""
==============================================================================
train_sarcasm.py — Phase 2: Sarcasm Detection & Sentiment Adjustment
==============================================================================
Financial Text Sentiment Analysis with Sarcasm Detection & Explainable AI

Sarcasm-Aware Pipeline Design:
    The key insight: sarcastic financial texts have surface-level sentiment
    that DIFFERS from the author's true intent. The baseline model only
    sees surface patterns. Phase 2 corrects this by:

    1. Detecting sarcasm in each text
    2. For sarcastic texts: using the FLIPPED sentiment as ground truth
    3. Training a new combined model that uses both text features AND
       sarcasm awareness to predict the TRUE sentiment

    This results in improved accuracy because the sarcasm-aware model
    correctly interprets texts where surface and true sentiment differ.

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from scipy.sparse import hstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data_chunked, preprocess_batch,
    simulate_sarcasm_labels_batch, simulate_sentiment_labels_batch,
    flip_sentiment, format_metrics
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'r_wallstreetbets_big.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
RANDOM_STATE = 42

# Sarcasm TF-IDF parameters
SARCASM_TFIDF_MAX_FEATURES = 30000
SARCASM_TFIDF_NGRAM_RANGE = (1, 2)

# Sarcasm Logistic Regression parameters
SARCASM_LR_MAX_ITER = 500
SARCASM_LR_SOLVER = 'saga'
SARCASM_LR_N_JOBS = -1

# Sarcasm-Aware Sentiment Model parameters
SA_TFIDF_MAX_FEATURES = 50000
SA_LR_MAX_ITER = 1000
SA_LR_SOLVER = 'saga'


def main():
    """
    Phase 2: Sarcasm-Aware Sentiment Classification Pipeline.
    
    Methodology:
        1. Load the same preprocessed data used in Phase 1
        2. Simulate sarcasm labels (which texts are sarcastic)
        3. Create "true sentiment" labels:
           - Non-sarcastic texts: original sentiment is the true sentiment
           - Sarcastic texts: FLIPPED sentiment is the true sentiment
           (This models the reality that sarcastic speakers mean the opposite)
        4. Train a sarcasm classifier
        5. Build a sarcasm-aware sentiment model that uses sarcasm signal
           as an additional feature alongside TF-IDF
        6. Compare against baseline accuracy
    """
    
    overall_start = time.time()
    
    print("=" * 70)
    print("PHASE 2: SARCASM DETECTION & SENTIMENT ADJUSTMENT")
    print("=" * 70)
    
    # ------------------------------------------------------------------
    # Step 1: Load Data and Baseline Artifacts
    # ------------------------------------------------------------------
    print("\n[STEP 1/7] Loading data and baseline artifacts...")
    step_start = time.time()
    
    # Load baseline artifacts
    baseline_metrics = joblib.load(os.path.join(MODEL_DIR, 'baseline_metrics.joblib'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    test_data = joblib.load(os.path.join(MODEL_DIR, 'test_data.joblib'))
    
    X_train_text = test_data['X_train_text']
    X_test_text = test_data['X_test_text']
    y_pred_baseline = test_data['y_pred_baseline']
    
    load_time = time.time() - step_start
    print(f"[DONE] Loaded in {load_time:.2f}s")
    print(f"  Training samples: {len(X_train_text):,}")
    print(f"  Test samples:     {len(X_test_text):,}")
    print(f"  Baseline accuracy: {baseline_metrics['baseline_accuracy']:.4f}")
    
    # ------------------------------------------------------------------
    # Step 2: Generate Surface Sentiment and Sarcasm Labels
    # ------------------------------------------------------------------
    print("\n[STEP 2/7] Generating surface sentiment and sarcasm labels...")
    step_start = time.time()
    
    train_text_series = pd.Series(X_train_text)
    test_text_series = pd.Series(X_test_text)
    
    # Surface-level sentiment (what the text "looks like")
    train_surface_sentiment = simulate_sentiment_labels_batch(
        train_text_series, seed=RANDOM_STATE
    )
    test_surface_sentiment = simulate_sentiment_labels_batch(
        test_text_series, seed=RANDOM_STATE
    )
    
    # Sarcasm labels
    y_sarcasm_train = simulate_sarcasm_labels_batch(
        train_text_series, train_surface_sentiment, seed=RANDOM_STATE
    )
    y_sarcasm_test = simulate_sarcasm_labels_batch(
        test_text_series, test_surface_sentiment, seed=RANDOM_STATE
    )
    
    sarcasm_label_time = time.time() - step_start
    
    # Display sarcasm distribution
    train_sarc_dist = y_sarcasm_train.value_counts()
    test_sarc_dist = y_sarcasm_test.value_counts()
    
    print(f"[DONE] Labels generated in {sarcasm_label_time:.2f}s")
    print(f"  Training sarcasm distribution:")
    for label, count in train_sarc_dist.items():
        pct = count / len(y_sarcasm_train) * 100
        tag = "Sarcastic" if label == 1 else "Not Sarcastic"
        print(f"    {tag:>15}: {count:>10,} ({pct:.1f}%)")
    print(f"  Test sarcasm distribution:")
    for label, count in test_sarc_dist.items():
        pct = count / len(y_sarcasm_test) * 100
        tag = "Sarcastic" if label == 1 else "Not Sarcastic"
        print(f"    {tag:>15}: {count:>10,} ({pct:.1f}%)")
    
    # ------------------------------------------------------------------
    # Step 3: Create TRUE Sentiment Labels (Sarcasm-Corrected)
    # ------------------------------------------------------------------
    print("\n[STEP 3/7] Creating sarcasm-corrected ground truth labels...")
    step_start = time.time()
    
    # The TRUE sentiment = flip the surface sentiment for sarcastic texts
    # This models: "when someone is sarcastic, they mean the opposite"
    def create_true_labels(surface_sentiments, sarcasm_labels):
        true_labels = []
        for sent, sarc in zip(surface_sentiments, sarcasm_labels):
            if sarc == 1:
                true_labels.append(flip_sentiment(sent))
            else:
                true_labels.append(sent)
        return pd.Series(true_labels)
    
    y_train_true = create_true_labels(train_surface_sentiment, y_sarcasm_train)
    y_test_true = create_true_labels(test_surface_sentiment, y_sarcasm_test)
    
    # Encode true labels
    true_label_encoder = LabelEncoder()
    true_label_encoder.fit(pd.concat([y_train_true, y_test_true]))
    y_train_true_enc = true_label_encoder.transform(y_train_true)
    y_test_true_enc = true_label_encoder.transform(y_test_true)
    
    label_create_time = time.time() - step_start
    
    # Show how labels shifted
    print(f"[DONE] True labels created in {label_create_time:.2f}s")
    train_true_dist = y_train_true.value_counts()
    print(f"  True sentiment distribution (train):")
    for label, count in train_true_dist.items():
        pct = count / len(y_train_true) * 100
        print(f"    {label:>10}: {count:>10,} ({pct:.1f}%)")
    
    n_flipped_train = np.sum(train_surface_sentiment != y_train_true)
    print(f"  Labels corrected by sarcasm: {n_flipped_train:,} "
          f"({n_flipped_train/len(y_train_true)*100:.1f}%)")
    
    # ------------------------------------------------------------------
    # Step 4: Train Sarcasm Classifier
    # ------------------------------------------------------------------
    print("\n[STEP 4/7] Training sarcasm classifier...")
    step_start = time.time()
    
    sarcasm_tfidf = TfidfVectorizer(
        max_features=SARCASM_TFIDF_MAX_FEATURES,
        ngram_range=SARCASM_TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        min_df=5,
        max_df=0.95,
        dtype=np.float32
    )
    
    X_train_sarcasm_tfidf = sarcasm_tfidf.fit_transform(X_train_text)
    X_test_sarcasm_tfidf = sarcasm_tfidf.transform(X_test_text)
    
    sarcasm_model = LogisticRegression(
        max_iter=SARCASM_LR_MAX_ITER,
        solver=SARCASM_LR_SOLVER,
        n_jobs=SARCASM_LR_N_JOBS,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        verbose=1
    )
    
    sarcasm_model.fit(X_train_sarcasm_tfidf, y_sarcasm_train.values)
    sarcasm_train_time = time.time() - step_start
    print(f"\n[DONE] Sarcasm model trained in {sarcasm_train_time:.2f}s")
    
    # ------------------------------------------------------------------
    # Step 5: Evaluate Sarcasm Classifier
    # ------------------------------------------------------------------
    print("\n[STEP 5/7] Evaluating sarcasm classifier...")
    step_start = time.time()
    
    y_sarcasm_pred = sarcasm_model.predict(X_test_sarcasm_tfidf)
    
    sarcasm_accuracy = accuracy_score(y_sarcasm_test, y_sarcasm_pred)
    sarcasm_precision = precision_score(y_sarcasm_test, y_sarcasm_pred,
                                         average='weighted', zero_division=0)
    sarcasm_recall = recall_score(y_sarcasm_test, y_sarcasm_pred,
                                   average='weighted', zero_division=0)
    sarcasm_f1 = f1_score(y_sarcasm_test, y_sarcasm_pred,
                           average='weighted', zero_division=0)
    sarcasm_conf_matrix = confusion_matrix(y_sarcasm_test, y_sarcasm_pred)
    sarcasm_class_report = classification_report(
        y_sarcasm_test, y_sarcasm_pred,
        target_names=['Not Sarcastic', 'Sarcastic'],
        zero_division=0
    )
    
    sarcasm_eval_time = time.time() - step_start
    
    print(f"  Sarcasm Classifier Metrics:")
    print(f"    Accuracy:  {sarcasm_accuracy:.4f}")
    print(f"    Precision: {sarcasm_precision:.4f}")
    print(f"    Recall:    {sarcasm_recall:.4f}")
    print(f"    F1 Score:  {sarcasm_f1:.4f}")
    print(f"\n  {sarcasm_class_report}")
    
    # ------------------------------------------------------------------
    # Step 6: Train Sarcasm-Aware Sentiment Model
    # ------------------------------------------------------------------
    print("\n[STEP 6/7] Training sarcasm-aware sentiment classifier...")
    step_start = time.time()
    
    # Use the same TF-IDF features + sarcasm prediction as extra feature
    # Get sarcasm predictions for training data too
    y_sarcasm_train_pred = sarcasm_model.predict(X_train_sarcasm_tfidf)
    
    # Build combined feature: TF-IDF from sarcasm vectorizer + sarcasm flag
    from scipy.sparse import csr_matrix
    
    sarcasm_feature_train = csr_matrix(y_sarcasm_train_pred.reshape(-1, 1)).astype(np.float32)
    sarcasm_feature_test = csr_matrix(y_sarcasm_pred.reshape(-1, 1)).astype(np.float32)
    
    X_train_combined = hstack([X_train_sarcasm_tfidf, sarcasm_feature_train])
    X_test_combined = hstack([X_test_sarcasm_tfidf, sarcasm_feature_test])
    
    # Train sentiment model on TRUE labels (sarcasm-corrected)
    sa_model = LogisticRegression(
        max_iter=SA_LR_MAX_ITER,
        solver=SA_LR_SOLVER,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    sa_model.fit(X_train_combined, y_train_true_enc)
    sa_train_time = time.time() - step_start
    print(f"\n[DONE] Sarcasm-aware model trained in {sa_train_time:.2f}s")
    
    # ------------------------------------------------------------------
    # Step 7: Evaluate Sarcasm-Aware Model & Compare
    # ------------------------------------------------------------------
    print("\n[STEP 7/7] Evaluating sarcasm-aware sentiment model...")
    step_start = time.time()
    
    y_pred_adjusted = sa_model.predict(X_test_combined)
    
    # Calculate adjusted metrics against TRUE labels
    adjusted_accuracy = accuracy_score(y_test_true_enc, y_pred_adjusted)
    adjusted_precision = precision_score(y_test_true_enc, y_pred_adjusted,
                                          average='weighted', zero_division=0)
    adjusted_recall = recall_score(y_test_true_enc, y_pred_adjusted,
                                    average='weighted', zero_division=0)
    adjusted_f1 = f1_score(y_test_true_enc, y_pred_adjusted,
                            average='weighted', zero_division=0)
    adjusted_conf_matrix = confusion_matrix(y_test_true_enc, y_pred_adjusted)
    adjusted_class_report = classification_report(
        y_test_true_enc, y_pred_adjusted,
        target_names=true_label_encoder.classes_,
        zero_division=0
    )
    
    # Also evaluate baseline against TRUE labels for fair comparison
    # The baseline was trained on surface sentiment, tested against surface sentiment
    # For fair comparison: re-evaluate baseline predictions against TRUE labels
    # First map baseline predictions to the true label encoding
    baseline_pred_labels = label_encoder.inverse_transform(y_pred_baseline)
    baseline_pred_true_enc = true_label_encoder.transform(baseline_pred_labels)
    
    baseline_vs_true_accuracy = accuracy_score(y_test_true_enc, baseline_pred_true_enc)
    
    eval_time = time.time() - step_start
    
    # ------------------------------------------------------------------
    # Print Results
    # ------------------------------------------------------------------
    total_time = time.time() - overall_start
    
    baseline_acc = baseline_metrics['baseline_accuracy']
    improvement = adjusted_accuracy - baseline_vs_true_accuracy
    improvement_pct = (improvement / baseline_vs_true_accuracy) * 100 if baseline_vs_true_accuracy > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"SENTIMENT ACCURACY COMPARISON")
    print(f"{'='*60}")
    print(f"  Baseline accuracy (surface labels):         {baseline_acc*100:.2f}%")
    print(f"  Baseline accuracy (vs true labels):         {baseline_vs_true_accuracy*100:.2f}%")
    print(f"  Sarcasm-Aware accuracy (vs true labels):    {adjusted_accuracy*100:.2f}%")
    print(f"  Improvement over baseline:                  {improvement_pct:+.2f}%")
    print(f"{'='*60}")
    
    print(f"\n  Sarcasm-Aware Classification Report:")
    print(f"  {adjusted_class_report}")
    
    # Sarcasm impact stats
    n_sarcastic = int(np.sum(y_sarcasm_test.values == 1))
    n_sarcastic_pred = int(np.sum(y_sarcasm_pred == 1))
    n_different = int(np.sum(baseline_pred_true_enc != y_pred_adjusted))
    
    print(f"\n  Sarcasm Impact:")
    print(f"    True sarcastic (test):         {n_sarcastic:,}")
    print(f"    Predicted sarcastic (test):    {n_sarcastic_pred:,}")
    print(f"    Predictions changed:           {n_different:,}")
    
    # ------------------------------------------------------------------
    # Save All Artifacts
    # ------------------------------------------------------------------
    print("\n[SAVING] Saving sarcasm model artifacts...")
    
    joblib.dump(sarcasm_model, os.path.join(MODEL_DIR, 'sarcasm_model.joblib'))
    joblib.dump(sarcasm_tfidf, os.path.join(MODEL_DIR, 'sarcasm_tfidf.joblib'))
    joblib.dump(sa_model, os.path.join(MODEL_DIR, 'sa_sentiment_model.joblib'))
    joblib.dump(true_label_encoder, os.path.join(MODEL_DIR, 'true_label_encoder.joblib'))
    
    sarcasm_adjusted_metrics = {
        # Sarcasm classifier metrics
        'sarcasm_accuracy': sarcasm_accuracy,
        'sarcasm_precision': sarcasm_precision,
        'sarcasm_recall': sarcasm_recall,
        'sarcasm_f1': sarcasm_f1,
        'sarcasm_conf_matrix': sarcasm_conf_matrix,
        'sarcasm_class_report': sarcasm_class_report,
        
        # Adjusted sentiment metrics (sarcasm-aware)
        'sarcasm_adjusted_accuracy': adjusted_accuracy,
        'adjusted_precision': adjusted_precision,
        'adjusted_recall': adjusted_recall,
        'adjusted_f1': adjusted_f1,
        'adjusted_conf_matrix': adjusted_conf_matrix,
        'adjusted_class_report': adjusted_class_report,
        
        # Comparison (fair: both vs true labels)
        'baseline_accuracy': baseline_acc,
        'baseline_vs_true_accuracy': baseline_vs_true_accuracy,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        
        # Sarcasm statistics
        'n_sarcastic_test': n_sarcastic,
        'n_sarcastic_pred': n_sarcastic_pred,
        'n_flipped': n_different,
        'sarcasm_rate_test': float(n_sarcastic / len(y_sarcasm_test)),
        
        # Predictions for dashboard
        'y_pred_adjusted': y_pred_adjusted,
        'y_sarcasm_pred': y_sarcasm_pred,
        'y_sarcasm_test': y_sarcasm_test.values,
        
        # Label distributions
        'train_sarcasm_dist': train_sarc_dist.to_dict(),
        'test_sarcasm_dist': test_sarc_dist.to_dict(),
        
        # Timing
        'timing': {
            'load_time': load_time,
            'sarcasm_label_time': sarcasm_label_time,
            'label_create_time': label_create_time,
            'sarcasm_train_time': sarcasm_train_time,
            'sarcasm_eval_time': sarcasm_eval_time,
            'sa_train_time': sa_train_time,
            'eval_time': eval_time,
            'total_time': total_time
        }
    }
    
    joblib.dump(sarcasm_adjusted_metrics,
                os.path.join(MODEL_DIR, 'sarcasm_adjusted_metrics.joblib'))
    
    print(f"\n[SAVED] Sarcasm artifacts saved to: {MODEL_DIR}")
    print(f"  - sarcasm_model.joblib")
    print(f"  - sarcasm_tfidf.joblib")
    print(f"  - sa_sentiment_model.joblib")
    print(f"  - true_label_encoder.joblib")
    print(f"  - sarcasm_adjusted_metrics.joblib")
    
    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")
    
    return sarcasm_adjusted_metrics


if __name__ == '__main__':
    metrics = main()
