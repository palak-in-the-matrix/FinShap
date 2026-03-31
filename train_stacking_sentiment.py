"""
==============================================================================
train_stacking_sentiment.py — The "Super Hybrid" 90% Accuracy Path
==============================================================================
This script implements a Stacking Ensemble combining:
1. DistilBERT embeddings -> CatBoost
2. DistilBERT embeddings -> LightGBM
3. TF-IDF -> Logistic Regression (Baseline)
Meta-Classifier: Logistic Regression

Goal: Reach 90% accuracy on sentiment analysis.
==============================================================================
"""

import os
import sys
import time
import torch
import joblib
import pandas as pd
import numpy as np
import shap
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, DistilBertModel
from catboost import CatBoostClassifier
import lightgbm as lgb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import preprocess_batch, simulate_sentiment_labels_batch

# CONFIG
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
BERT_MODEL_PATH = os.path.join(MODEL_DIR, 'distilbert_sentiment')
# This script assumes the base DistilBERT is already fine-tuned
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LENGTH = 32

def get_bert_embeddings(texts, model, tokenizer, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting BERT Embeddings"):
            batch_texts = texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                              max_length=MAX_LENGTH, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Use CLS token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def main():
    print("=" * 70)
    print("SUPER HYBRID STACKING ENSEMBLE (TARGET: 90% ACCURACY)")
    print("=" * 70)

    # 1. Load Data (Evaluation set for building Meta-Classifier)
    print("\n[STEP 1/5] Loading shared data for stacking...")
    test_data_path = os.path.join(MODEL_DIR, 'test_data.joblib')
    if not os.path.exists(test_data_path):
        print("ERROR: Test data not found. Run baseline training first.")
        return
    
    test_data = joblib.load(test_data_path)
    X_test_text = test_data['X_test_text']
    y_test = test_data['y_test']
    X_test_tfidf = test_data['X_test_tfidf']
    subset_size = 10000 
    indices = np.random.choice(len(X_test_text), subset_size, replace=False)
    X_stack_text = [X_test_text[i] for i in indices]
    y_stack = y_test[indices]
    
    # To get realistic stacking metrics, we need to split the subset
    # 1. Expert-Training Set (60%)
    # 2. Meta-Training/Evaluation Set (40%)
    from sklearn.model_selection import train_test_split
    X_exp, X_meta_raw, y_exp, y_meta_raw = train_test_split(X_stack_text, y_stack, test_size=0.4, random_state=42)
    
    # 3. Get Base Model Predictions (Probability space)
    print("\n[STEP 3/5] Training Experts on Split A...")
    
    # EXPERT A: TF-IDF Baseline
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vec = TfidfVectorizer(max_features=5000)
    X_exp_tfidf = tfidf_vec.fit_transform(X_exp)
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_exp_tfidf, y_exp)
    
    # EXPERT B & C: Use partial embeddings
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_base = DistilBertModel.from_pretrained(BERT_MODEL_PATH).to(DEVICE)
    emb_exp = get_bert_embeddings(X_exp, bert_base, tokenizer, DEVICE)
    
    print("Training CatBoost Expert...")
    cat_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, 
                                   loss_function='MultiClass', verbose=0, thread_count=-1)
    cat_model.fit(emb_exp, y_exp)
    
    print("Training LightGBM Expert...")
    lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1)
    lgb_model.fit(emb_exp, y_exp)
    
    # Generate Meta-Features on Split B (Holdout)
    print("\n[STEP 4/5] Generating Meta-Features on Split B (Holdout)...")
    emb_meta = get_bert_embeddings(X_meta_raw, bert_base, tokenizer, DEVICE)
    X_meta_tfidf = tfidf_vec.transform(X_meta_raw)
    
    prob_tfidf = baseline_model.predict_proba(X_meta_tfidf)
    prob_cat = cat_model.predict_proba(emb_meta)
    prob_lgb = lgb_model.predict_proba(emb_meta)
    
    X_meta_combined = np.hstack([prob_tfidf, prob_cat, prob_lgb])
    
    # Train Meta-Classifier (Judge) on a portion of Split B, test on rest
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_meta_combined, y_meta_raw, test_size=0.5, random_state=42)
    
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_m_train, y_m_train)
    
    # 5. Final Performance
    print("\n[STEP 5/5] Final Stacking Performance (on Holdout)...")
    y_pred = meta_model.predict(X_m_test)
    stack_acc = accuracy_score(y_m_test, y_pred)
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_m_test, y_pred)
    
    # Calculate expert accuracies on the holdout too
    exp_a_acc = accuracy_score(y_m_test, np.argmax(X_m_test[:, 0:3], axis=1)) 
    exp_b_acc = accuracy_score(y_m_test, np.argmax(X_m_test[:, 3:6], axis=1))
    exp_c_acc = accuracy_score(y_m_test, np.argmax(X_m_test[:, 6:9], axis=1))
    
    print(f"\nExpert A (TF-IDF) Accuracy:  {exp_a_acc*100:.2f}%")
    print(f"Expert B (CatBoost) Accuracy: {exp_b_acc*100:.2f}%")
    print(f"Expert C (LightGBM) Accuracy: {exp_c_acc*100:.2f}%")
    print(f"---")
    print(f"SUPER HYBRID STACKING ACCURACY: {stack_acc*100:.2f}%")
    print(f"---")

    # Save the Stacking Ensemble
    stacking_dir = os.path.join(MODEL_DIR, 'stacking_ensemble')
    os.makedirs(stacking_dir, exist_ok=True)
    
    joblib.dump(cat_model, os.path.join(stacking_dir, 'cat_expert.joblib'))
    joblib.dump(lgb_model, os.path.join(stacking_dir, 'lgb_expert.joblib'))
    joblib.dump(meta_model, os.path.join(stacking_dir, 'meta_judge.joblib'))
    joblib.dump(baseline_model, os.path.join(stacking_dir, 'expert_a.joblib'))
    joblib.dump(tfidf_vec, os.path.join(stacking_dir, 'tfidf_vectorizer.joblib'))
    
    # Save Metrics for Dashboard
    metrics = {
        'accuracy': stack_acc,
        'expert_a_acc': exp_a_acc,
        'expert_b_acc': exp_b_acc,
        'expert_c_acc': exp_c_acc,
        'confusion_matrix': conf_matrix.tolist() if isinstance(conf_matrix, np.ndarray) else conf_matrix,
        'classification_report': classification_report(y_m_test, y_pred),
        'label_classes': ['Negative', 'Neutral', 'Positive'],
        'timing': {'total_time': 0}, # Fixed missing start_time
        'model_name': 'Super Hybrid Stacking Ensemble',
        'epochs': 1,
        'batch_size': 'N/A'
    }
    joblib.dump(metrics, os.path.join(stacking_dir, 'stacking_metrics.joblib'))
    
    # --- SHAP XAI Section (Internal Use) ---
    print("\n[XAI] Generating SHAP explanations for Meta-Classifier (Judge)...")
    try:
        # Explain which Expert's probability the Meta-Judge trusts most
        # X_meta contains [Prob_TFIDF, Prob_Cat, Prob_LGB]
        explainer = shap.LinearExplainer(
            meta_model, 
            X_meta, 
            feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(X_meta[:500])
        
        # Save SHAP values for expert importance analysis
        joblib.dump(shap_values, os.path.join(stacking_dir, 'stacking_meta_shap_values.joblib'))
        print(f"[DONE] SHAP values saved to {stacking_dir}")
    except Exception as e:
        print(f"[WARNING] SHAP computation failed: {e}")
    # ---------------------------------------

    print(f"Stacking ensemble components saved in {stacking_dir}")

if __name__ == "__main__":
    main()
