"""
==============================================================================
eval_stacking_sarcasm.py - Sequential Sarcasm-Aware Evaluation
==============================================================================
"""

import os
import time
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertModel, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_data_chunked, preprocess_batch, simulate_sentiment_labels_batch, simulate_sarcasm_labels_batch, flip_sentiment

def get_bert_embeddings(texts, model, tokenizer, device, max_length=128):
    model.eval()
    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                              max_length=max_length, return_tensors="pt").to(device)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def main():
    print("=" * 70)
    print("EVALUATING STACKING ENSEMBLE + SARCASM PIPELINE")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Data
    SAMPLE_SIZE = 2000
    DATA_PATH = 'r_wallstreetbets_big.csv'
    print(f"\n[1/4] Loading {SAMPLE_SIZE} Test Rows...")
    df = load_data_chunked(DATA_PATH, max_rows=SAMPLE_SIZE)
    df['clean_title'] = preprocess_batch(df['title'])
    
    # Generate Ground Truth
    df['literal_sentiment'] = simulate_sentiment_labels_batch(df['clean_title'], seed=42)
    df['is_sarcastic_truth'] = simulate_sarcasm_labels_batch(df['clean_title'], df['literal_sentiment'], seed=42)
    df['true_intent'] = df.apply(lambda x: flip_sentiment(x['literal_sentiment']) if x['is_sarcastic_truth'] == 1 else x['literal_sentiment'], axis=1)

    # 2. Load Models
    print("[2/4] Loading Models...")
    try:
        # Sarcasm
        sarc_tok = AutoTokenizer.from_pretrained('models/distilbert_sarcasm')
        sarc_mod = DistilBertForSequenceClassification.from_pretrained('models/distilbert_sarcasm').to(device)
        sarc_mod.eval()
        
        # Stacking
        stacking_dir = 'models/stacking_ensemble'
        cat_model = joblib.load(os.path.join(stacking_dir, 'cat_expert.joblib'))
        lgb_model = joblib.load(os.path.join(stacking_dir, 'lgb_expert.joblib'))
        meta_model = joblib.load(os.path.join(stacking_dir, 'meta_judge.joblib'))
        expert_a = joblib.load(os.path.join(stacking_dir, 'expert_a.joblib'))
        tfidf_vec = joblib.load(os.path.join(stacking_dir, 'tfidf_vectorizer.joblib'))
        
        # Base BERT for embeddings
        bert_base = DistilBertModel.from_pretrained('models/distilbert_sentiment').to(device)
        bert_base.eval()
        sent_tok = AutoTokenizer.from_pretrained('models/distilbert_sentiment')
        
        # Label Encoder
        label_enc = joblib.load('models/distilbert_sentiment/label_encoder.joblib')
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 3. Execution
    print("[3/4] Running Inference (CPU-intensive)...")
    start_time = time.time()
    results = []
    
    texts = df['clean_title'].tolist()
    
    # Step A: Predict Sarcasm
    print("Predicting Sarcasm...")
    sarcasm_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 32)):
            batch = texts[i:i+32]
            inputs = sarc_tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            logits = sarc_mod(**inputs).logits
            preds = (torch.argmax(logits, dim=1) == 1).cpu().numpy()
            sarcasm_preds.extend(preds)
            
    # Step B: Get Stacking Predictions
    print("Extracting DistilBERT Embeddings...")
    embeddings = get_bert_embeddings(texts, bert_base, sent_tok, device, max_length=32)
    
    print("Running Experts and Meta-Classifier...")
    # TF-IDF
    X_tfidf = tfidf_vec.transform(texts)
    prob_tfidf = expert_a.predict_proba(X_tfidf)
    
    # CatBoost & LightGBM
    prob_cat = cat_model.predict_proba(embeddings)
    prob_lgb = lgb_model.predict_proba(embeddings)
    
    # Meta Judge
    X_meta = np.hstack([prob_tfidf, prob_cat, prob_lgb])
    sent_preds_idx = meta_model.predict(X_meta)
    
    # 4. Results
    print("[4/4] Calculating Results...")
    
    final_preds = []
    for sent_idx, is_sarcastic in zip(sent_preds_idx, sarcasm_preds):
        base_sent = label_enc.classes_[sent_idx]
        final = flip_sentiment(base_sent) if is_sarcastic else base_sent
        final_preds.append(final)
        
    df['pipeline_prediction'] = final_preds
    
    acc = accuracy_score(df['true_intent'], df['pipeline_prediction'])
    prec = precision_score(df['true_intent'], df['pipeline_prediction'], average='weighted')
    rec = recall_score(df['true_intent'], df['pipeline_prediction'], average='weighted')
    f1 = f1_score(df['true_intent'], df['pipeline_prediction'], average='weighted')
    
    end_time = time.time()
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    rpt = classification_report(df['true_intent'], df['pipeline_prediction'])
    cm = confusion_matrix(df['true_intent'], df['pipeline_prediction'], labels=label_enc.classes_)
    
    print("\n" + "="*60)
    print("STACKING ENSEMBLE + SARCASM PIPELINE RESULTS")
    print("="*60)
    print(f"Sequential Accuracy:  {acc*100:.2f}%")
    print(f"Weighted Precision:   {prec*100:.2f}%")
    print(f"Weighted Recall:      {rec*100:.2f}%")
    print(f"Weighted F1-Score:    {f1*100:.2f}%")
    print(f"Sample Size:          {SAMPLE_SIZE}")
    print(f"Inference Time:       {end_time - start_time:.2f} seconds")
    print("="*60)
    
    # Save metrics for Dashboard
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'classification_report': rpt,
        'confusion_matrix': cm,
        'label_classes': label_enc.classes_.tolist(),
        'sample_size': SAMPLE_SIZE,
        'inference_time': end_time - start_time
    }
    joblib.dump(metrics, 'models/stacking_ensemble/sarcasm_aware_metrics.joblib')
    print(f"\n[DONE] Pipeline metrics saved to: models/stacking_ensemble/sarcasm_aware_metrics.joblib")

if __name__ == "__main__":
    main()
