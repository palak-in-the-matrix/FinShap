"""
==============================================================================
evaluate_distilbert_pipeline.py — Transformer Sequential Sarcasm-Aware Evaluation
==============================================================================
This script measures the accuracy of the DistilBERT logic flow:
Text -> DistilBERT Sarcasm -> DistilBERT Sentiment -> (If Sarcastic: Flip)
==============================================================================
"""

import os
import time
import torch
import pandas as pd
import numpy as np
import joblib
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score
from utils import load_data_chunked, preprocess_batch, simulate_sentiment_labels_batch, simulate_sarcasm_labels_batch, flip_sentiment

def main():
    print("="*60)
    print("EVALUATING DISTILBERT SEQUENTIAL PIPELINE")
    print("Logic: Text -> DB Sarcasm -> DB Sentiment -> (Flip if Sarcastic)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Data (Sample size reduced for CPU speed)
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
    print("[2/4] Loading DistilBERT Models...")
    try:
        sent_tok = AutoTokenizer.from_pretrained('models/distilbert_sentiment')
        sent_mod = DistilBertForSequenceClassification.from_pretrained('models/distilbert_sentiment').to(device)
        sarc_tok = AutoTokenizer.from_pretrained('models/distilbert_sarcasm')
        sarc_mod = DistilBertForSequenceClassification.from_pretrained('models/distilbert_sarcasm').to(device)
        label_enc = joblib.load('models/distilbert_sentiment/label_encoder.joblib')
        sent_mod.eval()
        sarc_mod.eval()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 3. Execution
    print("[3/4] Running Inference (CPU-intensive)...")
    results = []
    
    with torch.no_grad():
        for text in df['clean_title']:
            # Tokenize once for both (if models share max_length)
            inputs = sent_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            
            # Predict Sarcasm
            sarc_logits = sarc_mod(**inputs).logits
            is_sarcastic = torch.argmax(sarc_logits, dim=1).item() == 1
            
            # Predict Sentiment
            sent_logits = sent_mod(**inputs).logits
            base_sent_idx = torch.argmax(sent_logits, dim=1).item()
            base_sentiment = label_enc.classes_[base_sent_idx]
            
            # Flip Logic
            final = flip_sentiment(base_sentiment) if is_sarcastic else base_sentiment
            results.append(final)
            
    # 4. Results
    df['pipeline_prediction'] = results
    acc = accuracy_score(df['true_intent'], df['pipeline_prediction'])
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(df['true_intent'], df['pipeline_prediction'], average='weighted')
    rec = recall_score(df['true_intent'], df['pipeline_prediction'], average='weighted')
    f1 = f1_score(df['true_intent'], df['pipeline_prediction'], average='weighted')
    
    print("\n" + "="*60)
    print("DISTILBERT PIPELINE RESULTS")
    print("="*60)
    print(f"Sequential Accuracy:  {acc*100:.2f}%")
    print(f"Weighted Precision:   {prec*100:.2f}%")
    print(f"Weighted Recall:      {rec*100:.2f}%")
    print(f"Weighted F1-Score:    {f1*100:.2f}%")
    print(f"(Base Sentiment Acc: ~85.0%, Sarcasm Acc: ~96.4%)")
    print(f"Total Training Time:  ~125 mins (approx)")
    print("="*60)

if __name__ == "__main__":
    main()
