"""
==============================================================================
evaluate_full_pipeline.py — Sequential Sarcasm-Aware Evaluation
==============================================================================
This script measures the accuracy of the full logic flow:
Text -> Sarcasm Detector -> Base Sentiment -> (If Sarcastic: Flip) -> Results

It calculates the metric for the entire test set (instead of just live prediction).
==============================================================================
"""

import os
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from utils import load_data_chunked, preprocess_batch, simulate_sentiment_labels_batch, simulate_sarcasm_labels_batch, flip_sentiment

def main():
    print("="*60)
    print("DEMONSTRATING FULL SEQUENTIAL PIPELINE ACCURACY")
    print("Logic: Text -> Sarcasm -> Sentiment -> (Flip if Sarcastic)")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load Data
    DATA_PATH = 'r_wallstreetbets_big.csv'
    print(f"\n[1/4] Loading Test Data (200,000 rows)...")
    df = load_data_chunked(DATA_PATH, max_rows=200000)
    df['clean_title'] = preprocess_batch(df['title'])
    
    # 2. Generate Ground Truth (True Flipped Intent)
    # This is what the user *actually* meant
    df['literal_sentiment'] = simulate_sentiment_labels_batch(df['clean_title'], seed=42)
    df['is_sarcastic_truth'] = simulate_sarcasm_labels_batch(df['clean_title'], df['literal_sentiment'], seed=42)
    df['true_intent'] = df.apply(lambda x: flip_sentiment(x['literal_sentiment']) if x['is_sarcastic_truth'] == 1 else x['literal_sentiment'], axis=1)

    # 3. Load Models
    print("[2/4] Loading Pipeline Models...")
    try:
        # Load Baseline Sentiment Model
        sentiment_model = joblib.load('models/baseline_model.joblib')
        sentiment_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        
        # Load Sarcasm Detection Model
        sarcasm_model = joblib.load('models/sarcasm_model.joblib')
        sarcasm_vectorizer = joblib.load('models/sarcasm_tfidf.joblib')
        
        # Load Label Encoder
        label_enc = joblib.load('models/label_encoder.joblib')
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please check models/ directory contents.")
        return

    # 4. Execute Pipeline
    print("[3/4] Running Sequential Inference on 200,000 rows...")
    
    # Step A: Detect Sarcasm
    X_sarc = sarcasm_vectorizer.transform(df['clean_title'])
    sarc_preds = sarcasm_model.predict(X_sarc)
    
    # Step B: Detect Literal Sentiment
    X_sent = sentiment_vectorizer.transform(df['clean_title'])
    sent_preds_enc = sentiment_model.predict(X_sent)
    sent_preds = label_enc.inverse_transform(sent_preds_enc)
    
    # Step C: Apply Flip Logic (The USER logic)
    df['pipeline_result'] = [
        flip_sentiment(sent) if sarc == 1 else sent 
        for sent, sarc in zip(sent_preds, sarc_preds)
    ]
    
    # 5. Calculate Metrics
    print("[4/4] Calculating Final Pipeline Metrics...")
    acc = accuracy_score(df['true_intent'], df['pipeline_result'])
    prec = precision_score(df['true_intent'], df['pipeline_result'], average='weighted')
    rec = recall_score(df['true_intent'], df['pipeline_result'], average='weighted')
    f1 = f1_score(df['true_intent'], df['pipeline_result'], average='weighted')
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print("FINAL PIPELINE RESULTS (SARCASM-AWARE)")
    print("="*60)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"Time Taken: {end_time - start_time:.1f} seconds")
    print("="*60)
    
    print("\nDetailed Report:")
    print(classification_report(df['true_intent'], df['pipeline_result']))

if __name__ == "__main__":
    main()
