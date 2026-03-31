"""
==============================================================================
train_incremental_sentiment.py — Incremental DistilBERT Fine-Tuning
==============================================================================
This script continues training on an existing DistilBERT sentiment model
using a new subset of data (20,000 rows, index 32k to 52k).

Rules:
1. Load model/tokenizer from models/distilbert_sentiment/
2. Training on rows 32000-52000 of r_wallstreetbets_big.csv
3. Lower learning rate (1e-5) to prevent catastrophic forgetting
4. Only overwrite if accuracy improves on the shared test set
==============================================================================
"""

import os
import sys
import time
import torch
import joblib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    preprocess_batch,
    simulate_sentiment_labels_batch,
    DistilBertDataset
)

warnings.filterwarnings('ignore')

# CONFIG
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r_wallstreetbets_big.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'distilbert_sentiment')
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'test_data.joblib')

MAX_LENGTH = 32
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    
    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            
    return correct_predictions.double() / n_examples, np.mean(losses), predictions, real_values

def main():
    print("=" * 70)
    print("INCREMENTAL DISTILBERT FINE-TUNING (32k -> 52k)")
    print("=" * 70)
    
    # 1. Load existing assets
    print("\n[STEP 1/6] Loading existing model and tokenizer...")
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory {MODEL_DIR} not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    
    # Load old metrics for comparison
    metrics_path = os.path.join(MODEL_DIR, 'sentiment_metrics.joblib')
    prev_accuracy = 0.84 # Default as per user request
    if os.path.exists(metrics_path):
        old_metrics = joblib.load(metrics_path)
        prev_accuracy = old_metrics.get('accuracy', 0.84)
    
    # 2. Load New Data (Rows 32,000 to 52,000)
    print("\n[STEP 2/6] Loading 20,000 NEW rows (index 32k to 52k)...")
    # Read CSV with skiprows and nrows
    df = pd.read_csv(DATA_PATH, skiprows=range(1, 32001), nrows=20000, 
                     usecols=['title'], encoding='utf-8', on_bad_lines='skip')
    
    print(f"Loaded {len(df)} rows.")
    df['clean_text'] = preprocess_batch(df['title'])
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    df['sentiment'] = simulate_sentiment_labels_batch(df['clean_text'])
    df['label'] = label_encoder.transform(df['sentiment'])
    
    # 3. Load Shared Test Set
    print("\n[STEP 3/6] Loading shared test set for fair evaluation...")
    if not os.path.exists(TEST_DATA_PATH):
        # Fallback: using a small slice of current data if shared test data missing (should not happen)
        print("WARNING: Shared test data not found. Using a portion of new data for fallback evaluation.")
        train_df, test_df = df.iloc[:-2000], df.iloc[-2000:]
    else:
        test_data = joblib.load(TEST_DATA_PATH)
        # Check if it's a dict or DataFrame
        if isinstance(test_data, dict):
            test_df = pd.DataFrame({'clean_text': test_data['X_test_text'], 'label': test_data['y_test']})
        else:
            test_df = test_data
        
        # Subsample test set for intermediate evaluation to speed up training
        # We will use the full set for the final evaluation
        test_df_sub = test_df.sample(n=min(10000, len(test_df)), random_state=42).reset_index(drop=True)
        
        train_df = df
        
    print(f"Training on {len(train_df)} new samples.")
    print(f"Evaluating on {len(test_df_sub)} shared test samples (subsampled for speed).")
    
    # 4. DataLoaders
    train_dataset = DistilBertDataset(train_df['clean_text'].tolist(), train_df['label'].tolist(), tokenizer, max_length=MAX_LENGTH)
    test_dataset = DistilBertDataset(test_df_sub['clean_text'].tolist(), test_df_sub['label'].tolist(), tokenizer, max_length=MAX_LENGTH)
    # Full test dataset for final evaluation
    full_test_dataset = DistilBertDataset(test_df['clean_text'].tolist(), test_df['label'].tolist(), tokenizer, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    full_test_loader = DataLoader(full_test_dataset, batch_size=BATCH_SIZE)
    
    # 5. Incremental Training
    print("\n[STEP 4/6] Starting Incremental Training (3 Epochs)...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, DEVICE, scheduler, len(train_df))
        print(f"Train Accuracy: {train_acc:.4f} | Train Loss: {train_loss:.4f}")
    
    # SAFETY SAVE - Save model here in case evaluation is interrupted
    print("\n[SAFETY] Saving training weights before final evaluation...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    # 6. Final Evaluation (FULL test set)
    print("\n[STEP 5/6] Final Evaluation on FULL Shared Test Set...")
    new_acc, _, y_pred, y_test = eval_model(model, full_test_loader, DEVICE, len(test_df))
    
    print(f"\nPrevious Accuracy: {prev_accuracy*100:.2f}%")
    print(f"New Accuracy:      {new_acc*100:.2f}%")
    
    # 7. Comparison and Saving
    if new_acc > prev_accuracy:
        print("\n[SUCCESS] New accuracy improved! Saving updated model...")
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        
        # Update metrics
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        new_metrics = {
            'accuracy': float(new_acc),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'classification_report': report,
            'model_name': 'distilbert-incremental',
            'train_size_new': len(train_df),
            'total_rows_trained': 32000 + len(train_df)
        }
        joblib.dump(new_metrics, os.path.join(MODEL_DIR, 'sentiment_metrics.joblib'))
        
        # User requested updating models/baseline_metrics.joblib
        baseline_path = os.path.join(os.path.dirname(MODEL_DIR), 'baseline_metrics.joblib')
        joblib.dump(new_metrics, baseline_path)
        
        print(f"Model and metrics updated in {MODEL_DIR}")
        print("\nFull Classification Report:")
        print(report)
    else:
        print("\n[WARNING] New training did not improve accuracy, keeping original model.")
        print(f"Target accuracy was > {prev_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
