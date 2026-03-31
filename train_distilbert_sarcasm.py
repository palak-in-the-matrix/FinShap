"""
==============================================================================
train_distilbert_sarcasm.py — DistilBERT Sarcasm Detection
==============================================================================
Financial Text Sentiment Analysis with Sarcasm Detection & Explainable AI

Author: Research Implementation (DistilBERT Conversion)
==============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import shap
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data_chunked, preprocess_batch,
    simulate_sentiment_labels_batch, simulate_sarcasm_labels_batch,
    DistilBertDataset
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r_wallstreetbets_big.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'distilbert_sarcasm')
os.makedirs(MODEL_DIR, exist_ok=True)

# DistilBERT parameters
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-5
RANDOM_STATE = 42

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    overall_start = time.time()
    
    print("=" * 70)
    print("DISTILBERT SARCASM TRAINING")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n[STEP 1/5] Loading and prepping data...")
    df = load_data_chunked(DATA_PATH, max_rows=5000) 
    df['clean_title'] = preprocess_batch(df['title'])
    df = df[df['clean_title'].str.len() > 0].reset_index(drop=True)
    
    # Generate labels
    df['sentiment'] = simulate_sentiment_labels_batch(df['clean_title'], seed=RANDOM_STATE)
    df['sarcasm'] = simulate_sarcasm_labels_batch(df['clean_title'], df['sentiment'], seed=RANDOM_STATE)
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['sarcasm']
    )
    
    print(f"Data ready. Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 2. Tokenizer and DataLoaders
    print("\n[STEP 2/5] Initializing Tokenizer and DataLoaders...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = DistilBertDataset(
        texts=train_df.clean_title.to_list(),
        labels=train_df.sarcasm.to_list(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    test_dataset = DistilBertDataset(
        texts=test_df.clean_title.to_list(),
        labels=test_df.sarcasm.to_list(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 3. Model, Optimizer, Scheduler
    print("\n[STEP 3/5] Initializing Model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # 4. Training Loop
    print("\n[STEP 4/5] Training loop...")
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, device, scheduler, len(train_df)
        )
        print(f"Train loss: {train_loss:.4f} accuracy: {train_acc:.4f}")
        
        val_acc, val_loss, _, _ = eval_model(
            model, test_loader, device, len(test_df)
        )
        print(f"Val loss: {val_loss:.4f} accuracy: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            best_accuracy = val_acc
            
    # 5. Final Evaluation and Saving
    print("\n[STEP 5/5] Final Evaluation...")
    best_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    val_acc, val_loss, y_pred, y_test = eval_model(
        best_model, test_loader, device, len(test_df)
    )
    
    # Metrics
    class_report = classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    sarcasm_metrics = {
        'accuracy': float(val_acc),
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'model_name': MODEL_NAME,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'timing': {
            'total_time': time.time() - overall_start
        }
    }
    
    joblib.dump(sarcasm_metrics, os.path.join(MODEL_DIR, 'sarcasm_metrics.joblib'))
    
    # --- SHAP XAI Section (Internal Use) ---
    print("\n[XAI] Generating SHAP explanations for Sarcasm Model...")
    try:
        def predict_proba(texts):
            inputs = tokenizer(texts.tolist() if isinstance(texts, np.ndarray) else texts, 
                              padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = best_model(**inputs).logits
                return torch.softmax(logits, dim=1).cpu().numpy()

        # Explain a small sample of test data
        explainer = shap.Explainer(predict_proba, tokenizer)
        shap_sample = test_df.clean_title.to_list()[:10]
        shap_values = explainer(shap_sample)
        
        # Save SHAP values for offline analysis
        joblib.dump(shap_values, os.path.join(MODEL_DIR, 'sarcasm_shap_values.joblib'))
        print(f"[DONE] SHAP values saved to {MODEL_DIR}")
    except Exception as e:
        print(f"[WARNING] SHAP computation failed: {e}")
    # ---------------------------------------

    print("\nSarcasm Training Complete!")
    print(class_report)
    print(f"Final Accuracy: {val_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
