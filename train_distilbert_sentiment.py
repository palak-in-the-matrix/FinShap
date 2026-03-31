"""
==============================================================================
train_distilbert_sentiment.py — DistilBERT Sentiment Classification
==============================================================================
Financial Text Sentiment Analysis with Sarcasm Detection & Explainable AI

Pipeline:
    Text → DistilBERT Tokenizer → DistilBERT Sequence Classification
    → Sentiment Prediction → Evaluation Metrics

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
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data_chunked, preprocess_batch,
    simulate_sentiment_labels_batch, format_metrics,
    DistilBertDataset
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r_wallstreetbets_big.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'distilbert_sentiment')
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
    print("DISTILBERT SENTIMENT TRAINING")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n[STEP 1/5] Loading and prepping data...")
    df = load_data_chunked(DATA_PATH, max_rows=5000) 
    df['clean_title'] = preprocess_batch(df['title'])
    df = df[df['clean_title'].str.len() > 0].reset_index(drop=True)
    df['sentiment'] = simulate_sentiment_labels_batch(df['clean_title'], seed=RANDOM_STATE)
    
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment'])
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['label']
    )
    
    print(f"Data ready. Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 2. Tokenizer and DataLoaders
    print("\n[STEP 2/5] Initializing Tokenizer and DataLoaders...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = DistilBertDataset(
        texts=train_df.clean_title.to_list(),
        labels=train_df.label.to_list(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    test_dataset = DistilBertDataset(
        texts=test_df.clean_title.to_list(),
        labels=test_df.label.to_list(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 3. Model, Optimizer, Scheduler
    print("\n[STEP 3/5] Initializing Model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
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
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    sentiment_metrics = {
        'accuracy': float(val_acc),
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'label_classes': list(label_encoder.classes_),
        'model_name': MODEL_NAME,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'timing': {
            'total_time': time.time() - overall_start
        }
    }
    
    joblib.dump(sentiment_metrics, os.path.join(MODEL_DIR, 'sentiment_metrics.joblib'))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    
    print("\nSentiment Training Complete!")
    print(class_report)
    print(f"Final Accuracy: {val_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
