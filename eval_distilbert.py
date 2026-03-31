"""
==============================================================================
eval_distilbert.py — Standalone Evaluation Script
==============================================================================
Evaluates the saved DistilBERT model on the full shared test set.
Does NOT retrain — just loads and evaluates.
==============================================================================
"""
import os, sys, torch, joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, DistilBertForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DistilBertDataset

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'distilbert_sentiment')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_LENGTH = 32

def main():
    print("=" * 70)
    print("DISTILBERT EVALUATION (Full Shared Test Set)")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    # Load test data
    print("[2/3] Loading shared test set...")
    test_data = joblib.load(os.path.join(os.path.dirname(MODEL_DIR), 'test_data.joblib'))
    X_test = test_data['X_test_text']
    y_test = test_data['y_test']
    print(f"Test set size: {len(X_test)} samples")

    # Evaluate
    print("[3/3] Evaluating...")
    dataset = DistilBertDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    acc = accuracy_score(y_test, all_preds)
    print(f"\n{'='*70}")
    print(f"FINAL DISTILBERT ACCURACY: {acc*100:.2f}%")
    print(f"{'='*70}")
    print(classification_report(y_test, all_preds, target_names=['Negative', 'Neutral', 'Positive']))

    # Save metrics
    metrics = {
        'accuracy': acc,
        'classification_report': classification_report(y_test, all_preds, target_names=['Negative', 'Neutral', 'Positive']),
        'confusion_matrix': confusion_matrix(y_test, all_preds).tolist()
    }
    joblib.dump(metrics, os.path.join(MODEL_DIR, 'sentiment_metrics.joblib'))
    print(f"Metrics saved to {MODEL_DIR}/sentiment_metrics.joblib")

if __name__ == "__main__":
    main()
