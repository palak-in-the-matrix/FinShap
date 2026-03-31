import joblib
import os
import json

METRICS_FILES = {
    "Baseline": "models/baseline_metrics.joblib",
    "Sentiment": "models/distilbert_sentiment/sentiment_metrics.joblib",
    "Sarcasm": "models/distilbert_sarcasm/sarcasm_metrics.joblib",
    "Stacking": "models/stacking_ensemble/stacking_metrics.joblib"
}

base_path = "."

results = {}

for name, rel_path in METRICS_FILES.items():
    full_path = os.path.join(base_path, rel_path)
    if os.path.exists(full_path):
        try:
            m = joblib.load(full_path)
            # Handle possible non-serializable objects (like numpy arrays)
            def clean_dict(d):
                if isinstance(d, dict):
                    return {k: clean_dict(v) for k, v in d.items()}
                elif hasattr(d, 'tolist'):
                    return d.tolist()
                return d

            results[name] = {
                "accuracy": m.get('accuracy', 'N/A'),
                "precision": m.get('precision', 'N/A'),
                "recall": m.get('recall', 'N/A'),
                "f1_score": m.get('f1_score', 'N/A'),
                "timing": clean_dict(m.get('timing', 'N/A')),
                "confusion_matrix": clean_dict(m.get('confusion_matrix', 'N/A')),
                "classification_report": m.get('classification_report', 'N/A')
            }
        except Exception as e:
            results[name] = f"Error: {e}"
    else:
        results[name] = "File not found"

print(json.dumps(results, indent=2))
