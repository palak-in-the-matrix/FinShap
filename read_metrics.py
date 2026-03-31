import joblib
import os
import json

paths = {
    "DistilBERT Sentiment": "d:/transformer/distill/palak/models/distilbert_sentiment/sentiment_metrics.joblib",
    "Sarcasm Detection": "d:/transformer/distill/palak/models/distilbert_sarcasm/sarcasm_metrics.joblib",
    "Hybrid (BERT + XGB)": "d:/transformer/distill/palak/models/hybrid_sentiment/hybrid_metrics.joblib",
    "Super Hybrid Stacking": "d:/transformer/distill/palak/models/stacking_ensemble/stacking_metrics.joblib"
}

results = {}

for name, path in paths.items():
    if os.path.exists(path):
        try:
            data = joblib.load(path)
            # data is usually a dict
            results[name] = data
        except Exception as e:
            results[name] = f"Error loading: {str(e)}"
    else:
        results[name] = f"File not found: {path}"

# Custom printer to handle non-serializable objects (like numpy arrays)
def default_serializer(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)

print(json.dumps(results, indent=2, default=default_serializer))
