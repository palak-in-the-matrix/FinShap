import joblib
import os
import json

path = "d:/transformer/distill/palak/models/stacking_ensemble/sarcasm_aware_metrics.joblib"
if os.path.exists(path):
    try:
        data = joblib.load(path)
        def default_serializer(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return str(obj)
        print(json.dumps(data, indent=2, default=default_serializer))
    except Exception as e:
        print(f"Error loading: {e}")
else:
    print(f"File not found: {path}")
