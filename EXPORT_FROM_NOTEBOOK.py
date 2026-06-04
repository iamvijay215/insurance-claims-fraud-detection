# Run this cell/code in your Jupyter Notebook AFTER best_model and feature_names are created.

import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/fraud_model_xgboost.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

# Optional: save scaler too, although XGBoost does not need scaling.
try:
    joblib.dump(scaler, "models/feature_scaler.pkl")
except NameError:
    print("Scaler not found; skipping scaler export.")

print("Saved model files successfully.")
