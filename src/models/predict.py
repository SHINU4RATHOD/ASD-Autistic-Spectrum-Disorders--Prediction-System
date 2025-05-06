# src/models/predict.py
import numpy as np
import pandas as pd
import pickle
import os
from ..utils.constants import MODEL_DIR, OUTPUT_DIR, SELECTED_FEATURES

def predict_asd(features, model, scaler):
    X = pd.DataFrame([features])[SELECTED_FEATURES]
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    asd_prob = probability[1]
    return "asd" if prediction == 1 else "non_asd", asd_prob

def load_model_and_scaler():
    try:
        with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"[ERROR] Failed to load model or scaler: {str(e)}")
        return None, None