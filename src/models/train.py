# src/models/train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from ..utils.constants import PROCESSED_DATA_DIR, MODEL_DIR, SELECTED_FEATURES, MODEL_CLASSES

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    X = df[SELECTED_FEATURES]
    y = df["label"].map({"asd": 1, "non_asd": 0})
    return X, y

def train_models():
    csv_path = os.path.join(PROCESSED_DATA_DIR, "1_MediaPipe_ASD_Behavior_Features.csv")
    print("Loading features from", csv_path)
    try:
        X, y = load_data(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load features: {str(e)}")
        return None, None

    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(**MODEL_CLASSES["LogisticRegression"]),
        "RandomForestClassifier": RandomForestClassifier(**MODEL_CLASSES["RandomForestClassifier"]),
        "SVC": SVC(**MODEL_CLASSES["SVC"]),
        "KNeighborsClassifier": KNeighborsClassifier(**MODEL_CLASSES["KNeighborsClassifier"]),
        "XGBClassifier": XGBClassifier(**MODEL_CLASSES["XGBClassifier"]),
        "MLPClassifier": MLPClassifier(**MODEL_CLASSES["MLPClassifier"])
    }

    results = []
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    print("\n=== Model Comparison ===")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
        
        results.append({
            "Model": name,
            "Test Accuracy": accuracy,
            "Precision (ASD)": report["1"]["precision"],
            "Recall (ASD)": report["1"]["recall"],
            "F1-Score (ASD)": report["1"]["f1-score"],
            "CV Mean Accuracy": cv_scores.mean(),
            "CV Std Accuracy": cv_scores.std()
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    print(f"\nBest Model: {best_model_name} with Test Accuracy: {best_accuracy:.4f}")

    try:
        with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        print("Saved best_model.pkl and scaler.pkl to", MODEL_DIR)
    except Exception as e:
        print(f"[ERROR] Failed to save model or scaler: {str(e)}")

    return best_model, scaler