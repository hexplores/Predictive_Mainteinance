
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the saved best model and validation data
model = joblib.load("models/best_model.joblib")

X_val = pd.read_csv(DATA_DIR / "X_val.csv")
y_val = pd.read_csv(DATA_DIR / "y_val.csv")["Machine failure"]

# Get predicted failure probabilities
y_prob = model.predict_proba(X_val)[:, 1]

results = []

# Compare different classification thresholds
for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True Positives": tp,
        "Missed Failures": fn,
        "False Alarms": fp,
    })

results_df = pd.DataFrame(results)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False,
)

print("\nSaved to outputs/training/threshold_comparison.csv")
