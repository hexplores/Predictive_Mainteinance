
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Load model and data
# --------------------------------------------------
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load("models/best_model.joblib")

X_val = pd.read_csv(DATA_DIR / "X_val.csv")
y_val = pd.read_csv(DATA_DIR / "y_val.csv")["Machine failure"]

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Machine failure"]

# --------------------------------------------------
# 2. Select threshold using validation data only
# --------------------------------------------------
val_prob = model.predict_proba(X_val)[:, 1]

minimum_recall = 0.90
thresholds = np.arange(0.01, 1.00, 0.01)
results = []

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, val_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_pred, zero_division=0
        ),
        "F1": f1_score(
            y_val, val_pred, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)
eligible = results_df[
    results_df["Recall"] >= minimum_recall
]

if eligible.empty:
    raise ValueError(
        "No threshold achieved the required validation recall. "
        "Review the model or the recall requirement."
    )

# Among thresholds meeting the recall requirement,
# choose the one with the highest precision.
best = eligible.sort_values(
    ["Precision", "F1"],
    ascending=False
).iloc[0]

selected_threshold = float(best["Threshold"])

print("Selected using validation data")
print(f"Threshold: {selected_threshold:.2f}")
print(f"Validation precision: {best['Precision']:.4f}")
print(f"Validation recall: {best['Recall']:.4f}")
print(f"Validation F1: {best['F1']:.4f}")

# Save the threshold so deployment can reuse it
(OUTPUT_DIR / "selected_threshold.txt").write_text(
    f"{selected_threshold:.2f}\n"
)

# --------------------------------------------------
# 3. Final evaluation on the untouched test set
# --------------------------------------------------
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= selected_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print(f"Threshold: {selected_threshold:.2f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    test_pred,
    zero_division=0,
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1],
))

print(
    "Test PR-AUC:",
    round(average_precision_score(y_test, test_prob), 4),
)
print(
    "Test ROC-AUC:",
    round(roc_auc_score(y_test, test_prob), 4),
)

# Save final test metrics
test_results = pd.DataFrame([{
    "Threshold": selected_threshold,
    "Precision": precision_score(
        y_test, test_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_pred, zero_division=0
    ),
    "F1": f1_score(
        y_test, test_pred, zero_division=0
    ),
    "PR-AUC": average_precision_score(y_test, test_prob),
    "ROC-AUC": roc_auc_score(y_test, test_prob),
}])

test_results.to_csv(
    OUTPUT_DIR / "final_test_results.csv",
    index=False,
)

print("\nSaved threshold and test results in:", OUTPUT_DIR)
