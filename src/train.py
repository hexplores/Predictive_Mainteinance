
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Paths and output folders
# --------------------------------------------------
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/training")
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Machine failure"

# --------------------------------------------------
# 2. Load the processed data
# --------------------------------------------------
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val = pd.read_csv(DATA_DIR / "X_val.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET]
y_val = pd.read_csv(DATA_DIR / "y_val.csv")[TARGET]

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("\nTraining failure distribution:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

# --------------------------------------------------
# 3. Identify numerical and categorical columns
# --------------------------------------------------
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# --------------------------------------------------
# 4. Preprocess the features
# --------------------------------------------------
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# --------------------------------------------------
# 5. Define the models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
report_lines = []

# --------------------------------------------------
# 6. Train and evaluate each model on validation data
# --------------------------------------------------
for model_name, classifier in models.items():
    print(f"\n{'=' * 55}")
    print(f"Training: {model_name}")
    print("=" * 55)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    # Predict classes and failure probabilities
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
    }
    results.append(metrics)

    # Print evaluation results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Save confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Failure"],
        yticklabels=["Actual Normal", "Actual Failure"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    plt.savefig(FIGURE_DIR / f"{filename}_confusion_matrix.png", dpi=300)
    plt.close()

    # Save the trained pipeline
    joblib.dump(pipeline, MODEL_DIR / f"{filename}.joblib")

    report_lines.append(
        f"\n{model_name}\n"
        f"{'-' * 40}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"PR-AUC: {pr_auc:.4f}\n"
        f"ROC-AUC: {roc_auc:.4f}\n\n"
        f"{classification_report(y_val, y_pred, zero_division=0)}\n"
    )

# --------------------------------------------------
# 7. Compare the models
# --------------------------------------------------
results_df = pd.DataFrame(results).sort_values(
    by="PR-AUC", ascending=False
)

print("\n\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# Save the complete evaluation report
with open(OUTPUT_DIR / "training_report.txt", "w") as file:
    file.write("\n".join(report_lines))
    file.write("\n\nMODEL COMPARISON\n")
    file.write(results_df.round(4).to_string(index=False))

# Save the best model according to validation PR-AUC
best_model_name = results_df.iloc[0]["Model"]
best_filename = best_model_name.lower().replace(" ", "_")

best_model = joblib.load(MODEL_DIR / f"{best_filename}.joblib")
joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

print(f"\nBest model based on validation PR-AUC: {best_model_name}")
print(f"Models saved in: {MODEL_DIR}")
print(f"Reports saved in: {OUTPUT_DIR}")
print(f"Confusion matrices saved in: {FIGURE_DIR}")
