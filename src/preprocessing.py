
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

DATA_PATH = Path("data/dataset.csv")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Original dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. DEFINE TARGET AND REMOVE LEAKAGE COLUMNS
# ------------------------------------------------------------

target = "Machine failure"

# Identifiers do not provide useful generalizable sensor signals.
# Failure-mode labels can reveal information about the target.
columns_to_drop = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

X = df.drop(columns=[target] + columns_to_drop, errors="ignore")
y = df[target]

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts(normalize=True).mul(100).round(2))

# ------------------------------------------------------------
# 3. SPLIT INTO TRAIN, VALIDATION, AND TEST SETS
# ------------------------------------------------------------

# 70% training, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplit shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nFailure percentage in each split:")
for name, labels in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    print(f"{name}: {labels.mean() * 100:.2f}%")

# ------------------------------------------------------------
# 4. SAVE SPLITS
# ------------------------------------------------------------

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_val.to_csv(OUTPUT_DIR / "y_val.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("\nProcessed datasets saved to:", OUTPUT_DIR)
