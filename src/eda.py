
# ============================================================
# PREDICTIVE MAINTENANCE ML PROJECT
# Exploratory Data Analysis (EDA)
# Dataset: AI4I 2020 Predictive Maintenance Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from contextlib import redirect_stdout
import sys

# ------------------------------------------------------------
# OUTPUT DIRECTORIES
# ------------------------------------------------------------

OUTPUT_DIR = Path("outputs/eda")
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "eda_report.txt"


# ------------------------------------------------------------
# HELPER FUNCTION: SAVE AND DISPLAY FIGURES
# ------------------------------------------------------------

def save_figure(filename):
    filepath = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {filepath}")

    plt.show()
    plt.close()


# ------------------------------------------------------------
# RUN EDA
# ------------------------------------------------------------

def run_eda():

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    data_path = Path("data/dataset.csv")
    df = pd.read_csv(data_path)

    df.columns = df.columns.str.strip()

    print("\n" + "=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset information:")
    df.info()

    # --------------------------------------------------------
    # 2. DATA TYPES
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("2. DATA TYPES")
    print("=" * 60)

    print(df.dtypes)

    print("\nNumerical columns:")
    print(df.select_dtypes(include=np.number).columns.tolist())

    print("\nCategorical columns:")
    print(df.select_dtypes(include=["object", "category"]).columns.tolist())

    # --------------------------------------------------------
    # 3. MISSING VALUES
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("3. MISSING VALUES")
    print("=" * 60)

    missing_values = df.isnull().sum()

    print(missing_values)
    print("\nTotal missing values:", df.isnull().sum().sum())

    # --------------------------------------------------------
    # 4. DUPLICATE RECORDS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("4. DUPLICATE RECORDS")
    print("=" * 60)

    print("Duplicate rows:", df.duplicated().sum())

    # --------------------------------------------------------
    # 5. SUMMARY STATISTICS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("5. SUMMARY STATISTICS")
    print("=" * 60)

    print(df.describe().round(2))

    # --------------------------------------------------------
    # 6. TARGET VARIABLE DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("6. MACHINE FAILURE DISTRIBUTION")
    print("=" * 60)

    target = "Machine failure"

    failure_counts = df[target].value_counts().sort_index()

    failure_percentages = (
        df[target].value_counts(normalize=True).sort_index() * 100
    )

    print("\nFailure counts:")
    print(failure_counts)

    print("\nFailure percentages:")
    print(failure_percentages.round(2))

    baseline_accuracy = failure_percentages.max()

    print(
        "\nMajority-class baseline accuracy:",
        round(baseline_accuracy, 2), "%"
    )

    plt.figure(figsize=(7, 5))

    ax = sns.countplot(data=df, x=target)

    plt.title("Distribution of Machine Failures")
    plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
    plt.ylabel("Number of Records")

    for container in ax.containers:
        ax.bar_label(container)

    save_figure("01_machine_failure_distribution.png")

    # --------------------------------------------------------
    # 7. FEATURE DISTRIBUTIONS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("7. FEATURE DISTRIBUTIONS")
    print("=" * 60)

    excluded_columns = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    feature_df = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    numeric_features = feature_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    print("Numeric sensor features:", numeric_features)

    axes = feature_df[numeric_features].hist(
        figsize=(14, 10),
        bins=30,
        edgecolor="black"
    )

    plt.gcf().suptitle(
        "Distribution of Numerical Features",
        fontsize=16
    )

    save_figure("02_numerical_feature_distributions.png")

    # --------------------------------------------------------
    # 8. CATEGORICAL FEATURE DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("8. CATEGORICAL FEATURE DISTRIBUTION")
    print("=" * 60)

    categorical_features = feature_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for i, column in enumerate(categorical_features, start=3):

        print(f"\n{column} counts:")
        print(df[column].value_counts())

        plt.figure(figsize=(7, 4))

        sns.countplot(data=df, x=column)

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")

        save_figure(
            f"{i:02d}_categorical_{column.lower().replace(' ', '_')}.png"
        )

    # --------------------------------------------------------
    # 9. CORRELATION HEATMAP
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("9. NUMERICAL FEATURE CORRELATIONS")
    print("=" * 60)

    correlation_data = df[
        numeric_features + [target]
    ].corr(numeric_only=True)

    print(correlation_data[target].sort_values(ascending=False))

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation_data,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    save_figure("04_correlation_heatmap.png")

    # --------------------------------------------------------
    # 10. SENSOR FEATURES VS MACHINE FAILURE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("10. SENSOR FEATURES VS MACHINE FAILURE")
    print("=" * 60)

    for i, column in enumerate(numeric_features, start=5):

        print(f"\n{column} grouped by machine failure:")

        print(
            df.groupby(target)[column]
            .agg(["mean", "median", "std"])
            .round(2)
        )

        plt.figure(figsize=(7, 5))

        sns.boxplot(
            data=df,
            x=target,
            y=column
        )

        plt.title(f"{column} vs Machine Failure")
        plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
        plt.ylabel(column)

        save_figure(
            f"{i:02d}_boxplot_{column.lower().replace(' ', '_')}.png"
        )

    # --------------------------------------------------------
    # 11. FAILURE MODE COUNTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("11. FAILURE MODE COUNTS")
    print("=" * 60)

    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    available_failure_modes = [
        column for column in failure_modes
        if column in df.columns
    ]

    if available_failure_modes:

        mode_counts = (
            df[available_failure_modes]
            .sum()
            .sort_values(ascending=False)
        )

        print(mode_counts)

        plt.figure(figsize=(8, 5))

        mode_counts.plot(kind="bar")

        plt.title("Number of Records by Failure Mode")
        plt.xlabel("Failure Mode")
        plt.ylabel("Number of Records")
        plt.xticks(rotation=0)

        save_figure("99_failure_mode_counts.png")

    # --------------------------------------------------------
    # 12. POTENTIAL OUTLIERS USING IQR
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("12. POTENTIAL NUMERICAL OUTLIERS")
    print("=" * 60)

    for column in numeric_features:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = (
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ).sum()

        print(f"{column}: {outlier_count} potential outliers")

    # --------------------------------------------------------
    # 13. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("13. EDA SUMMARY")
    print("=" * 60)

    print("Total records:", len(df))
    print("Total columns:", len(df.columns))
    print("Total missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    print("Failure records:", int((df[target] == 1).sum()))
    print("Non-failure records:", int((df[target] == 0).sum()))

    print(
        "Failure percentage:",
        round((df[target] == 1).mean() * 100, 2), "%"
    )

    print("\nEDA completed successfully!")


# ------------------------------------------------------------
# MAIN: SAVE ALL TEXT OUTPUT TO ONE FILE
# ------------------------------------------------------------

if __name__ == "__main__":

    with open(REPORT_PATH, "w", encoding="utf-8") as report:

        with redirect_stdout(report):
            run_eda()

    print(f"\nEDA report saved to: {REPORT_PATH}")
    print(f"EDA figures saved to: {FIGURES_DIR}")
