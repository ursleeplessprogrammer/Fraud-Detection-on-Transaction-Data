from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path("data/bank_transactions_data_2.csv")
OUTPUT_DIR = Path("artifacts")
MODEL_PATH = OUTPUT_DIR / "anomaly_detection_pipeline.joblib"
SCORED_DATA_PATH = OUTPUT_DIR / "transactions_scored.csv"
SUMMARY_PATH = OUTPUT_DIR / "data_summary.txt"

# This is a modeling assumption, NOT the dataset's confirmed fraud rate.
CONTAMINATION = 0.05
RANDOM_STATE = 42

# Identifiers are excluded from the model to reduce memorization.
ID_COLUMNS = {
    "TransactionID",
    "AccountID",
    "DeviceID",
    "IP Address",
    "MerchantID",
}

DATE_COLUMNS = {
    "TransactionDate",
    "PreviousTransactionDate",
}


# ---------------------------------------------------------------------
# Data loading and inspection
# ---------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and check basic data quality."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the CSV from Kaggle and place it at that path."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("The CSV contains no rows.")

    data.columns = [column.strip() for column in data.columns]

    print(f"Loaded {len(data):,} rows and {len(data.columns):,} columns.")
    print("\nColumns:")
    print(data.columns.tolist())

    print("\nMissing values by column:")
    print(data.isna().sum().sort_values(ascending=False).to_string())

    duplicate_count = int(data.duplicated().sum())
    print(f"\nDuplicate rows: {duplicate_count:,}")

    return data


def write_data_summary(data: pd.DataFrame, path: Path) -> None:
    """Save a basic dataset summary for review."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(f"Rows: {len(data)}\n")
        file.write(f"Columns: {len(data.columns)}\n\n")
        file.write("Column names and data types:\n")
        file.write(data.dtypes.to_string())
        file.write("\n\nMissing values:\n")
        file.write(data.isna().sum().to_string())
        file.write("\n\nNumeric summary:\n")
        file.write(data.describe(include=[np.number]).to_string())
        file.write("\n")

    print(f"\nDataset summary saved to: {path}")


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features without using identifiers or a target label.

    Timestamp-derived features are calculated from transaction timestamps.
    If the dataset's timestamp semantics are unclear, review them before
    treating these features as meaningful.
    """
    features = data.copy()

    # Remove identifiers. They can encourage memorization and often do not
    # generalize to future transactions.
    features = features.drop(
        columns=[
            column
            for column in features.columns
            if column in ID_COLUMNS
        ],
        errors="ignore",
    )

    # Remove any existing target-like column rather than accidentally using
    # it as an input. This script is unsupervised.
    target_like_columns = [
        column
        for column in features.columns
        if column.lower() in {"isfraud", "fraud", "label", "target"}
    ]
    if target_like_columns:
        warnings.warn(
            "Dropping target-like columns from unsupervised features: "
            + ", ".join(target_like_columns)
        )
        features = features.drop(columns=target_like_columns)

    parsed_dates = {}
    for column in DATE_COLUMNS:
        if column in features.columns:
            parsed_dates[column] = pd.to_datetime(
                features[column],
                errors="coerce",
                utc=True,
            )

    transaction_date = parsed_dates.get("TransactionDate")
    previous_date = parsed_dates.get("PreviousTransactionDate")

    if transaction_date is not None:
        features["TransactionHour"] = transaction_date.dt.hour
        features["TransactionDayOfWeek"] = transaction_date.dt.dayofweek
        features["TransactionMonth"] = transaction_date.dt.month
        features["IsWeekend"] = (
            transaction_date.dt.dayofweek >= 5
        ).astype("float")

    if transaction_date is not None and previous_date is not None:
        elapsed_seconds = (
            transaction_date - previous_date
        ).dt.total_seconds()

        # Negative time gaps are treated as invalid and made missing.
        features["SecondsSincePreviousTransaction"] = elapsed_seconds.where(
            elapsed_seconds >= 0
        )

    features = features.drop(
        columns=[column for column in DATE_COLUMNS if column in features.columns],
        errors="ignore",
    )

    # Drop entirely empty columns.
    features = features.dropna(axis=1, how="all")

    if features.empty:
        raise ValueError("No usable feature columns remain.")

    return features


# ---------------------------------------------------------------------
# Preprocessing and model
# ---------------------------------------------------------------------

def build_pipeline(features: pd.DataFrame) -> Pipeline:
    """Build preprocessing and Isolation Forest as one reusable pipeline."""
    numeric_columns = features.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column for column in features.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    detector = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("detector", detector),
        ]
    )


# ---------------------------------------------------------------------
# Scoring and reporting
# ---------------------------------------------------------------------

def score_transactions(
    pipeline: Pipeline,
    features: pd.DataFrame,
    original_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add anomaly flags and scores.

    Isolation Forest returns:
      1  = inlier
     -1  = outlier

    Higher decision_function values are generally more normal.
    """
    predictions = pipeline.predict(features)
    decision_scores = pipeline.decision_function(features)

    results = original_data.copy()
    results["is_anomaly"] = (predictions == -1).astype(int)

    # Larger values mean more unusual for easier sorting.
    results["anomaly_score"] = -decision_scores

    return results.sort_values(
        "anomaly_score",
        ascending=False,
    ).reset_index(drop=True)


def print_report(results: pd.DataFrame) -> None:
    """Print a summary and a small sample of the most unusual rows."""
    total = len(results)
    flagged = int(results["is_anomaly"].sum())
    flagged_rate = flagged / total if total else 0.0

    print("\n" + "=" * 60)
    print("ANOMALY-DETECTION SUMMARY")
    print("=" * 60)
    print(f"Transactions analyzed: {total:,}")
    print(f"Transactions flagged:  {flagged:,}")
    print(f"Flagged share:         {flagged_rate:.2%}")
    print("\nMost unusual transactions:")
    print(results.head(10).to_string(index=False))

    print(
        "\nReminder: these are anomaly flags, not verified fraud findings. "
        "Review them with appropriate domain knowledge."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(DATA_PATH)
    write_data_summary(data, SUMMARY_PATH)

    features = prepare_features(data)
    print(f"\nFeatures used: {features.columns.tolist()}")

    pipeline = build_pipeline(features)
    pipeline.fit(features)

    results = score_transactions(
        pipeline=pipeline,
        features=features,
        original_data=data,
    )

    results.to_csv(SCORED_DATA_PATH, index=False)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": features.columns.tolist(),
            "contamination": CONTAMINATION,
            "warning": (
                "Anomaly detection output; not verified fraud probability."
            ),
        },
        MODEL_PATH,
    )

    print_report(results)
    print(f"\nScored CSV saved to: {SCORED_DATA_PATH}")
    print(f"Model bundle saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
