# 🏦 Bank Transaction Anomaly Detection

<p align="center">
      <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
      <img src="https://img.shields.io/badge/IDE-Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" />
</p>

> An educational Python project that analyzes bank transactions and flags unusual records for review using Isolation Forest.

## 📌 Project Overview

This project applies **unsupervised anomaly detection** to a bank transaction dataset. It uses transaction and customer-related features to identify records that differ from patterns learned across the dataset.

The main script, `fraud_detection.py`, loads a CSV file, prepares the available features, trains an Isolation Forest model, scores transactions, and saves the results.

> **Important:** The output identifies statistical anomalies—not confirmed fraud. An unusual transaction may be legitimate, and a fraudulent transaction may not be flagged. This dataset should not be treated as verified fraud data unless its labels and their origin have been independently confirmed.

## 🎯 Project Goals

- Explore and prepare transaction data.
- Create basic time-based features where timestamp columns are available.
- Detect unusual transactions without relying on a fraud label.
- Save a reusable preprocessing and model pipeline.
- Produce a scored CSV that can be reviewed and investigated.

## 🧰 Technology Stack

- **Python** — project language
- **Pandas** — loading and preparing transaction data
- **NumPy** — numerical operations
- **scikit-learn** — preprocessing and Isolation Forest
- **Joblib** — saving the trained pipeline

## 📁 Project Structure

```text
bank-fraud-detection/
├── data/
│   └── bank_transactions_data_2.csv
├── artifacts/
│   ├── anomaly_detection_pipeline.joblib
│   ├── data_summary.txt
│   └── transactions_scored.csv
├── main_train.py
└── README.md
```

The `data/` folder contains the downloaded dataset. The `artifacts/` folder is created by the script to hold the model, data summary, and scored transactions.

## ⚙️ Setup

### 1. Install Python

Use a supported Python 3 version and make sure `python` and `pip` are available from your terminal.

### 2. Create a project environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### 4. Download the dataset

Download the CSV from the [Kaggle dataset page](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection/data) and place it here:

```text
data/bank_transactions_data_2.csv
```

The script expects that exact path. If your downloaded file has a different name, update `DATA_PATH` near the top of `fraud_detection.py`.

## ▶️ Run the Project

From the project’s root directory, run:

```bash
python fraud_detection.py
```

The script checks that the CSV exists, loads the data, prints a basic inspection report, prepares features, fits the anomaly detector, scores the transactions, and writes its outputs to `artifacts/`.

## 🔍 What the Script Does

### 1. Loads and inspects the CSV

The script reports the number of rows and columns, lists column names, counts missing values, and reports duplicate rows.

It also saves a more detailed summary to:

```text
artifacts/data_summary.txt
```

This initial inspection helps reveal data-quality issues. It does not prove that the dataset is representative or suitable for real-world fraud detection.

### 2. Removes identifier and target-like columns

The script excludes configured identifiers, such as transaction, account, device, IP address, or merchant IDs. These can encourage a model to memorize records rather than learn patterns that generalize.

It also removes columns with names such as `IsFraud`, `fraud`, `label`, or `target`. This project is unsupervised, so it should not accidentally train on a target-like field.

Column names vary between datasets. Review the printed column list and adjust the configuration if your CSV uses different names.

### 3. Creates time-based features

If the expected timestamp columns are present, the script derives:

- `TransactionHour`
- `TransactionDayOfWeek`
- `TransactionMonth`
- `IsWeekend`
- `SecondsSincePreviousTransaction`

Invalid or negative time gaps are treated as missing values.

These features are only meaningful if the timestamps represent what their names imply. Confirm the dataset’s definitions before interpreting them.

### 4. Preprocesses numeric and categorical data

The script builds a scikit-learn pipeline:

- Numeric values are filled with their median and scaled.
- Categorical values are filled with their most frequent value and one-hot encoded.
- The transformed features are passed to the anomaly detector.

Keeping preprocessing and the model in one pipeline helps ensure the same transformations are applied when the saved model is reused.

### 5. Fits Isolation Forest

**Isolation Forest** is an unsupervised algorithm that identifies observations that are easier to isolate from the rest of the data. It returns:

- `1` for an inlier, or a record the model treats as relatively typical.
- `-1` for an outlier, or a record the model treats as relatively unusual.

The script converts those predictions into an `is_anomaly` column:

- `0` means the model did not flag the transaction.
- `1` means the model flagged it for review.

An anomaly flag is not a fraud verdict.

### 6. Saves scores and model artifacts

The script saves:

```text
artifacts/transactions_scored.csv
artifacts/anomaly_detection_pipeline.joblib
artifacts/data_summary.txt
```

The scored CSV contains the original transaction columns plus:

- **`is_anomaly`** — whether Isolation Forest flagged the record.
- **`anomaly_score`** — a relative score used to rank transactions by unusualness.

The script negates scikit-learn’s `decision_function` so larger `anomaly_score` values generally represent more unusual records. The score is **not a probability** and should not be described as a probability of fraud.

## ⚙️ Configuration

The script’s main settings appear near the top of `fraud_detection.py`:

```python
DATA_PATH = Path("data/bank_transactions_data_2.csv")
OUTPUT_DIR = Path("artifacts")

CONTAMINATION = 0.05
RANDOM_STATE = 42
```

### `DATA_PATH`

The location of the input CSV. Change it if your file uses another path or filename.

### `OUTPUT_DIR`

The folder where generated reports, scores, and the saved model are written.

### `CONTAMINATION`

The approximate fraction of records the Isolation Forest is configured to treat as outliers. The default `0.05` is an initial modeling assumption—not a measured fraud rate and not evidence that 5% of transactions are fraudulent.

Changing this parameter changes how many transactions are flagged. Choose it through documented investigation and validation, not by assuming it reflects reality.

### `RANDOM_STATE`

Controls randomization so the experiment is easier to reproduce with the same data and software setup.

## 📊 Interpreting the Results

The most unusual transactions appear at the top of `transactions_scored.csv`, sorted by `anomaly_score`.

Treat these records as **review candidates**. Anomaly detection can surface unusual patterns, but it does not establish why a transaction is unusual. Possible explanations include:

- A legitimate transaction that is uncommon for the dataset
- A data-entry or timestamp issue
- A rare but normal customer behavior
- A pattern that deserves investigation

Do not use the output by itself to block transactions, accuse customers, or make other consequential decisions.

## ⚠️ Limitations

### No verified fraud labels

This script does not train a supervised fraud classifier because it does not rely on an independently verified fraud-label column. Without ground-truth labels, standard supervised performance metrics cannot establish how well the model detects fraud.

If a dataset includes an `IsFraud` field, verify how it was created before using it for evaluation. Labels generated from another anomaly-detection model are not independent confirmation of fraud.

### Small or synthetic data

A small or synthetic dataset may not capture the complexity, class imbalance, customer behavior, or changing patterns found in real payment systems.

### No fraud probabilities

Isolation Forest’s scores are not calibrated fraud probabilities. Do not interpret `anomaly_score` as the likelihood that a transaction is fraudulent.

### No production safeguards

This is a learning project, not a production banking system. It does not include a real-time payment integration, human-review workflow, customer-risk controls, regulatory review, or production monitoring.

### Feature and timestamp assumptions

Feature names and timestamp meanings must be checked against the actual CSV. Incorrect assumptions can produce misleading scores.

## 🔒 Data and Security

- Do not commit private, regulated, or identifiable financial data to a public repository.
- Check the dataset license and permitted uses before redistributing it.
- Keep secrets and credentials out of source code.
- Limit access to data and model artifacts.
- Avoid using the model output as the sole basis for a consequential decision.

## 🚀 Possible Improvements

To extend the project responsibly:

1. Confirm the dataset’s schema, provenance, license, and label definitions.
2. Add data-validation checks for required columns, types, and valid ranges.
3. Compare Isolation Forest with other anomaly-detection methods.
4. Review flagged records with domain experts.
5. If verified labels become available, build a separate supervised workflow.
6. For labeled data, use a time-based test split when transaction chronology matters.
7. Evaluate precision, recall, F1-score, average precision, and the review workload at different thresholds.
8. Track model and data versions so experiments are reproducible.
9. Add tests for feature preparation and scoring.
10. Design a human-review process before considering actions based on model output.

## 📝 Key Takeaways

- This project uses **unsupervised anomaly detection**, not verified fraud classification.
- Isolation Forest flags records that differ from learned patterns.
- The anomaly score ranks unusualness; it is not a fraud probability.
- The `contamination` setting is a modeling assumption, not a known fraud rate.
- Results need investigation and validation before they can support meaningful claims.

## 📚 References

- [Kaggle: Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection/data)
- [scikit-learn: Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [scikit-learn: Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn: Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---
## 📄 License

Please refer to the repository for applicable licensing information.

## 👨‍💻 Author

**ursleeplessprogrammer**

*This repository is for educational exploration. Its anomaly flags are not confirmed fraud findings and should not be used as a standalone fraud-prevention system.*
