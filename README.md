# Fraud Detection & Transaction Risk Agent

An AI-powered system that analyzes financial transactions and identifies potentially fraudulent activity, built for the HackForge FinTech Codeathon challenge.

**🔗 Live demo: https://fraud-detection-agent-codethon.streamlit.app/

## Problem

Financial systems process thousands of transactions, making it difficult to manually identify fraudulent activity, since fraudulent transactions often closely resemble legitimate ones. This project builds a machine learning pipeline that flags suspicious transactions automatically, scores them by fraud likelihood, classifies their risk level, and explains why each one was flagged.

## How It Meets the Requirements

| Requirement | Implementation |
|---|---|
| Transaction data processing | `process_data()` — cleans nulls/duplicates, encodes categorical fields |
| Fraud classification | `RandomForestClassifier`, trained on a class-balanced version of 10,000 labeled transactions |
| Anomaly detection | `IsolationForest` — unsupervised outlier detection |
| Fraud probability score | `predict_proba()` — outputs a 0–1 likelihood score per transaction |
| Risk-level classification | `assign_risk_level()` — buckets scores into Low / Medium / High |
| Suspicious transaction identification | Pipeline filters and ranks all Medium/High risk transactions |
| Explainable prediction | `explain_transaction()` — generates a plain-English reason per flag, based on which risk factors are present (high amount, new location/device, odd hour, transaction burst, anomaly flag) |

## Project Structure

```
|-- fraud-detection-risk-agent.pptx
├── app.py                         # Live interactive demo (Streamlit)
├── fraud_detection_notebook.ipynb # Full pipeline as a Jupyter/Colab notebook
├── transactions_data.csv         # 10,000-row training dataset
├── train_fraud_model.py           # Same pipeline as a plain .py script
├── fraud_detection_model.pkl      # Trained model (classifier + anomaly detector + encoder)
├── fraud_detection_results.csv    # Scored output for all 10,000 transactions
├── requirements.txt               # Python dependencies
├── graphs/                        # Evaluation charts
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── risk_level_distribution.png
│   └── fraud_probability_distribution.png
└── README.md
```

## How to Run

**Option A — Live demo app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens a browser page where you can enter transaction details and instantly see the fraud probability, risk level, and explanation.

**Option B — Notebook:
 Open `fraud_detection_notebook.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com), upload `transactions_data.csv` to the same session, then Run All.

**Option C — Script:
```bash
pip install -r requirements.txt
python train_fraud_model.py
```
## Model Performance & Design Decisions

- **Test accuracy:** 97.45% on a held-out 20% test split
- **Fraud recall:** ~66% — catches roughly 2 out of every 3 real fraud cases
- **Why not just optimize for accuracy?** Fraud is rare (~2.5% of transactions), so a model can score 98%+ accuracy while barely detecting any real fraud, simply by predicting "legit" most of the time. To avoid this trap:
  - We **oversample the fraud class** in the training data so the model sees enough fraud examples to learn real patterns
  - We **lower the decision threshold to 0.3** (instead of the default 0.5), since in fraud detection, missing real fraud (false negative) is usually costlier than a false alarm (false positive)
  - This trades some precision for meaningfully better fraud recall — a deliberate, explainable choice, not an oversight
- See `graphs/` for the confusion matrix, ROC curve (AUC ≈ 0.98), and feature importance breakdown

## Using the Saved Model on New Data (no retraining)

```python
import joblib, pandas as pd
from train_fraud_model import process_data, detect_anomalies, assign_risk_level, explain_transaction

bundle = joblib.load("fraud_detection_model.pkl")
clf, iso_forest, label_encoder, feature_cols = (
    bundle["classifier"], bundle["isolation_forest"],
    bundle["label_encoder"], bundle["feature_cols"]
)

new_df = pd.read_csv("new_transactions.csv")
df, _, _ = process_data(new_df, label_encoder=label_encoder)
df, _ = detect_anomalies(df, feature_cols, iso_forest=iso_forest)
df["fraud_probability"] = clf.predict_proba(df[feature_cols])[:, 1]
df["risk_level"] = df["fraud_probability"].apply(assign_risk_level)
df["explanation"] = df.apply(explain_transaction, axis=1)
```

## Dataset

`transactions_data.csv` contains 10,000 synthetic transactions with the fields: `transaction_id`, `amount`, `hour_of_day`, `merchant_category`, `location_is_new`, `device_is_new`, `transactions_last_hour`, and the ground-truth label `is_fraud`. It was generated to reflect realistic fraud patterns (large amounts, new devices/locations, late-night activity, transaction bursts). It is synthetic, not real transaction history — it can be swapped for a real dataset with the same schema.

## Tech Stack

- Python, pandas, numpy
- scikit-learn (RandomForestClassifier, IsolationForest)
- matplotlib (evaluation graphs)
- Streamlit (live interactive demo)
- joblib (model serialization)

