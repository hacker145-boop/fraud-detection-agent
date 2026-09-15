"""
Fraud Detection & Transaction Risk Agent - Training Pipeline
================================================================
Loads transactions_data.csv (10,000 transactions), trains the fraud
detection model, evaluates it, generates graphs, and saves everything
into a single model file: fraud_detection_model.pkl

Covers all 7 minimum requirements:
1. Transaction data processing
2. Fraud classification
3. Anomaly detection
4. Fraud probability score
5. Risk-level classification
6. Suspicious transaction identification
7. Explainable prediction

Usage:
    pip install pandas numpy scikit-learn matplotlib joblib
    python train_fraud_model.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_FILE = "transactions_data.csv"
MODEL_FILE = "fraud_detection_model.pkl"
RESULTS_FILE = "fraud_detection_results.csv"
GRAPH_DIR = "graphs"

import os
os.makedirs(GRAPH_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# REQUIREMENT 1: Transaction Data Processing
# ---------------------------------------------------------------------
def process_data(df, label_encoder=None):
    """Clean and prepare raw transaction data for modeling."""
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates(subset="transaction_id")

    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["merchant_category_encoded"] = label_encoder.fit_transform(df["merchant_category"])
    else:
        df["merchant_category_encoded"] = label_encoder.transform(df["merchant_category"])

    feature_cols = [
        "amount",
        "hour_of_day",
        "merchant_category_encoded",
        "location_is_new",
        "device_is_new",
        "transactions_last_hour",
    ]
    return df, feature_cols, label_encoder


# ---------------------------------------------------------------------
# REQUIREMENT 3: Anomaly Detection
# ---------------------------------------------------------------------
def detect_anomalies(df, feature_cols, iso_forest=None):
    if iso_forest is None:
        iso_forest = IsolationForest(contamination=0.1, random_state=RANDOM_SEED)
        iso_forest.fit(df[feature_cols])
    raw_flags = iso_forest.predict(df[feature_cols])
    df["anomaly_flag"] = (raw_flags == -1).astype(int)
    return df, iso_forest


# ---------------------------------------------------------------------
# REQUIREMENT 2 & 4: Fraud Classification + Probability Score
# ---------------------------------------------------------------------
def train_fraud_classifier(df, feature_cols):
    X = df[feature_cols]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Fraud is rare (~2.5% of transactions), so a model trained naively
    # tends to under-predict fraud (high accuracy, but misses real fraud
    # cases). We oversample the minority (fraud) class in the TRAINING
    # data only -- the test set stays untouched and realistic -- so the
    # model sees fraud patterns often enough to learn them properly.
    train_df = X_train.copy()
    train_df["is_fraud"] = y_train.values
    fraud_rows = train_df[train_df["is_fraud"] == 1]
    legit_rows = train_df[train_df["is_fraud"] == 0]
    fraud_oversampled = fraud_rows.sample(
        n=len(legit_rows), replace=True, random_state=RANDOM_SEED
    )
    balanced_train = pd.concat([legit_rows, fraud_oversampled]).sample(
        frac=1, random_state=RANDOM_SEED
    )
    X_train_bal = balanced_train[feature_cols]
    y_train_bal = balanced_train["is_fraud"]

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    clf.fit(X_train_bal, y_train_bal)

    y_proba = clf.predict_proba(X_test)[:, 1]
    # In fraud detection, missing real fraud (a false negative) is usually
    # far costlier than a false alarm (a false positive), so we lower the
    # decision threshold from the default 0.5 to 0.3 to catch more fraud,
    # accepting more false positives as a deliberate trade-off.
    DECISION_THRESHOLD = 0.3
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model test accuracy: {accuracy:.2%}\n")
    print("Classification report (decision threshold = 0.3):")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return clf, (X_test, y_test, y_pred, y_proba)


# ---------------------------------------------------------------------
# REQUIREMENT 5: Risk-Level Classification
# ---------------------------------------------------------------------
def assign_risk_level(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------
# REQUIREMENT 7: Explainable Prediction
# ---------------------------------------------------------------------
def explain_transaction(row):
    reasons = []
    if row["amount"] > 8000:
        reasons.append("unusually high amount")
    if row["location_is_new"] == 1:
        reasons.append("transaction from a new/unfamiliar location")
    if row["device_is_new"] == 1:
        reasons.append("transaction from a new device")
    if row["hour_of_day"] in [0, 1, 2, 3]:
        reasons.append("occurred at an unusual hour (late night)")
    if row["transactions_last_hour"] > 3:
        reasons.append("multiple transactions in a short time window")
    if row["anomaly_flag"] == 1:
        reasons.append("statistically anomalous pattern detected")
    return "Flagged due to: " + ", ".join(reasons) if reasons else "No strong risk indicators found."


# ---------------------------------------------------------------------
# SAVE / LOAD single model file
# ---------------------------------------------------------------------
def save_model(clf, iso_forest, label_encoder, feature_cols, path=MODEL_FILE):
    bundle = {
        "classifier": clf,
        "isolation_forest": iso_forest,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
    }
    joblib.dump(bundle, path)
    print(f"Saved full model bundle to '{path}'\n")


def load_model(path=MODEL_FILE):
    bundle = joblib.load(path)
    return (
        bundle["classifier"],
        bundle["isolation_forest"],
        bundle["label_encoder"],
        bundle["feature_cols"],
    )


# ---------------------------------------------------------------------
# GRAPHS
# ---------------------------------------------------------------------
def generate_graphs(clf, feature_cols, X_test, y_test, y_pred, y_proba, df):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#2563eb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # 3. Feature importance
    importances = clf.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_cols)[order], importances[order], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/feature_importance.png", dpi=150)
    plt.close()

    # 4. Risk level distribution
    risk_counts = df["risk_level"].value_counts().reindex(["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    ax.bar(risk_counts.index, risk_counts.values, color=colors)
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Risk Level Distribution (Full Dataset)")
    for i, v in enumerate(risk_counts.values):
        ax.text(i, v + max(risk_counts.values) * 0.01, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/risk_level_distribution.png", dpi=150)
    plt.close()

    # 5. Fraud probability histogram
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(df["fraud_probability"], bins=30, color="#2563eb", edgecolor="white")
    ax.set_xlabel("Fraud Probability Score")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Distribution of Fraud Probability Scores")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fraud_probability_distribution.png", dpi=150)
    plt.close()

    print(f"Saved 5 graphs to '{GRAPH_DIR}/' folder\n")


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def run_pipeline():
    print("=" * 60)
    print("FRAUD DETECTION & TRANSACTION RISK AGENT - TRAINING")
    print("=" * 60)

    # 1. Load + process data
    raw_df = pd.read_csv(DATA_FILE)
    df, feature_cols, label_encoder = process_data(raw_df)
    print(f"Loaded and processed {len(df)} transactions from '{DATA_FILE}'.\n")

    # 3. Anomaly detection
    df, iso_forest = detect_anomalies(df, feature_cols)

    # 2 & 4. Train classifier + get probability scores
    clf, (X_test, y_test, y_pred, y_proba) = train_fraud_classifier(df, feature_cols)
    df["fraud_probability"] = clf.predict_proba(df[feature_cols])[:, 1]

    # 5. Risk level
    df["risk_level"] = df["fraud_probability"].apply(assign_risk_level)

    # 7. Explainability
    df["explanation"] = df.apply(explain_transaction, axis=1)

    # 6. Suspicious transactions
    suspicious = df[df["risk_level"].isin(["High", "Medium"])].sort_values(
        "fraud_probability", ascending=False
    )
    print(f"Flagged {len(suspicious)} suspicious transactions out of {len(df)}.\n")

    # Save single model file
    save_model(clf, iso_forest, label_encoder, feature_cols)

    # Save results
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Full results saved to '{RESULTS_FILE}'")

    # Graphs
    generate_graphs(clf, feature_cols, X_test, y_test, y_pred, y_proba, df)

    return df, clf


if __name__ == "__main__":
    run_pipeline()
