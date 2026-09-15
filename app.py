"""
Fraud Detection & Transaction Risk Agent - Interactive Demo
================================================================
A live web app where you enter a transaction and instantly see its
fraud probability, risk level, and an explanation of why it was flagged.

Run locally:
    pip install streamlit pandas numpy scikit-learn joblib
    streamlit run app.py

Deploy for free (public link) via Streamlit Community Cloud:
    1. Push this repo to GitHub (must include app.py, fraud_detection_model.pkl,
       requirements.txt)
    2. Go to https://share.streamlit.io, sign in with GitHub
    3. Click "New app", select your repo, set main file to "app.py", deploy
    4. You'll get a public URL like https://yourapp.streamlit.app
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection Agent", page_icon="🛡️", layout="centered")

MODEL_FILE = "fraud_detection_model.pkl"


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_FILE)
    return (
        bundle["classifier"],
        bundle["isolation_forest"],
        bundle["label_encoder"],
        bundle["feature_cols"],
    )


def assign_risk_level(prob):
    if prob >= 0.7:
        return "High", "🔴"
    elif prob >= 0.3:
        return "Medium", "🟠"
    else:
        return "Low", "🟢"


def explain_transaction(amount, location_is_new, device_is_new, hour_of_day,
                         transactions_last_hour, anomaly_flag):
    reasons = []
    if amount > 8000:
        reasons.append("unusually high amount")
    if location_is_new:
        reasons.append("transaction from a new/unfamiliar location")
    if device_is_new:
        reasons.append("transaction from a new device")
    if hour_of_day in [0, 1, 2, 3]:
        reasons.append("occurred at an unusual hour (late night)")
    if transactions_last_hour > 3:
        reasons.append("multiple transactions in a short time window")
    if anomaly_flag:
        reasons.append("statistically anomalous pattern detected")
    return reasons if reasons else ["No strong risk indicators found."]


st.title("🛡️ Fraud Detection & Transaction Risk Agent")
st.write(
    "Enter a transaction's details below to get an instant fraud risk assessment, "
    "powered by the trained model (Random Forest + Isolation Forest)."
)

try:
    clf, iso_forest, label_encoder, feature_cols = load_model()
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_FILE}' not found. Make sure it's in the same "
        "folder as this app (or the same GitHub repo, if deployed)."
    )
    st.stop()

st.subheader("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Amount ($)", min_value=0.0, value=500.0, step=50.0)
    hour_of_day = st.slider("Hour of day (0-23)", 0, 23, 14)
    merchant_category = st.selectbox(
        "Merchant category",
        list(label_encoder.classes_),
    )
with col2:
    location_is_new = st.checkbox("Transaction from a new/unfamiliar location")
    device_is_new = st.checkbox("Transaction from a new device")
    transactions_last_hour = st.slider("Transactions in the last hour", 0, 10, 1)

if st.button("Analyze Transaction", type="primary", use_container_width=True):
    merchant_encoded = label_encoder.transform([merchant_category])[0]

    input_df = pd.DataFrame([{
        "amount": amount,
        "hour_of_day": hour_of_day,
        "merchant_category_encoded": merchant_encoded,
        "location_is_new": int(location_is_new),
        "device_is_new": int(device_is_new),
        "transactions_last_hour": transactions_last_hour,
    }])[feature_cols]

    fraud_probability = clf.predict_proba(input_df)[:, 1][0]
    anomaly_flag = int(iso_forest.predict(input_df)[0] == -1)
    risk_level, risk_icon = assign_risk_level(fraud_probability)
    reasons = explain_transaction(
        amount, location_is_new, device_is_new, hour_of_day,
        transactions_last_hour, anomaly_flag
    )

    st.divider()
    st.subheader("Result")

    m1, m2 = st.columns(2)
    m1.metric("Fraud Probability", f"{fraud_probability:.1%}")
    m2.metric("Risk Level", f"{risk_icon} {risk_level}")

    st.progress(min(float(fraud_probability), 1.0))

    st.write("**Explanation:**")
    for r in reasons:
        st.write(f"- {r}")

st.divider()
st.caption(
    "Built for the HackForge FinTech Codeathon · Model: Random Forest Classifier "
    "+ Isolation Forest, trained on 10,000 transactions."
)
