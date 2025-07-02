# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(
    page_title="SunFi Default Risk Predictor",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load Assets ---
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# --- Header Section ---
st.markdown(
    """
    <h1 style="text-align: center; color: #1F4E79;">🔍 SunFi Customer Default Risk Predictor</h1>
    <p style="text-align: center;">Predict if a customer is likely to default on asset financing using machine learning.</p>
    """,
    unsafe_allow_html=True,
)

# --- User Input Form ---
with st.form("input_form"):
    st.subheader("🔧 Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("📅 Age", 18, 100, 40)
        credit_score = st.number_input("💳 Credit Score", 30000, 85000, 60000)
        asset_amount = st.number_input("💼 Asset Amount (₦)", 0, 10000000, 3000000)

    with col2:
        monthly_income = st.number_input("💰 Monthly Income (₦)", 10000, 2000000, 300000)
        term_months = st.selectbox("🕐 Term (Months)", [6, 12, 18, 24, 36, 48])
        gender = st.selectbox("🧍 Gender", ["Male", "Female"])
        payment_history = st.selectbox("📊 Payment History", ["Always on time", "Frequently late"])

    submitted = st.form_submit_button("📈 Predict Risk")

# --- Prediction Logic ---
if submitted:
    gender_encoded = 1 if gender == "Male" else 0
    history_encoded = 0 if payment_history == "Always on time" else 1
    debt_to_income = asset_amount / (monthly_income + 1e-6)

    input_df = pd.DataFrame([{
        "Age": age,
        "Credit_Score": credit_score,
        "Asset_Amount_NGN": asset_amount,
        "Term_Months": term_months,
        "Monthly_Income_NGN": monthly_income,
        "Gender": gender_encoded,
        "Payment_History": history_encoded,
        "Debt_to_Income": debt_to_income
    }])

    # Ensure proper feature order
    input_df = input_df[feature_names]
    X_scaled = scaler.transform(input_df)

    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    # --- Results Display ---
    st.markdown("## 🧾 Prediction Result")
    colA, colB = st.columns([1, 2])
    with colA:
        if prediction == 1:
            st.error("🚨 **Customer likely to Default**")
        else:
            st.success("✅ **Customer unlikely to Default**")

    with colB:
        st.metric(label="📉 Probability of Default", value=f"{prob:.2%}")

    with st.expander("🔍 View Input Summary"):
        st.table(input_df)

# Optional: Add footer
st.markdown(
    """
    <hr style="margin-top: 40px;">
    <small style="text-align: center; display: block;">
        Built with ❤️ by SunFi Data Team | Powered by Streamlit & XGBoost
    </small>
    """,
    unsafe_allow_html=True,
)
