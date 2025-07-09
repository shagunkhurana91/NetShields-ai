import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
model = joblib.load("rf_fraud_model.pkl")              
scaler = joblib.load("rf_scaler.pkl")                  

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
      

explainer = shap.Explainer(model)

# LangChain Setup
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
template = """
You are an AI fraud detection assistant.

Based on the SHAP analysis:
{shap_summary}

Please explain why this login attempt might be suspicious in simple terms in short simple manner.
"""
prompt = PromptTemplate(input_variables=["shap_summary"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)


st.set_page_config(page_title="🛡️ Fraud Detection + GenAI", layout="centered")
st.title("🔐 Fintech Fraud Detection App")
st.markdown("Real-time prediction + SHAP + GenAI Explanation 🚀")

# Session Log
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

with st.form("fraud_form"):
    st.subheader("Enter login details:")
    total_logins = st.number_input("Total Logins", min_value=1, value=10)
    failed_logins = st.number_input("Failed Logins", min_value=0, value=5)
    hour = st.slider("Login Hour (0–23)", 0, 23, value=2)
    user_id = st.text_input("User ID (optional)", "user_001")
    submitted = st.form_submit_button("Detect Fraud")

if submitted:
    # Feature Engineering
    failed_login_ratio = failed_logins / (total_logins + 1)
    success_rate = (total_logins - failed_logins) / (total_logins + 1)
    is_off_hours = 1 if hour < 6 or hour > 22 else 0
    logins_per_hour = total_logins / (hour + 1)
    login_intensity = total_logins * failed_logins
    is_night_time = is_off_hours

    sample = pd.DataFrame([[total_logins, failed_logins, hour,
                            failed_login_ratio, success_rate, is_off_hours,
                            logins_per_hour, login_intensity, is_night_time]],
                          columns=["total_logins", "failed_logins", "hour",
                                   "failed_login_ratio", "success_rate", "is_off_hours",
                                   "logins_per_hour", "login_intensity", "is_night_time"])

    scaled = scaler.transform(sample)
    prob = model.predict_proba(scaled)[0][1]
    threshold = 0.3
    is_fraud = int(prob >= threshold)
    risk_score = round(prob * 100)

    #  Heuristic Override
    if failed_login_ratio > 0.8 or failed_logins > 100:
        st.warning("⚠️ Heuristic override triggered due to high failed login ratio")
        risk_score = max(risk_score, 90)
        prob = max(prob, 0.9)
        is_fraud = 1

    # Show Result
    st.markdown("### 🔍 Prediction Result")
    st.write(f"📊 **Risk Score**: `{risk_score} / 100`")

    if risk_score >= 75:
        st.markdown("** High Risk**")
    elif risk_score >= 40:
        st.markdown("** Medium Risk**")
    else:
        st.markdown("** Low Risk**")

    if is_fraud:
        st.error(f"🚨 Fraud Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Looks Safe. (Probability: {prob:.2f})")

    # SHAP Explanation
    shap_vals = explainer(scaled)
    shap_text = []
    for i, feat in enumerate(sample.columns):
        val = shap_vals.values[0][i]
        if isinstance(val, (np.ndarray, list)):
            val = float(val[0]) if len(val) > 0 else 0.0
        else:
            val = float(val)
        if abs(val) > 0.01:
            sign = "increased" if val > 0 else "decreased"
            shap_text.append(f"{feat} {sign} fraud score by {abs(val):.2f}")
    shap_summary = "; ".join(shap_text)

    st.markdown("### 📊 SHAP Explanation")
    st.write(shap_summary)

    # GenAI Explanation
    with st.spinner("💬 Generating GenAI Explanation..."):
        explanation = chain.run({"shap_summary": shap_summary})
    st.markdown("### 🤖 GenAI Explanation")
    st.info(explanation)

    # Log this prediction
    st.session_state.prediction_log.append({
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "total_logins": total_logins,
        "failed_logins": failed_logins,
        "hour": hour,
        "probability": round(prob, 4),
        "risk_score": risk_score,
        "prediction": "Fraud" if is_fraud else "Safe",
        "shap_summary": shap_summary,
        "genai_explanation": explanation
    })

    # Download Button
    st.markdown("---")
    st.subheader("📥 Download Prediction Log")
    if st.session_state.prediction_log:
        df_log = pd.DataFrame(st.session_state.prediction_log)
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "fraud_predictions_log.csv", "text/csv")
    else:
        st.info("No predictions yet to log.")
