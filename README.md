# 🛡️ NetShield AI — Explainable Security
NetShields AI is a real-time anomaly detection platform built to serve a variety of industries including cybersecurity, IoT, e-commerce, and more. Leveraging traditional ML with explainable AI and GenAI-based explanations, it transforms black-box predictions into human-friendly insights.

## 🚀 Live Demo
👉 [Try NetShields AI on Streamlit Cloud](https://netshields-ai.streamlit.app) *(Link placeholder — deploy on Streamlit Cloud and attach)*

---

## 🧠 Project Summary

| Aspect              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Use Case**        | Universal anomaly detection with human-readable explanations                |
| **ML Approach**     | Random Forest Classifier with SMOTE balancing and threshold tuning          |
| **Explainability**  | SHAP (feature impact) + LangChain GenAI natural language explanations        |
| **Deployment**      | Streamlit UI with CSV export, real-time scoring, and override logic         |
| **Adaptability**    | Built for plug-and-play in any domain: just modify feature input!           |

---

## 💡 Key Features

- 🔍 Real-time anomaly detection with configurable thresholds
- 📊 SHAP-based feature importance explanations
- 🤖 GenAI explanation engine using Llama3 + LangChain
- 📈 Risk scoring (0–100) with industry-specific heuristics
- 💬 Override engine for rule-based edge cases
- 📤 CSV download of predictions for audits
- 🧠 Interactive Streamlit UI for demo and validation

---

## 🏭 Applicability Across Domains

| Domain         | Examples of Use Case                              |
|----------------|---------------------------------------------------|
| Cybersecurity  | Login attacks, credential stuffing, odd access    |
| E-commerce     | Suspicious checkout, click fraud                  |
| IoT/Devices    | Unusual sensor spikes or device failures          |
| Healthcare     | Abnormal vital pattern alerts                     |
| Operations     | Unusual system load or task processing delays     |

---

## 🔢 ML Models Tried

| Model               | Status     | Outcome Summary                   |
|---------------------|------------|-----------------------------------|
| Isolation Forest    | ✅ Tried    | Poor precision, unsupervised base |
| XGBoost             | ✅ Tried    | High recall, poor F1, unstable    |
| Random Forest (final)| ✅ Best   | Tuned + stable + interpretable    |
| Logistic Regression | ✅ Baseline | Low performance                   |
| DBSCAN              | ✅ Failed   | Resource-heavy                    |

---

## 📊 Business Impact

| Metric               | Traditional    | NetShields AI        |
|----------------------|----------------|-----------------------|
| False Positives      | 40%+           | 12–18% ✅             |
| Review Time          | Manual hours   | < 3 sec ⚡            |
| Explanation Clarity  | None           | GenAI + SHAP 💡      |
| Risk Visibility      | Hidden         | Risk Score 0–100      |
| Deployment Time      | Weeks          | Plug-n-play template  |

---

## 🧰 Tech Stack

- **PySpark**: Initial data cleaning and aggregation
- **Random Forest**: Best performing supervised model
- **SMOTE**: To handle extreme class imbalance
- **SHAP**: For explainable feature impact
- **LangChain + Llama3**: Natural language explanation
- **Streamlit**: User-facing dashboard
- **Joblib + dotenv**: Model persistence + secure config

---

## 🛡️ Architecture

```
    A[Login Activity] --> B[Feature Engineering]
    B --> C[Random Forest Model]
    C --> D[SHAP Explainer]
    D --> E[GenAI (LangChain)]
    E --> F[Risk Scoring (0–100)]
    F --> G[Streamlit App]
    G --> H[CSV Export]
    H --> I[Analyst Review or Feedback]
```

## 📁 Export Formats

- ✅ One-click CSV export (single or multiple predictions)
- 📈 Risk scores + GenAI explanations included
- 💬 Useful for audit logs or business handoffs

---

## 🧾 Example Output

```json
{
  "user_id": "user_001",
  "total_logins": 100,
  "failed_logins": 95,
  "hour": 2,
  "risk_score": 91,
  "prediction": "High Risk",
  "explanation": "High failed login ratio, off-hour timing..."
}
```

---

## 📦 Getting Started

```bash
git clone https://github.com/shagunkhurana/netshields-ai
cd netshields-ai
pip install -r requirements.txt
streamlit run app.py
```

---


### 🌟 Make anomalies explainable. Make intelligence actionable.
