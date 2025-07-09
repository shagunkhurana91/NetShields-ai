import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.utils import resample
import joblib

# 📥 Load dataset
df = pd.read_parquet("outputs/user_login_flags.parquet")
df = df.dropna(subset=["total_logins", "failed_logins"])
df["is_account_takeover"] = df["is_account_takeover"].astype(bool)

# 🧠 Feature engineering
df["failed_login_ratio"] = df["failed_logins"] / (df["total_logins"] + 1)
df["success_rate"] = (df["total_logins"] - df["failed_logins"]) / (df["total_logins"] + 1)
df["is_off_hours"] = df["hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)
df["logins_per_hour"] = df["total_logins"] / (df["hour"] + 1)
df["login_intensity"] = df["total_logins"] * df["failed_logins"]
df["is_night_time"] = df["hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# 🧪 Features + Target
features = [
    "total_logins", "failed_logins", "hour", "failed_login_ratio",
    "success_rate", "is_off_hours", "logins_per_hour", "login_intensity", "is_night_time"
]
X = df[features]
y = df["is_account_takeover"]

# 🧹 Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📉 Downsample legit users (3000)
df_legit = df[df["is_account_takeover"] == False]
df_fraud = df[df["is_account_takeover"] == True]
df_legit_down = resample(df_legit, n_samples=3000, random_state=42, replace=False)
df_balanced = pd.concat([df_legit_down, df_fraud]).sample(frac=1, random_state=42)

# Final X, y
X_final = df_balanced[features]
y_final = df_balanced["is_account_takeover"]
X_scaled_final = scaler.transform(X_final)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_final, y_final, test_size=0.2, random_state=42)

# ✅ Train Random Forest with class_weight='balanced'
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# 📊 Evaluation
print(f"✅ Features used: {features}")
print(f"🔍 Threshold: {threshold}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 💾 Save model + scaler
joblib.dump(model, "rf_fraud_model.pkl")
joblib.dump(scaler, "rf_scaler.pkl")
