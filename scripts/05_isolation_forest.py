import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

df=pd.read_parquet("outputs/user_login_flags.parquet")
df=df.dropna(subset=["total_logins","failed_logins"])

X = df[["total_logins", "failed_logins"]]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_scaled)

df["anomaly_score"]=model.decision_function(X_scaled)
df["is_anomaly"]=model.predict(X_scaled)
df["is_anomaly"]=df["is_anomaly"].map({-1:1,1:0})
df["is_account_takeover"] = df["is_account_takeover"].astype(int)

y_true=df["is_account_takeover"]
y_pred=df["is_anomaly"]
print(df["is_account_takeover"].value_counts())

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f" Precision: {precision:.3f}")
print(f" Recall:    {recall:.3f}")
print(f" F1 Score:  {f1:.3f}")

df.to_parquet("outputs/if_results.parquet", index=False)
import joblib
joblib.dump(model, "isolation_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")