import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import IsolationForestAnomalyDetector
from investigation import query_ai_analyst

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

feats = pd.read_csv(os.path.join(data_dir, "processed_user_features.csv"))
detector = IsolationForestAnomalyDetector.load(os.path.join(models_dir, "isolation_forest_model.joblib"))

preds = detector.predict_risk(feats)
combined = pd.concat([feats, preds], axis=1).sort_values(by="risk_score", ascending=False)
top_user = combined.iloc[0]
user_id = top_user["user_id"]
explanation = detector.explain_user(top_user)

raw_logs = pd.read_csv(os.path.join(data_dir, "raw_user_activity_logs.csv"))
user_raw = raw_logs[raw_logs["user_id"] == user_id]

# Read from environment or .env
env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key and os.path.exists(env_file):
    with open(env_file, "r") as f:
        for line in f:
            if line.startswith("GROQ_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip("'\"")

print("=" * 70)
print(f"Testing Groq AI Cyber Analyst for flagged user: {user_id}")
print(f"Calibrated Isolation Forest Risk Score: {top_user['risk_score']}/100 ({top_user['risk_tier']})")
print(f"Raw Tree Anomaly Score: {top_user['raw_anomaly_score']}")
print("=" * 70)

response = query_ai_analyst(user_id, top_user, explanation, user_raw, provider="groq", api_key=api_key)

print("\nAI FORENSIC THREAT BRIEFING OUTPUT:")
with open("groq_output.txt", "w", encoding="utf-8") as f:
    f.write(response)

print(response.encode("ascii", "replace").decode("ascii"))
print("=" * 70)
