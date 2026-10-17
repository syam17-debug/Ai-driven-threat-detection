"""
run_pipeline.py
===============
End-to-end orchestration CLI for the Insider Threat Anomaly Detection Pipeline:
1. Enterprise Log Generation (SIEM / CERT format)
2. Behavioral Feature Engineering & Aggregation
3. Isolation Forest Model Training & Calibration
4. Multi-Detector Benchmarking (IF vs One-Class SVM vs LOF vs Z-Score)
5. Model Persistence & Benchmark Reporting
"""

import os
import sys
import json
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

sys.path.append(SRC_DIR)

from log_generator import generate_enterprise_logs
from feature_engineering import compute_user_behavior_features, FEATURE_COLUMNS
from model import IsolationForestAnomalyDetector
from evaluator import run_comprehensive_benchmark


def main():
    print("=" * 70)
    print("   INSIDER THREAT ANOMALY DETECTOR - FULL PIPELINE EXECUTION")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Generate realistic SIEM logs
    print("\n[Step 1/5] Generating enterprise activity & access logs...")
    raw_logs_path = os.path.join(DATA_DIR, "raw_user_activity_logs.csv")
    meta_path = os.path.join(DATA_DIR, "users_metadata.csv")
    
    raw_logs, users_meta = generate_enterprise_logs(num_users=150, days=14, threat_ratio=0.06)
    raw_logs.to_csv(raw_logs_path, index=False)
    users_meta.to_csv(meta_path, index=False)
    print(f" -> Generated {len(raw_logs):,} SIEM events across {len(users_meta)} users.")
    print(f" -> Threat users: {users_meta['is_threat'].sum()} ({users_meta['is_threat'].mean()*100:.1f}%)")

    # 2. Extract behavioral features
    print("\n[Step 2/5] Performing behavioral feature engineering...")
    features_df = compute_user_behavior_features(raw_logs)
    feats_path = os.path.join(DATA_DIR, "processed_user_features.csv")
    features_df.to_csv(feats_path, index=False)
    print(f" -> Extracted {features_df.shape[1]} features for {features_df.shape[0]} user accounts.")

    # 3. Train Isolation Forest
    print("\n[Step 3/5] Training Isolation Forest Anomaly Detector...")
    detector = IsolationForestAnomalyDetector(n_estimators=180, contamination=0.06, random_state=42)
    detector.fit(features_df, feature_names=FEATURE_COLUMNS)
    
    model_path = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
    detector.save(model_path)
    print(f" -> Model trained and saved to: {model_path}")
    print(f" -> Dynamic anomaly threshold calibrated at score: {detector.threshold:.4f}")

    # 4. Predict risk and score users
    print("\n[Step 4/5] Scoring all monitored accounts & ranking insider risk...")
    preds = detector.predict_risk(features_df)
    scored_df = pd.concat([features_df, preds], axis=1)
    
    print("\nRisk Tier Breakdown:")
    for tier, count in preds["risk_tier"].value_counts().items():
        print(f"   * {tier:<10}: {count:>3} users")

    # 5. Run comparative benchmark against baselines
    print("\n[Step 5/5] Running multi-algorithm benchmark against baselines...")
    bench_results, curves = run_comprehensive_benchmark(features_df, users_meta, contamination=0.06)
    
    bench_path = os.path.join(DATA_DIR, "benchmark_metrics.csv")
    bench_results.to_csv(bench_path, index=False)
    
    print("\n" + "=" * 70)
    print("   MULTI-DETECTOR CYBERSECURITY BENCHMARK")
    print("=" * 70)
    print(bench_results[["model", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "false_positive_rate", "inference_latency_ms"]].to_string(index=False))
    print("=" * 70)
    print("\n[SUCCESS] Pipeline completed successfully!")
    print("To launch the Security Analyst Risk Queue Dashboard, run:")
    print("  streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
