"""
test_pipeline.py
================
Automated test suite verifying the integrity of the data generation,
feature extraction, Isolation Forest model, explainability, and triage queue.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from log_generator import generate_enterprise_logs
from feature_engineering import compute_user_behavior_features, FEATURE_COLUMNS
from model import IsolationForestAnomalyDetector
from investigation import InsiderRiskQueueManager, generate_investigation_report


def test_log_generator():
    """Verifies synthetic log stream structure and distributions."""
    raw_logs, users_meta = generate_enterprise_logs(num_users=20, days=3, threat_ratio=0.1)
    
    assert len(raw_logs) > 0, "Raw logs should not be empty"
    assert len(users_meta) == 20, "Users metadata should have 20 records"
    assert "timestamp" in raw_logs.columns
    assert "action_type" in raw_logs.columns
    assert "user_id" in raw_logs.columns
    assert "bytes_transferred" in raw_logs.columns
    assert users_meta["is_threat"].sum() > 0, "Should have injected threat accounts"


def test_feature_engineering():
    """Verifies behavioral feature transformation."""
    raw_logs, _ = generate_enterprise_logs(num_users=15, days=3, threat_ratio=0.1)
    feats = compute_user_behavior_features(raw_logs)
    
    assert len(feats) == 15, "Should have 1 row per user"
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"Missing feature column: {col}"
        assert not feats[col].isnull().any(), f"NaNs found in feature: {col}"


def test_isolation_forest_model():
    """Verifies model training, risk scoring, and explainability attribution."""
    raw_logs, _ = generate_enterprise_logs(num_users=30, days=4, threat_ratio=0.1)
    feats = compute_user_behavior_features(raw_logs)
    
    detector = IsolationForestAnomalyDetector(n_estimators=50, contamination=0.1, random_state=42)
    detector.fit(feats, feature_names=FEATURE_COLUMNS)
    assert detector.is_fitted
    
    preds = detector.predict_risk(feats)
    assert len(preds) == 30
    assert "risk_score" in preds.columns
    assert "risk_tier" in preds.columns
    assert preds["risk_score"].min() >= 0.0
    assert preds["risk_score"].max() <= 100.0
    
    # Check explainability
    explanation = detector.explain_user(feats.iloc[0])
    assert "top_contributions" in explanation
    assert "analyst_narrative" in explanation
    assert len(explanation["top_contributions"]) > 0


def test_queue_manager_persistence(tmp_path):
    """Verifies triage queue state saving and retrieval."""
    test_file = str(tmp_path / "test_queue.json")
    mgr = InsiderRiskQueueManager(test_file)
    
    mgr.update_user_status("USR_0001", "UNDER_INVESTIGATION", "Suspicious late night activity", "Analyst_A")
    status = mgr.get_user_status("USR_0001")
    
    assert status["status"] == "UNDER_INVESTIGATION"
    assert status["analyst_notes"] == "Suspicious late night activity"
    assert status["assigned_analyst"] == "Analyst_A"
    
    # Reload from disk
    mgr2 = InsiderRiskQueueManager(test_file)
    status2 = mgr2.get_user_status("USR_0001")
    assert status2["status"] == "UNDER_INVESTIGATION"


if __name__ == "__main__":
    print("Running tests manually...")
    test_log_generator()
    print("[PASS] test_log_generator passed.")
    test_feature_engineering()
    print("[PASS] test_feature_engineering passed.")
    test_isolation_forest_model()
    print("[PASS] test_isolation_forest_model passed.")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        test_queue_manager_persistence(Path(td))
    print("[PASS] test_queue_manager_persistence passed.")
    print("All tests passed successfully!")
