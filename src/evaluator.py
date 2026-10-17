"""
evaluator.py
============
Model evaluation and benchmarking suite for Insider Threat Detection.
Compares Isolation Forest against One-Class SVM, Local Outlier Factor (LOF),
and Statistical Z-Score baselines with comprehensive cybersecurity metrics.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from model import IsolationForestAnomalyDetector
from feature_engineering import FEATURE_COLUMNS


def evaluate_anomaly_detector(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    y_scores: np.ndarray,
    model_name: str,
    latency_ms: float
) -> Dict[str, Any]:
    """
    Computes standard cybersecurity metrics for anomaly detection.
    """
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Try computing ROC-AUC and PR-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except Exception:
        pr_auc = 0.0
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "model": model_name,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "false_positive_rate": round(float(fpr), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "inference_latency_ms": round(float(latency_ms), 3)
    }


def run_comprehensive_benchmark(
    features_df: pd.DataFrame,
    users_meta_df: pd.DataFrame,
    contamination: float = 0.06
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs multi-model benchmark comparing Isolation Forest with One-Class SVM,
    Local Outlier Factor, and Departmental Z-Score Baseline.
    """
    # Merge ground truth
    merged = pd.merge(features_df, users_meta_df[["user_id", "is_threat", "threat_scenario"]], on="user_id")
    y_true = merged["is_threat"].astype(int).values
    
    X = merged[FEATURE_COLUMNS]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    curves_data = {}
    
    # ---------------- 1. Isolation Forest (Proposed) ----------------
    t0 = time.perf_counter()
    iso = IsolationForestAnomalyDetector(n_estimators=180, contamination=contamination, random_state=42)
    iso.fit(X, feature_names=FEATURE_COLUMNS)
    preds = iso.predict_risk(X)
    t1 = time.perf_counter()
    iso_latency = ((t1 - t0) / len(X)) * 1000
    
    # Inverted decision function so higher = more anomalous
    iso_scores = preds["risk_score"].values
    iso_binary = preds["is_anomaly"].astype(int).values
    iso_eval = evaluate_anomaly_detector(y_true, iso_binary, iso_scores, "Isolation Forest (Proposed)", iso_latency)
    results.append(iso_eval)
    
    fpr_iso, tpr_iso, _ = roc_curve(y_true, iso_scores)
    prec_iso, rec_iso, _ = precision_recall_curve(y_true, iso_scores)
    curves_data["Isolation Forest"] = {
        "fpr": fpr_iso.tolist(), "tpr": tpr_iso.tolist(),
        "precision": prec_iso.tolist(), "recall": rec_iso.tolist()
    }
    
    # ---------------- 2. One-Class SVM ----------------
    t0 = time.perf_counter()
    oc_svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
    oc_svm.fit(X_scaled)
    svm_raw = oc_svm.decision_function(X_scaled)
    t1 = time.perf_counter()
    svm_latency = ((t1 - t0) / len(X)) * 1000
    
    svm_scores = -svm_raw # Invert so higher = more anomalous
    svm_binary = (oc_svm.predict(X_scaled) == -1).astype(int)
    svm_eval = evaluate_anomaly_detector(y_true, svm_binary, svm_scores, "One-Class SVM", svm_latency)
    results.append(svm_eval)
    
    fpr_svm, tpr_svm, _ = roc_curve(y_true, svm_scores)
    prec_svm, rec_svm, _ = precision_recall_curve(y_true, svm_scores)
    curves_data["One-Class SVM"] = {
        "fpr": fpr_svm.tolist(), "tpr": tpr_svm.tolist(),
        "precision": prec_svm.tolist(), "recall": rec_svm.tolist()
    }
    
    # ---------------- 3. Local Outlier Factor (LOF) ----------------
    t0 = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
    lof_binary = (lof.fit_predict(X_scaled) == -1).astype(int)
    t1 = time.perf_counter()
    lof_latency = ((t1 - t0) / len(X)) * 1000
    
    lof_scores = -lof.negative_outlier_factor_
    lof_eval = evaluate_anomaly_detector(y_true, lof_binary, lof_scores, "Local Outlier Factor (LOF)", lof_latency)
    results.append(lof_eval)
    
    fpr_lof, tpr_lof, _ = roc_curve(y_true, lof_scores)
    prec_lof, rec_lof, _ = precision_recall_curve(y_true, lof_scores)
    curves_data["Local Outlier Factor"] = {
        "fpr": fpr_lof.tolist(), "tpr": tpr_lof.tolist(),
        "precision": prec_lof.tolist(), "recall": rec_lof.tolist()
    }
    
    # ---------------- 4. Simple Statistical Z-Score Baseline ----------------
    t0 = time.perf_counter()
    z_scores = merged["peer_vol_zscore"].values
    z_binary = (z_scores > 2.5).astype(int)
    t1 = time.perf_counter()
    z_latency = ((t1 - t0) / len(X)) * 1000
    
    z_eval = evaluate_anomaly_detector(y_true, z_binary, z_scores, "Peer Z-Score Heuristic", z_latency)
    results.append(z_eval)
    
    benchmark_df = pd.DataFrame(results)
    return benchmark_df, curves_data


if __name__ == "__main__":
    import os
    data_dir = r"C:\Users\DELL\.gemini\antigravity\scratch\insider_threat_detector\data"
    feats = pd.read_csv(os.path.join(data_dir, "processed_user_features.csv"))
    users = pd.read_csv(os.path.join(data_dir, "users_metadata.csv"))
    
    bench, curves = run_comprehensive_benchmark(feats, users)
    print("\n================ BENCHMARK RESULTS ================")
    print(bench[["model", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "inference_latency_ms"]].to_string(index=False))
