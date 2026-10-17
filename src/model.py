"""
model.py
========
Isolation Forest Anomaly Detector with SOC-Calibrated Risk Scoring (0-100),
Dynamic Thresholding, Local Explainability, and Model Persistence.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib


class IsolationForestAnomalyDetector:
    """
    Production-grade Isolation Forest Anomaly Detector tailored for
    enterprise insider threat detection from user activity logs.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_samples: str = "auto",
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=1
        )
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False
        self.threshold: float = 0.0
        
    def fit(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> "IsolationForestAnomalyDetector":
        """
        Fits the RobustScaler and Isolation Forest model on behavioral feature vectors.
        Computes baseline normal statistics for explainability attribution.
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
            
        X_df = X[self.feature_names] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        
        # Compute baseline reference distribution (mean, std, median, IQR) for explainability
        self.baseline_stats = {}
        for col in self.feature_names:
            series = X_df[col].astype(float)
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            self.baseline_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std() if series.std() > 0 else 1.0),
                "median": float(series.median()),
                "iqr": float(iqr if iqr > 0 else 1.0),
                "max": float(series.max())
            }
            
        # Fit scaler and Isolation Forest
        X_scaled = self.scaler.fit_transform(X_df)
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        # Compute raw decision function scores to calibrate risk threshold
        raw_scores = self.model.decision_function(X_scaled)
        self.threshold = float(np.percentile(raw_scores, self.contamination * 100))
        
        return self

    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates feature vectors and computes:
        - raw_decision_score
        - is_anomaly (True/False)
        - risk_score (calibrated 0 - 100 scale)
        - risk_tier (CRITICAL, HIGH, MEDIUM, LOW)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
            
        X_df = X[self.feature_names] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        X_scaled = self.scaler.transform(X_df)
        
        raw_scores = self.model.decision_function(X_scaled)
        binary_preds = self.model.predict(X_scaled) # -1: anomaly, 1: inlier
        
        # In scikit-learn IsolationForest, lower decision_function scores indicate higher abnormality.
        # Typically scores range between approx -0.3 (extreme anomaly) and +0.3 (very normal).
        # We map raw scores to a security risk scale from 0 to 100:
        # Lower raw score -> Higher risk score.
        # We use a calibrated sigmoid mapping centered around the threshold:
        norm_scores = (self.threshold - raw_scores) / (abs(self.threshold) + 0.15)
        risk_scores = 1.0 / (1.0 + np.exp(-4.5 * norm_scores))
        risk_scores = np.clip(np.round(risk_scores * 100, 1), 0.0, 100.0)
        
        risk_tiers = []
        for r, is_anom in zip(risk_scores, binary_preds):
            if r >= 80 or (is_anom == -1 and r >= 75):
                risk_tiers.append("CRITICAL")
            elif r >= 60:
                risk_tiers.append("HIGH")
            elif r >= 40:
                risk_tiers.append("MEDIUM")
            else:
                risk_tiers.append("LOW")
                
        results_df = pd.DataFrame({
            "raw_anomaly_score": np.round(raw_scores, 4),
            "is_anomaly": binary_preds == -1,
            "risk_score": risk_scores,
            "risk_tier": risk_tiers
        })
        
        return results_df

    def explain_user(self, user_series: pd.Series, top_k: int = 5) -> Dict[str, Any]:
        """
        Explainability module: breaks down which specific behavioral features
        contributed most to flagging the employee, generating human-readable narrative.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
            
        contributions = []
        for feat in self.feature_names:
            if feat not in user_series:
                continue
            val = float(user_series[feat])
            base = self.baseline_stats.get(feat, {"mean": 0.0, "std": 1.0, "median": 0.0, "iqr": 1.0})
            
            # Compute standardized robust deviation
            robust_z = (val - base["median"]) / max(base["iqr"], 0.01)
            standard_z = (val - base["mean"]) / max(base["std"], 0.01)
            
            # Ratio to baseline
            multiplier = val / max(base["mean"], 0.01) if base["mean"] > 0 else (val if val > 0 else 1.0)
            
            contributions.append({
                "feature": feat,
                "user_value": val,
                "baseline_mean": base["mean"],
                "baseline_median": base["median"],
                "robust_zscore": round(robust_z, 2),
                "multiplier": round(multiplier, 2),
                "importance_score": max(0.0, robust_z)
            })
            
        # Sort by highest positive anomaly deviation
        contributions.sort(key=lambda x: x["importance_score"], reverse=True)
        top_factors = contributions[:top_k]
        
        # Generate automated analyst narrative
        narrative_parts = []
        for factor in top_factors:
            f_name = factor["feature"]
            val = factor["user_value"]
            mult = factor["multiplier"]
            
            if "cloud" in f_name and val > 0:
                narrative_parts.append(f"Exfiltrated {val:.1f} MB via unauthorized cloud storage endpoints")
            elif "usb" in f_name and val > 0:
                narrative_parts.append(f"Connected unauthorized external USB media {int(val)} times")
            elif "off_hours" in f_name and mult > 2.0:
                narrative_parts.append(f"{mult:.1f}x higher off-hours activity ({int(val)} events outside normal 08:00-18:00 shift)")
            elif "failed_login" in f_name and val > 0:
                narrative_parts.append(f"Recorded {int(val)} failed login attempts indicating credential probing / brute-force")
            elif "sensitive" in f_name and val > 0:
                narrative_parts.append(f"Accessed {int(val)} restricted/confidential corporate assets")
            elif "total_mb" in f_name and mult > 3.0:
                narrative_parts.append(f"Total data volume {val:.1f} MB exceeds peer baseline by {mult:.1f}x")
            elif "devices" in f_name and val > 2:
                narrative_parts.append(f"Accessed {int(val)} distinct machines (lateral movement warning)")
            elif factor["importance_score"] > 2.0:
                narrative_parts.append(f"Abnormal pattern on '{f_name}': {val:.2f} (peer baseline: {factor['baseline_mean']:.2f})")
                
        if not narrative_parts:
            narrative = "Activity patterns are consistent with normal departmental baseline."
        else:
            narrative = "Critical Risk Indicators: " + "; ".join(narrative_parts) + "."
            
        return {
            "top_contributions": top_factors,
            "all_contributions": contributions,
            "analyst_narrative": narrative
        }

    def save(self, filepath: str) -> None:
        """Persists trained model and metadata."""
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "baseline_stats": self.baseline_stats,
            "threshold": self.threshold,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "IsolationForestAnomalyDetector":
        """Loads serialized model and reconstructs detector state."""
        payload = joblib.load(filepath)
        detector = cls(
            n_estimators=payload["n_estimators"],
            contamination=payload["contamination"]
        )
        detector.model = payload["model"]
        detector.scaler = payload["scaler"]
        detector.feature_names = payload["feature_names"]
        detector.baseline_stats = payload["baseline_stats"]
        detector.threshold = payload["threshold"]
        detector.is_fitted = True
        return detector


if __name__ == "__main__":
    from feature_engineering import FEATURE_COLUMNS
    
    data_dir = r"C:\Users\DELL\.gemini\antigravity\scratch\insider_threat_detector\data"
    feats_path = os.path.join(data_dir, "processed_user_features.csv")
    
    if os.path.exists(feats_path):
        df = pd.read_csv(feats_path)
        detector = IsolationForestAnomalyDetector(n_estimators=150, contamination=0.06)
        detector.fit(df, feature_names=FEATURE_COLUMNS)
        
        preds = detector.predict_risk(df)
        df_scored = pd.concat([df, preds], axis=1)
        
        print(f"Model trained successfully on {len(df)} user profiles.")
        print("Risk Tier Distribution:")
        print(preds["risk_tier"].value_counts())
        
        model_save_path = r"C:\Users\DELL\.gemini\antigravity\scratch\insider_threat_detector\models\isolation_forest_model.joblib"
        detector.save(model_save_path)
        print(f"Saved model to: {model_save_path}")
    else:
        print("Features file not found. Run feature_engineering.py first.")
