"""
feature_engineering.py
======================
Extracts behavioral, volumetric, temporal, and peer-deviation features from
raw enterprise activity logs to feed the Isolation Forest anomaly detector.
"""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd


RISKY_CLOUD_DOMAINS = ["mega.nz", "dropbox.com", "wetransfer.com", "pastebin.com"]


def compute_user_behavior_features(raw_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw SIEM event streams into comprehensive user-level behavioral feature vectors.
    
    Parameters:
        raw_logs_df: DataFrame of raw log events.
        
    Returns:
        DataFrame containing 1 row per user with rich statistical features.
    """
    df = raw_logs_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    df["is_off_hours_calc"] = (df["hour"] < 8) | (df["hour"] >= 18) | df["is_weekend"]
    
    # Bytes in Megabytes
    df["bytes_mb"] = df["bytes_transferred"] / (1024 * 1024)
    
    user_features: List[Dict[str, Any]] = []
    
    grouped = df.groupby("user_id")
    for uid, group in grouped:
        dept = group["department"].iloc[0]
        total_events = len(group)
        
        # Authentication metrics
        logon_events = group[group["action_type"] == "LOGON"]
        login_count = len(logon_events)
        failed_logins = len(group[(group["action_type"] == "LOGON") & (group["status"] == "FAILURE")])
        failed_login_ratio = failed_logins / max(1, login_count)
        
        # Temporal metrics
        off_hours_events = len(group[group["is_off_hours_calc"]])
        off_hours_ratio = off_hours_events / max(1, total_events)
        weekend_events = len(group[group["is_weekend"]])
        
        # Removable media & device usage
        usb_connects = len(group[group["action_type"] == "DEVICE_CONNECT"])
        distinct_devices = group["device_id"].nunique()
        
        # Sensitive file operations
        file_events = group[group["action_type"].isin(["FILE_ACCESS", "FILE_COPY"])]
        sensitive_files = group[group["sensitivity"].isin(["CONFIDENTIAL", "RESTRICTED"])]
        sensitive_file_count = len(sensitive_files)
        sensitive_ratio = sensitive_file_count / max(1, len(file_events))
        file_copy_count = len(group[group["action_type"] == "FILE_COPY"])
        distinct_sensitive_paths = sensitive_files["resource"].nunique()
        
        # Data Volume & Exfiltration channels
        total_mb_xfer = group["bytes_mb"].sum()
        max_single_xfer_mb = group["bytes_mb"].max() if total_events > 0 else 0.0
        avg_xfer_per_event_mb = group["bytes_mb"].mean() if total_events > 0 else 0.0
        
        # Risky Cloud uploads
        cloud_uploads = group[
            (group["action_type"] == "HTTP_REQUEST") & 
            (group["resource"].apply(lambda r: any(d in str(r).lower() for d in RISKY_CLOUD_DOMAINS)))
        ]
        risky_cloud_count = len(cloud_uploads)
        risky_cloud_bytes_mb = cloud_uploads["bytes_mb"].sum()
        
        # External Email
        ext_emails = group[
            (group["action_type"] == "EMAIL_SENT") & 
            (group["resource"].str.contains("@", na=False)) & 
            (~group["resource"].str.endswith("@enterprise.corp", na=False))
        ]
        ext_email_count = len(ext_emails)
        ext_email_bytes_mb = ext_emails["bytes_mb"].sum()
        
        # Action velocity: max events within any single hour
        hourly_counts = group.set_index("timestamp").resample("1h")["action_type"].count()
        max_events_per_hour = hourly_counts.max() if len(hourly_counts) > 0 else total_events
        
        user_features.append({
            "user_id": uid,
            "department": dept,
            "total_events": total_events,
            "login_count": login_count,
            "failed_logins": failed_logins,
            "failed_login_ratio": round(failed_login_ratio, 4),
            "off_hours_events": off_hours_events,
            "off_hours_ratio": round(off_hours_ratio, 4),
            "weekend_events": weekend_events,
            "usb_connects": usb_connects,
            "distinct_devices": distinct_devices,
            "sensitive_file_count": sensitive_file_count,
            "sensitive_ratio": round(sensitive_ratio, 4),
            "file_copy_count": file_copy_count,
            "distinct_sensitive_paths": distinct_sensitive_paths,
            "total_mb_xfer": round(total_mb_xfer, 2),
            "max_single_xfer_mb": round(max_single_xfer_mb, 2),
            "avg_xfer_per_event_mb": round(avg_xfer_per_event_mb, 3),
            "risky_cloud_count": risky_cloud_count,
            "risky_cloud_bytes_mb": round(risky_cloud_bytes_mb, 2),
            "ext_email_count": ext_email_count,
            "ext_email_bytes_mb": round(ext_email_bytes_mb, 2),
            "max_events_per_hour": int(max_events_per_hour)
        })
        
    feat_df = pd.DataFrame(user_features)
    
    # Departmental baseline peer deviation z-score
    dept_means = feat_df.groupby("department")["total_mb_xfer"].transform("mean")
    dept_stds = feat_df.groupby("department")["total_mb_xfer"].transform("std").replace(0, 1)
    feat_df["peer_vol_zscore"] = ((feat_df["total_mb_xfer"] - dept_means) / dept_stds).round(3)
    
    return feat_df


def extract_timeline_features(raw_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily time-series feature snapshots per user for trend visualization
    in the Security Analyst Queue.
    """
    df = raw_logs_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["is_off_hours"] = (df["hour"] < 8) | (df["hour"] >= 18) | (df["timestamp"].dt.weekday >= 5)
    df["bytes_mb"] = df["bytes_transferred"] / (1024 * 1024)
    
    daily = df.groupby(["user_id", "date"]).agg(
        total_actions=("action_type", "count"),
        failed_logins=("status", lambda s: (s == "FAILURE").sum()),
        off_hours_actions=("is_off_hours", "sum"),
        bytes_transferred_mb=("bytes_mb", "sum"),
        sensitive_accesses=("sensitivity", lambda s: s.isin(["CONFIDENTIAL", "RESTRICTED"]).sum()),
        usb_connections=("action_type", lambda a: (a == "DEVICE_CONNECT").sum())
    ).reset_index()
    
    return daily


FEATURE_COLUMNS = [
    "failed_logins",
    "failed_login_ratio",
    "off_hours_events",
    "off_hours_ratio",
    "weekend_events",
    "usb_connects",
    "distinct_devices",
    "sensitive_file_count",
    "sensitive_ratio",
    "file_copy_count",
    "distinct_sensitive_paths",
    "total_mb_xfer",
    "max_single_xfer_mb",
    "avg_xfer_per_event_mb",
    "risky_cloud_count",
    "risky_cloud_bytes_mb",
    "ext_email_count",
    "ext_email_bytes_mb",
    "max_events_per_hour",
    "peer_vol_zscore"
]


if __name__ == "__main__":
    data_dir = r"C:\Users\DELL\.gemini\antigravity\scratch\insider_threat_detector\data"
    raw_path = os.path.join(data_dir, "raw_user_activity_logs.csv")
    if os.path.exists(raw_path):
        logs = pd.read_csv(raw_path)
        feats = compute_user_behavior_features(logs)
        out_path = os.path.join(data_dir, "processed_user_features.csv")
        feats.to_csv(out_path, index=False)
        print(f"Extracted {feats.shape[1]} features for {feats.shape[0]} users.")
        print(f"Saved to: {out_path}")
    else:
        print(f"Raw logs not found at {raw_path}")
