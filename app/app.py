"""
app.py
======
Security Analyst Insider-Risk Queue & Threat Detection Dashboard.
Built with Streamlit & Plotly. Powered by Isolation Forest Anomaly Detection.
"""

import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to system path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Auto-load .env file if present
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ or not os.environ[k.strip()]:
                        os.environ[k.strip()] = v.strip().strip("'\"")
    except Exception:
        pass

from model import IsolationForestAnomalyDetector
from feature_engineering import compute_user_behavior_features, extract_timeline_features, FEATURE_COLUMNS
from evaluator import run_comprehensive_benchmark
from investigation import InsiderRiskQueueManager, generate_investigation_report, query_ai_analyst, PLAYBOOKS, STATUS_OPTIONS
from log_generator import generate_enterprise_logs


# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="SOC Insider-Risk Queue | Isolation Forest",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Cyber SOC Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 12% 8%, rgba(14, 165, 233, 0.09) 0%, transparent 40%),
                    radial-gradient(circle at 88% 90%, rgba(168, 85, 247, 0.08) 0%, transparent 45%),
                    #080c14;
        color: #f1f5f9;
    }
    
    /* Command Header Banner */
    .soc-header-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(15, 23, 42, 0.65) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    .soc-header-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00f2fe 0%, #38bdf8 40%, #818cf8 80%, transparent 100%);
    }
    .soc-main-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 6px 0;
        background: linear-gradient(135deg, #ffffff 15%, #38bdf8 65%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .soc-sub-title {
        font-size: 0.92rem;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
        line-height: 1.4;
    }
    .soc-pulse-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    .pulsing-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseDot 2s infinite;
    }
    @keyframes pulseDot {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .soc-defense-pill {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 10px 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .pill-label {
        font-size: 0.68rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
    }
    .pill-val {
        font-size: 0.96rem;
        color: #38bdf8;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Modern Glassmorphic KPI Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.65) 100%);
        backdrop-filter: blur(16px);
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid rgba(56, 189, 248, 0.16);
        box-shadow: 0 10px 28px -5px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.07);
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.7), 0 0 20px rgba(56, 189, 248, 0.22);
        border-color: rgba(56, 189, 248, 0.38);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .card-cyan::before { background: linear-gradient(90deg, #00f2fe, #38bdf8); }
    .card-crimson::before { background: linear-gradient(90deg, #ff3366, #ef4444); }
    .card-amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .card-blue::before { background: linear-gradient(90deg, #38bdf8, #60a5fa); }
    .card-purple::before { background: linear-gradient(90deg, #a855f7, #c084fc); }

    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .metric-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
        color: #94a3b8;
    }
    .metric-icon-box {
        font-size: 1.15rem;
        opacity: 0.9;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .text-cyan { color: #00f2fe; text-shadow: 0 0 18px rgba(0, 242, 254, 0.35); }
    .text-crimson { color: #ff3366; text-shadow: 0 0 18px rgba(255, 51, 102, 0.45); }
    .text-amber { color: #f59e0b; text-shadow: 0 0 18px rgba(245, 158, 11, 0.35); }
    .text-blue { color: #38bdf8; text-shadow: 0 0 18px rgba(56, 189, 248, 0.35); }
    .text-purple { color: #c084fc; text-shadow: 0 0 18px rgba(192, 132, 252, 0.35); }
    
    .metric-sub {
        font-size: 0.77rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(14px);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(56, 189, 248, 0.16);
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 9px 20px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease-in-out;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background: rgba(51, 65, 85, 0.45);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.24), rgba(59, 130, 246, 0.28)) !important;
        border: 1px solid #00f2fe !important;
        color: #00f2fe !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
    }

    /* Multiselect Tags - Sleek Cyan/Slate Overhaul (Fixes squished red tags) */
    .stMultiSelect [data-baseweb="tag"] {
        background: rgba(14, 165, 233, 0.15) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        color: #38bdf8 !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 3px 8px !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: #e2e8f0 !important;
    }
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: #38bdf8 !important;
    }

    /* Badges */
    .badge-critical {
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        box-shadow: 0 0 14px rgba(239, 68, 68, 0.35);
    }
    .badge-high {
        background: rgba(249, 115, 22, 0.18);
        border: 1px solid #f97316;
        color: #fdba74;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-medium {
        background: rgba(234, 179, 8, 0.18);
        border: 1px solid #eab308;
        color: #fde047;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-low {
        background: rgba(34, 197, 94, 0.18);
        border: 1px solid #22c55e;
        color: #86efac;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.78rem;
    }

    /* SOC Threat Intelligence Detection Cards */
    .soc-threat-alert-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(153, 27, 27, 0.12) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0 20px 0;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.18);
    }
    .soc-medium-alert-box {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.14) 0%, rgba(161, 98, 7, 0.1) 100%);
        border: 1px solid rgba(234, 179, 8, 0.45);
        border-left: 5px solid #eab308;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0 20px 0;
    }
    .soc-normal-alert-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 95, 70, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0 20px 0;
    }
    .soc-alert-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .soc-alert-body {
        font-size: 0.9rem;
        color: #f1f5f9;
        line-height: 1.55;
    }

    /* High-Tech Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 22px rgba(56, 189, 248, 0.55) !important;
        transform: translateY(-1px) !important;
        border-color: #00f2fe !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* Sidebar Aesthetics */
    [data-testid="stSidebar"] {
        background: #080c14 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
    }
    .sidebar-header-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED DATA & MODEL LOADERS
# ==========================================
@st.cache_resource
def load_or_initialize_pipeline():
    """Initializes or loads datasets, model, and queue manager."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    raw_logs_path = os.path.join(DATA_DIR, "raw_user_activity_logs.csv")
    meta_path = os.path.join(DATA_DIR, "users_metadata.csv")
    feats_path = os.path.join(DATA_DIR, "processed_user_features.csv")
    model_path = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
    queue_path = os.path.join(DATA_DIR, "analyst_queue_state.json")
    
    # Generate data if not present
    if not (os.path.exists(raw_logs_path) and os.path.exists(meta_path)):
        raw_logs_df, meta_df = generate_enterprise_logs(num_users=140, days=14, threat_ratio=0.06)
        raw_logs_df.to_csv(raw_logs_path, index=False)
        meta_df.to_csv(meta_path, index=False)
    else:
        raw_logs_df = pd.read_csv(raw_logs_path)
        meta_df = pd.read_csv(meta_path)
        
    raw_logs_df["timestamp"] = pd.to_datetime(raw_logs_df["timestamp"])
    
    # Feature extraction if not present
    if not os.path.exists(feats_path):
        feats_df = compute_user_behavior_features(raw_logs_df)
        feats_df.to_csv(feats_path, index=False)
    else:
        feats_df = pd.read_csv(feats_path)
        
    # Model training or loading
    if not os.path.exists(model_path):
        detector = IsolationForestAnomalyDetector(n_estimators=180, contamination=0.06)
        detector.fit(feats_df, feature_names=FEATURE_COLUMNS)
        detector.save(model_path)
    else:
        detector = IsolationForestAnomalyDetector.load(model_path)
        
    queue_mgr = InsiderRiskQueueManager(queue_path)
    
    return raw_logs_df, meta_df, feats_df, detector, queue_mgr


# Load pipeline assets
raw_logs, users_meta, features_df, detector, queue_mgr = load_or_initialize_pipeline()

# Generate model predictions for all users
predictions_df = detector.predict_risk(features_df)
scored_users_df = pd.concat([features_df, predictions_df], axis=1)

# Merge metadata
scored_users_df = pd.merge(
    scored_users_df,
    users_meta[["user_id", "primary_device", "threat_scenario", "is_threat"]],
    on="user_id",
    how="left"
)

# Merge Analyst Queue state
triage_statuses = []
analyst_notes_list = []
for uid in scored_users_df["user_id"]:
    q = queue_mgr.get_user_status(uid)
    triage_statuses.append(q["status"])
    analyst_notes_list.append(q["analyst_notes"])

scored_users_df["triage_status"] = triage_statuses
scored_users_df["analyst_notes"] = analyst_notes_list

# Sort by risk score descending
scored_users_df = scored_users_df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)


# ==========================================
# SIDEBAR CONTROLS & SOC FILTERS
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header-box">
        <div style="font-size: 2.2rem; filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.4));">🛡️</div>
        <div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">CYBER COMMAND</div>
            <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 700; letter-spacing: 0.08em;">SOC INTEL // ISOLATION FOREST</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Filter Risk Queue")
    selected_dept = st.multiselect(
        "Department:",
        options=sorted(scored_users_df["department"].unique()),
        default=sorted(scored_users_df["department"].unique())
    )
    
    selected_tiers = st.multiselect(
        "Risk Tier:",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM"]
    )
    
    status_filter = st.multiselect(
        "Triage Status:",
        options=STATUS_OPTIONS,
        default=STATUS_OPTIONS
    )
    
    min_score = st.slider("Minimum Risk Score (0-100):", min_value=0, max_value=100, value=35)
    
    st.markdown("---")
    with st.expander("🤖 AI Threat Copilot (LLM)", expanded=False):
        ai_provider = st.selectbox("LLM Provider:", ["Groq (Llama-3.3-70B)", "Google Gemini (1.5 Flash)"])
        env_key = os.environ.get("GROQ_API_KEY", "") if "Groq" in ai_provider else os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input("API Key (Free):", value=env_key, type="password", help="Enter free Groq or Google Gemini key for AI forensic reasoning.")
        st.caption("Auto-analyzes IoCs with MITRE ATT&CK mapping & threat actor intent.")

    st.markdown("### ⚙️ Engine Parameters")
    st.write(f"**Model:** Isolation Forest")
    st.write(f"**Estimators:** `{detector.n_estimators}` trees")
    st.write(f"**Base Contamination:** `{detector.contamination*100:.1f}%`")
    st.write(f"**Decision Threshold:** `{detector.threshold:.4f}`")
    
    if st.button("🔄 Retrain / Refresh Pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# ==========================================
# TOP TELEMETRY / KPI HEADER
# ==========================================
total_users = len(scored_users_df)
critical_count = (scored_users_df["risk_tier"] == "CRITICAL").sum()
high_count = (scored_users_df["risk_tier"] == "HIGH").sum()
active_triages = (scored_users_df["triage_status"].isin(["NEW_ALERT", "UNDER_INVESTIGATION"])).sum()
avg_risk = scored_users_df["risk_score"].mean()

st.markdown(f"""
<div class="soc-header-banner">
    <div style="max-width: 72%;">
        <div class="soc-pulse-badge">
            <span class="pulsing-dot"></span> LIVE SENSOR STREAM ACTIVE
        </div>
        <h1 class="soc-main-title">Enterprise Insider-Risk Telemetry & Triage Queue</h1>
        <p class="soc-sub-title">Unsupervised User & Entity Behavior Analytics (UEBA) • Active Surveillance: <b style="color: #38bdf8;">{total_users} Accounts</b> across 6 Corporate Divisions</p>
    </div>
    <div>
        <div class="soc-defense-pill">
            <span style="font-size: 1.6rem;">⚡</span>
            <div>
                <div class="pill-label">DETECTION CORE</div>
                <div class="pill-val">Isolation Forest v2.4</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card card-cyan">
        <div class="card-top">
            <span class="metric-title">Monitored Accounts</span>
            <span class="metric-icon-box">👥</span>
        </div>
        <div class="metric-value text-cyan">{total_users}</div>
        <div class="metric-sub"><span style="color:#00f2fe;">●</span> 6 Enterprise Units</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card card-crimson">
        <div class="card-top">
            <span class="metric-title">Critical Anomalies</span>
            <span class="metric-icon-box">🚨</span>
        </div>
        <div class="metric-value text-crimson">{critical_count}</div>
        <div class="metric-sub"><span style="color:#ff3366;">●</span> Immediate Action Required</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card card-amber">
        <div class="card-top">
            <span class="metric-title">High Risk Queue</span>
            <span class="metric-icon-box">⚠️</span>
        </div>
        <div class="metric-value text-amber">{high_count}</div>
        <div class="metric-sub"><span style="color:#f59e0b;">●</span> Significant Deviation</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card card-blue">
        <div class="card-top">
            <span class="metric-title">Pending Investigation</span>
            <span class="metric-icon-box">⏳</span>
        </div>
        <div class="metric-value text-blue">{active_triages}</div>
        <div class="metric-sub"><span style="color:#38bdf8;">●</span> Awaiting Analyst Review</div>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="metric-card card-purple">
        <div class="card-top">
            <span class="metric-title">Mean Enterprise Risk</span>
            <span class="metric-icon-box">🌐</span>
        </div>
        <div class="metric-value text-purple">{avg_risk:.1f}<span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">/100</span></div>
        <div class="metric-sub"><span style="color:#c084fc;">●</span> Baseline Operational</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# MAIN TAB NAVIGATION
# ==========================================
tab_queue, tab_investigation, tab_benchmark, tab_simulator = st.tabs([
    "🚨 Security Analyst Risk Queue",
    "🔍 Deep-Dive User Forensics & XAI",
    "📊 Model Performance & Benchmarks",
    "🧪 Threat Simulator & Log Ingestion"
])


# ==========================================
# TAB 1: SECURITY ANALYST RISK QUEUE
# ==========================================
with tab_queue:
    st.markdown("### 📋 Prioritized Insider-Risk Triage Queue")
    st.markdown(
        "Isolation Forest dynamically surfaces suspicious behavioral deviations before data exfiltration occurs. "
        "Select an employee record to open their comprehensive forensic dossier."
    )
    
    # Filter the queue based on sidebar
    filtered_queue = scored_users_df[
        (scored_users_df["department"].isin(selected_dept)) &
        (scored_users_df["risk_tier"].isin(selected_tiers)) &
        (scored_users_df["triage_status"].isin(status_filter)) &
        (scored_users_df["risk_score"] >= min_score)
    ]
    
    # Display table columns
    display_cols = [
        "user_id", "department", "risk_tier", "risk_score", "triage_status",
        "total_mb_xfer", "off_hours_events", "usb_connects", "sensitive_file_count", "failed_logins"
    ]
    
    # Queue Action Toolbar
    tb_col1, tb_col2, tb_col3 = st.columns([2, 1, 1])
    with tb_col1:
        st.write(f"Showing **{len(filtered_queue)}** prioritized alerts (Sorted by Risk Score)")
    with tb_col2:
        csv_bytes = filtered_queue.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Triage Queue (CSV)",
            data=csv_bytes,
            file_name=f"soc_insider_risk_queue_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with tb_col3:
        if st.button("⚡ Bulk Mark 'Under Investigation'", use_container_width=True):
            for uid in filtered_queue["user_id"]:
                if queue_mgr.get_user_status(uid)["status"] == "NEW_ALERT":
                    queue_mgr.update_user_status(uid, "UNDER_INVESTIGATION", "Bulk triage initiated from SOC queue.")
            st.success("Updated queue status!")
            st.rerun()

    # Formatted dataframe display
    st.dataframe(
        filtered_queue[display_cols].rename(columns={
            "user_id": "User ID",
            "department": "Department",
            "risk_tier": "Risk Tier",
            "risk_score": "Risk Score (0-100)",
            "triage_status": "Triage Status",
            "total_mb_xfer": "Data Volume (MB)",
            "off_hours_events": "Off-Hours Acts",
            "usb_connects": "USB Inserts",
            "sensitive_file_count": "Sensitive Files",
            "failed_logins": "Failed Logins"
        }),
        use_container_width=True,
        column_config={
            "Risk Score (0-100)": st.column_config.ProgressColumn(
                "Risk Score",
                help="Calibrated Isolation Forest Anomaly Risk (0-100)",
                format="%.1f",
                min_value=0,
                max_value=100
            )
        },
        height=380
    )
    
    # Quick user selector to jump to tab 2
    st.markdown("#### ⚡ Quick Investigate Employee")
    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        target_uid = st.selectbox(
            "Select User to Inspect:",
            options=filtered_queue["user_id"].tolist() if len(filtered_queue) > 0 else scored_users_df["user_id"].tolist(),
            index=0
        )
    with q_col2:
        st.write("")
        st.write("")
        if st.button("🔍 Open Investigation Dossier", use_container_width=True):
            st.session_state["selected_investigate_user"] = target_uid
            st.info(f"Loaded dossier for {target_uid}. Switch to the 'Deep-Dive User Forensics & XAI' tab above!")


# ==========================================
# TAB 2: DEEP-DIVE USER FORENSICS & XAI
# ==========================================
with tab_investigation:
    # Get active user from session state or default
    active_user_id = st.session_state.get("selected_investigate_user", scored_users_df.iloc[0]["user_id"])
    
    inv_col1, inv_col2 = st.columns([3, 1])
    with inv_col1:
        user_list = scored_users_df["user_id"].tolist()
        default_idx = user_list.index(active_user_id) if active_user_id in user_list else 0
        investigated_user = st.selectbox("Select Target Employee for Forensic Inspection:", options=user_list, index=default_idx)
    with inv_col2:
        st.write("")
        st.write("")
        # Quick status badge
        curr_user_row = scored_users_df[scored_users_df["user_id"] == investigated_user].iloc[0]
        badge_class = f"badge-{curr_user_row['risk_tier'].lower()}"
        st.markdown(f"<div style='text-align:right;'><span class='{badge_class}'>{curr_user_row['risk_tier']} RISK ({curr_user_row['risk_score']}/100)</span></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Explainability breakdown
    explanation = detector.explain_user(curr_user_row, top_k=6)
    queue_info = queue_mgr.get_user_status(investigated_user)
    
    # Profile & Threat Summary Card
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        st.markdown(f"**Department:** `{curr_user_row['department']}`")
        st.markdown(f"**Primary Workstation:** `{curr_user_row.get('primary_device', 'DEV-WS-001')}`")
    with p_col2:
        st.markdown(f"**Current Status:** `{queue_info['status']}`")
        st.markdown(f"**Last Triage:** `{queue_info['last_updated']}`")
    with p_col3:
        st.markdown(f"**Total Volume:** `{curr_user_row['total_mb_xfer']:.1f} MB`")
        st.markdown(f"**Off-Hours Ratio:** `{curr_user_row['off_hours_ratio']*100:.1f}%`")
    with p_col4:
        st.markdown(f"**USB Devices:** `{int(curr_user_row['usb_connects'])}`")
        st.markdown(f"**Failed Logins:** `{int(curr_user_row['failed_logins'])}`")

    # Threat Intelligence Narrative Alert Box
    if curr_user_row["risk_tier"] in ["CRITICAL", "HIGH"]:
        st.markdown(f"""
        <div class="soc-threat-alert-box">
            <div class="soc-alert-title" style="color: #fca5a5;">
                <span style="font-size: 1.25rem;">🚨</span> 
                <span>CRITICAL SECURITY ALERT // ANOMALOUS INSIDER DETECTED</span>
                <span class="badge-critical" style="margin-left: auto;">RISK: {curr_user_row['risk_score']:.1f}/100</span>
            </div>
            <div class="soc-alert-body">
                {explanation['analyst_narrative']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif curr_user_row["risk_tier"] == "MEDIUM":
        st.markdown(f"""
        <div class="soc-medium-alert-box">
            <div class="soc-alert-title" style="color: #fde047;">
                <span style="font-size: 1.25rem;">⚠️</span> 
                <span>ELEVATED RISK NOTICE // BEHAVIORAL DEVIATION</span>
                <span class="badge-medium" style="margin-left: auto;">RISK: {curr_user_row['risk_score']:.1f}/100</span>
            </div>
            <div class="soc-alert-body">
                {explanation['analyst_narrative']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="soc-normal-alert-box">
            <div class="soc-alert-title" style="color: #86efac;">
                <span style="font-size: 1.25rem;">✅</span> 
                <span>NOMINAL OPERATIONAL BASELINE // NO ANOMALIES DETECTED</span>
                <span class="badge-low" style="margin-left: auto;">RISK: {curr_user_row['risk_score']:.1f}/100</span>
            </div>
            <div class="soc-alert-body">
                {explanation['analyst_narrative']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Visualizations Row 1: XAI Feature Attribution & Radar Peer Comparison
    st.markdown("#### 🧠 Anomaly Attribution & Peer Baseline Comparison")
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        # Waterfall / Bar chart of top anomaly drivers
        top_factors = pd.DataFrame(explanation["top_contributions"])
        fig_bar = px.bar(
            top_factors,
            x="robust_zscore",
            y="feature",
            orientation="h",
            color="robust_zscore",
            color_continuous_scale="Reds",
            title=f"Feature Deviation vs Baseline ({investigated_user})",
            labels={"robust_zscore": "Standardized Deviation (+σ)", "feature": "Behavioral Metric"}
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.85)",
            plot_bgcolor="rgba(15, 23, 42, 0.85)",
            yaxis={"categoryorder": "total ascending"},
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with vis_col2:
        # Radar chart comparing user against department baseline
        dept_name = curr_user_row["department"]
        dept_peers = scored_users_df[scored_users_df["department"] == dept_name]
        
        radar_metrics = [
            ("Failed Logins", "failed_logins", 10.0),
            ("Off-Hours Act.", "off_hours_events", 30.0),
            ("USB Connects", "usb_connects", 5.0),
            ("Sensitive Access", "sensitive_file_count", 20.0),
            ("Data Vol (MB)", "total_mb_xfer", 1500.0),
            ("Cloud Uploads", "risky_cloud_count", 5.0)
        ]
        
        categories = [m[0] for m in radar_metrics]
        # Normalize to 0-100 scale for radar
        user_vals = [min(100.0, (curr_user_row[m[1]] / m[2]) * 100.0) for m in radar_metrics]
        dept_vals = [min(100.0, (dept_peers[m[1]].mean() / m[2]) * 100.0) for m in radar_metrics]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f"{investigated_user}",
            line_color='#ef4444' if curr_user_row["risk_tier"] in ["CRITICAL", "HIGH"] else '#38bdf8'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=dept_vals + [dept_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f"{dept_name} Baseline",
            line_color='#64748b'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.85)",
            title=f"Peer Group Comparison ({dept_name})",
            height=320,
            margin=dict(l=30, r=30, t=40, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Activity Timeline & Audit Log
    st.markdown("#### 🕒 User Audit Trail & Activity Timeline")
    user_raw_logs = raw_logs[raw_logs["user_id"] == investigated_user].sort_values(by="timestamp", ascending=False)
    
    # Group by day for timeline chart
    user_daily = user_raw_logs.copy()
    user_daily["date"] = user_daily["timestamp"].dt.date
    user_daily["bytes_mb"] = user_daily["bytes_transferred"] / (1024 * 1024)
    daily_summary = user_daily.groupby("date").agg(
        events=("action_type", "count"),
        bytes_mb=("bytes_mb", "sum"),
        off_hours=("is_off_hours", "sum")
    ).reset_index()
    
    fig_time = px.line(
        daily_summary,
        x="date",
        y="bytes_mb",
        title="Daily Data Transfer Volume (MB) Over Time",
        markers=True,
        labels={"bytes_mb": "MB Transferred", "date": "Observation Date"}
    )
    fig_time.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.85)",
        plot_bgcolor="rgba(15, 23, 42, 0.85)",
        height=240,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Filterable raw log events
    with st.expander(f"📜 View Raw SIEM Audit Trail ({len(user_raw_logs)} events)", expanded=False):
        st.dataframe(
            user_raw_logs[["timestamp", "action_type", "resource", "sensitivity", "destination", "status", "bytes_transferred", "is_off_hours"]].head(100),
            use_container_width=True,
            height=260
        )

    # AI Threat Copilot & MITRE ATT&CK Investigation
    st.markdown("---")
    st.markdown("### 🤖 AI Cyber Forensics Copilot (Groq / Gemini)")
    ai_col1, ai_col2 = st.columns([1, 2])
    with ai_col1:
        st.markdown(
            "Leverage LLM reasoning over anomalous tree isolation paths, peer $Z$-score deviations, and raw SIEM events to generate an executive threat briefing."
        )
        run_ai_btn = st.button("⚡ Generate MITRE ATT&CK Briefing", use_container_width=True)
    with ai_col2:
        if run_ai_btn:
            with st.spinner("Analyzing log telemetry with AI Cyber Forensics engine..."):
                p_slug = "groq" if "Groq" in ai_provider else "gemini"
                ai_briefing = query_ai_analyst(
                    investigated_user,
                    curr_user_row,
                    explanation,
                    user_raw_logs.head(20),
                    provider=p_slug,
                    api_key=api_key_input
                )
                st.session_state[f"ai_report_{investigated_user}"] = ai_briefing
                
        if f"ai_report_{investigated_user}" in st.session_state:
            st.markdown(st.session_state[f"ai_report_{investigated_user}"])
        else:
            st.info("💡 Click **'Generate MITRE ATT&CK Briefing'** above to activate the AI Analyst Copilot. (Make sure to enter your free Groq or Gemini API key in the sidebar).")

    # Incident Response Action Center & Containment Playbook
    st.markdown("---")
    st.markdown("### 🛠️ Incident Triage & Containment Center")
    t_col1, t_col2 = st.columns([1, 1])
    
    with t_col1:
        st.subheader("Update Incident Status")
        new_status = st.selectbox(
            "Triage Decision:",
            options=STATUS_OPTIONS,
            index=STATUS_OPTIONS.index(queue_info["status"]) if queue_info["status"] in STATUS_OPTIONS else 0
        )
        assigned = st.text_input("Lead SOC Analyst:", value=queue_info.get("assigned_analyst", "SOC_Analyst_Alpha"))
        analyst_notes_input = st.text_area(
            "Analyst Investigation Notes & Evidence Log:",
            value=queue_info.get("analyst_notes", ""),
            height=120,
            placeholder="Document observed indicators of compromise (IoCs), interview findings, or justification for resolution..."
        )
        
        save_btn_col1, save_btn_col2 = st.columns(2)
        with save_btn_col1:
            if st.button("💾 Save Triage State", use_container_width=True):
                queue_mgr.update_user_status(investigated_user, new_status, analyst_notes_input, assigned)
                st.success("Triage status and notes saved successfully!")
                st.rerun()
        with save_btn_col2:
            # Download full dossier
            dossier_text = generate_investigation_report(
                investigated_user,
                curr_user_row,
                explanation,
                {"status": new_status, "assigned_analyst": assigned, "analyst_notes": analyst_notes_input},
                user_raw_logs.head(25)
            )
            st.download_button(
                "📄 Download Incident Briefing",
                data=dossier_text,
                file_name=f"incident_dossier_{investigated_user}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    with t_col2:
        st.subheader("Recommended Containment Playbook")
        scenario_key = curr_user_row.get("threat_scenario", "GENERAL_ANOMALY")
        playbook = PLAYBOOKS.get(scenario_key, PLAYBOOKS["GENERAL_ANOMALY"])
        st.info(f"Target Playbook Triggered: `{scenario_key}`")
        for step in playbook:
            st.markdown(f"- {step}")


# ==========================================
# TAB 3: MODEL PERFORMANCE & BENCHMARKS
# ==========================================
with tab_benchmark:
    st.markdown("### 📊 Isolation Forest Model Evaluation & Multi-Detector Benchmark")
    st.markdown(
        "Demonstrates the theoretical and empirical superiority of **Isolation Forest** compared to "
        "traditional anomaly detection paradigms (**One-Class SVM**, **Local Outlier Factor**, and **Z-Score Baseline**) "
        "on enterprise log telemetry."
    )
    
    with st.spinner("Calculating cybersecurity evaluation metrics across detector models..."):
        benchmark_results, curves_data = run_comprehensive_benchmark(features_df, users_meta, contamination=detector.contamination)
        
    # Metric Summary Table
    st.dataframe(
        benchmark_results.rename(columns={
            "model": "Anomaly Detector Algorithm",
            "precision": "Precision (PPV)",
            "recall": "Recall (Sensitivity)",
            "f1_score": "F1-Score",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "false_positive_rate": "False Positive Rate (FPR)",
            "inference_latency_ms": "Inference Latency (ms/user)"
        }),
        use_container_width=True
    )
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        # ROC Curves
        fig_roc = go.Figure()
        for m_name, c_dict in curves_data.items():
            fig_roc.add_trace(go.Scatter(
                x=c_dict["fpr"], y=c_dict["tpr"],
                mode='lines', name=m_name,
                line=dict(width=2.5 if "Isolation" in m_name else 1.5)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Random Guess (0.50)',
            line=dict(dash='dash', color='gray')
        ))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC Curves)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.85)",
            plot_bgcolor="rgba(15, 23, 42, 0.85)",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with b_col2:
        # Precision-Recall Curves
        fig_pr = go.Figure()
        for m_name, c_dict in curves_data.items():
            fig_pr.add_trace(go.Scatter(
                x=c_dict["recall"], y=c_dict["precision"],
                mode='lines', name=m_name,
                line=dict(width=2.5 if "Isolation" in m_name else 1.5)
            ))
        fig_pr.update_layout(
            title="Precision-Recall Curves (Critical for Rare Threats)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.85)",
            plot_bgcolor="rgba(15, 23, 42, 0.85)",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        
    # Anomaly Score Distribution Histogram
    st.markdown("#### 📈 Isolation Forest Score Distribution & Contamination Threshold")
    fig_hist = px.histogram(
        scored_users_df,
        x="raw_anomaly_score",
        color="risk_tier",
        nbins=40,
        title="Distribution of Raw Tree Path Isolation Scores (Lower = Shorter Path / Anomaly)",
        color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
    )
    fig_hist.add_vline(x=detector.threshold, line_dash="dash", line_color="red", annotation_text="Contamination Cutoff")
    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.85)",
        plot_bgcolor="rgba(15, 23, 42, 0.85)",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ==========================================
# TAB 4: THREAT SIMULATOR & LOG INGESTION
# ==========================================
with tab_simulator:
    st.markdown("### 🧪 Live Threat Scenario Simulator & SIEM Log Ingestion")
    st.markdown(
        "Test how the Isolation Forest pipeline behaves when presented with newly injected insider threats "
        "or arbitrary external log batches."
    )
    
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.markdown("#### Inject Synthetic Attack Scenarios")
        scenario_choice = st.selectbox(
            "Select Pre-Built Attack Vector:",
            options=[
                "Scenario A: Departing Engineer Exfiltrating Source Code to Mega.nz & USB",
                "Scenario B: Rogue System Admin Probing Domain Controllers at 3 AM",
                "Scenario C: Covert Financial Leaker Mailing Confidential PII to ProtonMail",
                "Scenario D: Benign Normal Enterprise User (Control Test)"
            ]
        )
        
        sim_user_id = st.text_input("Simulated Employee ID:", value="USR_SIM_9999")
        sim_dept = st.selectbox("Department:", options=["Engineering", "IT_Admin", "Finance", "Sales"])
        
        if st.button("🚀 Inject Attack Vector & Run Anomaly Detector", use_container_width=True):
            # Generate simulated user behavioral vector
            if "Scenario A" in scenario_choice:
                sim_row = {
                    "user_id": sim_user_id, "department": sim_dept,
                    "total_events": 280, "login_count": 14, "failed_logins": 2, "failed_login_ratio": 0.14,
                    "off_hours_events": 95, "off_hours_ratio": 0.34, "weekend_events": 40,
                    "usb_connects": 6, "distinct_devices": 3, "sensitive_file_count": 22,
                    "sensitive_ratio": 0.65, "file_copy_count": 18, "distinct_sensitive_paths": 6,
                    "total_mb_xfer": 3450.0, "max_single_xfer_mb": 1800.0, "avg_xfer_per_event_mb": 12.3,
                    "risky_cloud_count": 5, "risky_cloud_bytes_mb": 2200.0, "ext_email_count": 1,
                    "ext_email_bytes_mb": 20.0, "max_events_per_hour": 75, "peer_vol_zscore": 6.8
                }
            elif "Scenario B" in scenario_choice:
                sim_row = {
                    "user_id": sim_user_id, "department": sim_dept,
                    "total_events": 190, "login_count": 45, "failed_logins": 38, "failed_login_ratio": 0.84,
                    "off_hours_events": 110, "off_hours_ratio": 0.58, "weekend_events": 25,
                    "usb_connects": 1, "distinct_devices": 7, "sensitive_file_count": 8,
                    "sensitive_ratio": 0.40, "file_copy_count": 4, "distinct_sensitive_paths": 3,
                    "total_mb_xfer": 450.0, "max_single_xfer_mb": 120.0, "avg_xfer_per_event_mb": 2.3,
                    "risky_cloud_count": 0, "risky_cloud_bytes_mb": 0.0, "ext_email_count": 0,
                    "ext_email_bytes_mb": 0.0, "max_events_per_hour": 60, "peer_vol_zscore": 1.8
                }
            elif "Scenario C" in scenario_choice:
                sim_row = {
                    "user_id": sim_user_id, "department": sim_dept,
                    "total_events": 160, "login_count": 15, "failed_logins": 1, "failed_login_ratio": 0.06,
                    "off_hours_events": 65, "off_hours_ratio": 0.40, "weekend_events": 35,
                    "usb_connects": 0, "distinct_devices": 1, "sensitive_file_count": 15,
                    "sensitive_ratio": 0.50, "file_copy_count": 6, "distinct_sensitive_paths": 4,
                    "total_mb_xfer": 850.0, "max_single_xfer_mb": 200.0, "avg_xfer_per_event_mb": 5.3,
                    "risky_cloud_count": 0, "risky_cloud_bytes_mb": 0.0, "ext_email_count": 12,
                    "ext_email_bytes_mb": 650.0, "max_events_per_hour": 35, "peer_vol_zscore": 3.4
                }
            else: # Benign Control
                sim_row = {
                    "user_id": sim_user_id, "department": sim_dept,
                    "total_events": 380, "login_count": 14, "failed_logins": 0, "failed_login_ratio": 0.0,
                    "off_hours_events": 8, "off_hours_ratio": 0.02, "weekend_events": 0,
                    "usb_connects": 0, "distinct_devices": 1, "sensitive_file_count": 2,
                    "sensitive_ratio": 0.03, "file_copy_count": 1, "distinct_sensitive_paths": 1,
                    "total_mb_xfer": 85.0, "max_single_xfer_mb": 12.0, "avg_xfer_per_event_mb": 0.22,
                    "risky_cloud_count": 0, "risky_cloud_bytes_mb": 0.0, "ext_email_count": 0,
                    "ext_email_bytes_mb": 0.0, "max_events_per_hour": 25, "peer_vol_zscore": -0.4
                }
                
            sim_df = pd.DataFrame([sim_row])
            sim_pred = detector.predict_risk(sim_df)
            sim_exp = detector.explain_user(sim_df.iloc[0])
            
            st.session_state["last_sim_result"] = {
                "pred": sim_pred.iloc[0].to_dict(),
                "exp": sim_exp,
                "data": sim_row
            }
            
        if "last_sim_result" in st.session_state:
            res = st.session_state["last_sim_result"]
            p = res["pred"]
            st.markdown("---")
            st.markdown(f"### Detection Result: `{res['data']['user_id']}`")
            tier_color = "#ef4444" if p["risk_tier"] == "CRITICAL" else ("#f97316" if p["risk_tier"] == "HIGH" else "#22c55e")
            st.markdown(f"**Risk Tier:** <span style='color:{tier_color}; font-weight:700; font-size:1.2rem;'>{p['risk_tier']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Calibrated Risk Score:** **{p['risk_score']}/100**")
            st.markdown(f"**Model Anomaly Flag:** `{p['is_anomaly']}` (Raw Path Score: `{p['raw_anomaly_score']}`)")
            st.info(f"**Attribution Narrative:** {res['exp']['analyst_narrative']}")

    with sim_col2:
        st.markdown("#### Upload Raw SIEM Event CSV")
        st.markdown(
            "Upload any CSV file matching standard SIEM event headers (`user_id`, `timestamp`, `action_type`, `bytes_transferred`, `sensitivity`, etc.):"
        )
        uploaded_file = st.file_uploader("Upload Raw Log CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.write(f"Parsed {len(uploaded_df):,} events from `{uploaded_file.name}`.")
                if st.button("⚙️ Transform & Score Log Stream", use_container_width=True):
                    with st.spinner("Extracting behavioral features and scoring with Isolation Forest..."):
                        up_features = compute_user_behavior_features(uploaded_df)
                        up_preds = detector.predict_risk(up_features)
                        up_combined = pd.concat([up_features, up_preds], axis=1).sort_values(by="risk_score", ascending=False)
                        st.success(f"Successfully processed {len(up_combined)} user profiles!")
                        st.dataframe(up_combined[["user_id", "department", "risk_tier", "risk_score", "total_mb_xfer", "off_hours_events"]].head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error parsing log file: {str(e)}")

st.markdown("---")
st.caption("AI-Driven Cybersecurity Threat Detection Engine • Powered by Scikit-Learn Isolation Forest & Streamlit")
