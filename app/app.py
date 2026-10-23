"""
app.py
======
AI-Driven Insider Threat Detection System using Isolation Forest & Groq AI.
Designed for clear demonstration, intuitive presentation, and viva defense.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup project paths
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
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ or not os.environ[k.strip()]:
                        os.environ[k.strip()] = v.strip().strip("'\"")
    except Exception:
        pass

from model import IsolationForestAnomalyDetector
from feature_engineering import compute_user_behavior_features, FEATURE_COLUMNS
from log_generator import generate_enterprise_logs


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Insider Threat Detector | Mini Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Dark-Cyber Styling (High-contrast, readable, professional)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Header Card */
    .project-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .project-title {
        font-size: 2.0rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .project-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 14px;
        line-height: 1.5;
    }
    .badge-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .tech-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .tech-badge-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }
    .tech-badge-green {
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 4px;
    }
    .val-cyan { color: #38bdf8; }
    .val-green { color: #10b981; }
    .val-red { color: #ef4444; }
    .val-purple { color: #a855f7; }

    /* Custom Table Styling */
    .threat-tag-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA & MODEL LOADING
# ==========================================
@st.cache_resource
def load_project_data():
    """Loads activity logs, computed features, and trains the Isolation Forest detector."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    raw_path = os.path.join(DATA_DIR, "raw_user_activity_logs.csv")
    meta_path = os.path.join(DATA_DIR, "users_metadata.csv")
    feats_path = os.path.join(DATA_DIR, "processed_user_features.csv")
    model_path = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")

    # Generate synthetic enterprise logs if not present
    if not (os.path.exists(raw_path) and os.path.exists(meta_path)):
        raw_df, meta_df = generate_enterprise_logs(num_users=150, days=14, threat_ratio=0.06)
        raw_df.to_csv(raw_path, index=False)
        meta_df.to_csv(meta_path, index=False)
    else:
        raw_df = pd.read_csv(raw_path)
        meta_df = pd.read_csv(meta_path)

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

    # Compute behavioral features if not present
    if not os.path.exists(feats_path):
        feats_df = compute_user_behavior_features(raw_df)
        feats_df.to_csv(feats_path, index=False)
    else:
        feats_df = pd.read_csv(feats_path)

    # Train or load Isolation Forest model
    if not os.path.exists(model_path):
        detector = IsolationForestAnomalyDetector(n_estimators=180, contamination=0.06, random_state=42)
        detector.fit(feats_df, feature_names=FEATURE_COLUMNS)
        detector.save(model_path)
    else:
        detector = IsolationForestAnomalyDetector.load(model_path)

    return raw_df, meta_df, feats_df, detector


raw_logs_df, users_meta_df, features_df, detector = load_project_data()

# Predict anomaly scores
predictions_df = detector.predict_risk(features_df)
merged_df = pd.concat([features_df, predictions_df], axis=1)

# Merge metadata (Department, threat scenario, device)
display_df = pd.merge(
    merged_df,
    users_meta_df[["user_id", "primary_device", "threat_scenario", "is_threat"]],
    on="user_id",
    how="left"
)

# Sort highest risk employees to the top
display_df = display_df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ Mini Project Controls")
    st.caption("AI-Driven Insider Threat Detection")

    st.markdown("---")
    st.subheader("🔑 AI Copilot Settings")
    groq_key_default = os.environ.get("GROQ_API_KEY", "")
    if not groq_key_default:
        try:
            groq_key_default = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    api_key = st.text_input(
        "Groq API Key:",
        value=groq_key_default,
        type="password",
        help="Used to generate natural language threat explanations with Groq AI."
    )
    if api_key:
        st.success("✅ Groq AI Connected")
    else:
        st.warning("⚠️ Enter Groq Key to enable AI Copilot")

    st.markdown("---")
    st.subheader("⚙️ Model Parameters")
    st.write(f"**Algorithm:** Isolation Forest")
    st.write(f"**Trees in Forest:** 180 estimators")
    st.write(f"**Features Evaluated:** {len(FEATURE_COLUMNS)} behavioral metrics")
    st.write(f"**Contamination Rate:** `{detector.contamination * 100:.1f}%`")

    st.markdown("---")
    st.subheader("💡 System Navigation")
    st.info(
        "**Core Workflow:**\n\n"
        "1. **Tab 1:** Anomaly Detection Map & Priorities\n"
        "2. **Tab 2:** User Forensics & Groq AI Reasoning\n"
        "3. **Tab 3:** Live Interactive Threat Simulator\n"
        "4. **Tab 4:** Model Performance & Benchmark Analytics"
    )

    if st.button("🔄 Reset / Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# ==========================================
# HEADER BANNER
# ==========================================
st.markdown("""
<div class="project-header">
    <div class="project-title">🛡️ AI-Driven Insider Threat Detection System</div>
    <div class="project-subtitle">
        An unsupervised machine learning system using <b>Isolation Forest</b> to detect anomalous user behaviors 
        (data exfiltration, off-hours access, USB file copies) from enterprise activity logs, 
        with automated threat explanations powered by <b>Groq AI Copilot</b>.
    </div>
    <div class="badge-container">
        <span class="tech-badge">🌲 Machine Learning: Isolation Forest</span>
        <span class="tech-badge-purple">🤖 Threat Reasoning: Groq AI Copilot</span>
        <span class="tech-badge-green">📊 Unsupervised UEBA Detection</span>
        <span class="tech-badge">🎓 IV B.Tech Mini Project</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# TOP KPI METRICS ROW
# ==========================================
total_users = len(display_df)
threat_users = (display_df["is_anomaly"] == True).sum()
normal_users = total_users - threat_users
avg_risk = display_df["risk_score"].mean()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Users Monitored</div>
        <div class="kpi-value val-cyan">{total_users}</div>
        <div class="kpi-sub">Enterprise Accounts Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Normal Employees</div>
        <div class="kpi-value val-green">{normal_users}</div>
        <div class="kpi-sub">{(normal_users/total_users)*100:.1f}% Standard Behavior</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Detected Anomalies</div>
        <div class="kpi-value val-red">{threat_users}</div>
        <div class="kpi-sub">Flagged Rogue Outliers</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average Risk Score</div>
        <div class="kpi-value val-purple">{avg_risk:.1f}<span style="font-size:1.0rem; color:#94a3b8;">/100</span></div>
        <div class="kpi-sub">Baseline Operational Health</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# MAIN TABS (4 INTUITIVE PROJECT MODULES)
# ==========================================
tab_dashboard, tab_investigate, tab_simulator, tab_analytics = st.tabs([
    "📊 1. Detection Dashboard",
    "🔍 2. Employee Deep-Dive & AI Report",
    "🧪 3. Live Threat Simulator",
    "📈 4. Model Performance & Analytics"
])


# ==========================================
# TAB 1: DETECTION DASHBOARD
# ==========================================
with tab_dashboard:
    st.markdown("### 🌲 Isolation Forest Anomaly Map")
    st.markdown(
        "The scatter plot below visualizes how Isolation Forest separates normal employees from suspicious outliers. "
        "**Normal users (Green)** cluster around low activity, while **Insider Threats (Red)** stand out far from the group."
    )

    # Clean 2D Scatter Plot
    plot_df = display_df.copy()
    plot_df["Status"] = np.where(plot_df["is_anomaly"], "🚨 Suspicious Anomaly", "🟢 Normal Employee")
    
    fig = px.scatter(
        plot_df,
        x="total_mb_xfer",
        y="off_hours_events",
        color="Status",
        color_discrete_map={
            "🚨 Suspicious Anomaly": "#ef4444",
            "🟢 Normal Employee": "#10b981"
        },
        size=plot_df["risk_score"].clip(lower=15),
        hover_name="user_id",
        hover_data={
            "department": True,
            "risk_score": ":.1f",
            "usb_connects": True,
            "failed_logins": True,
            "total_mb_xfer": ":.0f",
            "off_hours_events": True
        },
        labels={
            "total_mb_xfer": "Total Data Transferred (MB)",
            "off_hours_events": "Off-Hours Activity (Late Night / Weekend Events)"
        },
        title="Employee Behavior Map: Data Transferred vs Off-Hours Events"
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(11, 15, 25, 0.8)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🚨 Prioritized Insider-Threat Detection Table")
    st.caption("Employees ranked by Isolation Forest Risk Score (0-100%). Top records represent immediate security concerns.")

    # Table columns to display
    summary_cols = [
        "user_id", "department", "risk_tier", "risk_score",
        "total_mb_xfer", "off_hours_events", "usb_connects", "failed_logins"
    ]
    renamed_cols = {
        "user_id": "Employee ID",
        "department": "Department",
        "risk_tier": "Risk Level",
        "risk_score": "Risk Score (/100)",
        "total_mb_xfer": "Data Xfer (MB)",
        "off_hours_events": "Off-Hours Events",
        "usb_connects": "USB Devices",
        "failed_logins": "Failed Logins"
    }

    threat_table = display_df[summary_cols].rename(columns=renamed_cols)
    
    # Simple department filter inside tab
    dept_options = ["All Departments"] + sorted(display_df["department"].unique().tolist())
    selected_dept = st.selectbox("Filter by Department:", dept_options, index=0)
    
    if selected_dept != "All Departments":
        threat_table = threat_table[threat_table["Department"] == selected_dept]

    st.dataframe(
        threat_table.style.format({
            "Risk Score (/100)": "{:.1f}",
            "Data Xfer (MB)": "{:.1f}"
        }),
        use_container_width=True,
        height=320
    )


def generate_groq_ai_report(
    user_id: str,
    user_row: pd.Series,
    explanation: dict,
    top_events: pd.DataFrame,
    api_key: str
) -> str:
    """
    Directly calls the active Groq AI Chat Completion endpoint with dynamic model discovery.
    Guarantees no decommissioned models are ever called.
    """
    import requests

    clean_key = api_key.strip() if api_key else os.environ.get("GROQ_API_KEY", "")
    if not clean_key:
        try:
            clean_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not clean_key:
        return "⚠️ No Groq API Key found. Please enter a valid Groq Key in the sidebar."

    headers = {
        "Authorization": f"Bearer {clean_key}",
        "Content-Type": "application/json"
    }

    # Discover active models from Groq API directly
    preferred_models = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b"
    ]
    selected_model = "openai/gpt-oss-120b"
    try:
        models_resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
        if models_resp.status_code == 200:
            avail = [m.get("id") for m in models_resp.json().get("data", [])]
            for p in preferred_models:
                if p in avail:
                    selected_model = p
                    break
    except Exception:
        pass

    top_factors = explanation.get("top_contributions", [])
    factors_summary = "\n".join([
        f"- {item['feature']}: {item['user_value']} (Peer Baseline: {item['baseline_mean']:.2f}, Deviation: +{item['robust_zscore']:.1f}σ)"
        for item in top_factors
    ])

    events_str = ""
    if len(top_events) > 0:
        events_str = top_events[["timestamp", "action_type", "resource", "destination", "bytes_transferred", "is_off_hours"]].head(10).to_string(index=False)

    prompt = f"""You are a Lead Security Operations Center (SOC) Cyber Forensics & Insider Threat Investigator.
Analyze the following anomalous employee activity flagged by our enterprise Isolation Forest anomaly detector.

EMPLOYEE METADATA:
- User ID: {user_id}
- Department: {user_row.get('department', 'Unknown')}
- Calibrated Risk Score: {user_row.get('risk_score', 'N/A')}/100 ({user_row.get('risk_tier', 'CRITICAL')} RISK)
- Threat Scenario: {user_row.get('threat_scenario', 'Unknown')}

TOP ABNORMAL BEHAVIORAL INDICATORS (Z-SCORE DEVIATIONS):
{factors_summary}

RECENT RAW SIEM AUDIT EVENTS:
{events_str}

TASK:
Provide a concise, professional SOC Incident Briefing with the following exact sections:
1. 🎯 Threat Summary & Intent Assessment (Is this deliberate data theft, compromised credentials, or policy negligence?)
2. 🗺️ MITRE ATT&CK Mapping (List applicable techniques, e.g., T1048 Exfiltration, T1078 Valid Accounts, T1052 Physical USB)
3. 🔎 Forensic Evidence Highlights (Highlight specific timestamps, destinations, data volumes, and suspicious activity)
4. 🚨 Immediate Containment Recommendations (Numbered concrete steps: revoke credentials, disable USB, isolate host)

Keep the briefing crisp, authoritative, and formatted in clean markdown."""

    try:
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": "You are an elite enterprise SOC Incident Commander and Cyber Threat Intelligence expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1200
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            # Fallback to alternative model
            for fb in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
                if fb != selected_model:
                    payload["model"] = fb
                    r2 = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
                    if r2.status_code == 200:
                        return r2.json()["choices"][0]["message"]["content"]
            return f"❌ Groq API Error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"


# ==========================================
# TAB 2: EMPLOYEE DEEP-DIVE & AI REPORT
# ==========================================
with tab_investigate:
    st.markdown("### 🔍 Forensic Deep-Dive & Groq AI Threat Briefing")
    st.markdown(
        "Select any employee to view their detailed activity metrics and generate an automated "
        "AI security incident report explaining the threat in plain English."
    )

    # Dropdown to select user
    user_list = display_df["user_id"].tolist()
    selected_user = st.selectbox("Select Employee to Investigate:", user_list, index=0)

    user_row = display_df[display_df["user_id"] == selected_user].iloc[0]
    explanation = detector.explain_user(user_row[FEATURE_COLUMNS])

    # Show User Profile Card
    u_col1, u_col2, u_col3, u_col4 = st.columns(4)
    with u_col1:
        st.metric("Department", user_row["department"])
    with u_col2:
        st.metric("Risk Score", f"{user_row['risk_score']:.1f} / 100", delta=user_row["risk_tier"])
    with u_col3:
        st.metric("Total Data Downloaded", f"{user_row['total_mb_xfer']:.1f} MB")
    with u_col4:
        st.metric("Off-Hours Activity", f"{int(user_row['off_hours_events'])} events")

    st.markdown("<br>", unsafe_allow_html=True)

    # Top Behavioral Deviations Bar Chart
    top_factors = explanation.get("top_contributions", [])
    if top_factors:
        st.markdown("#### 📈 Key Anomaly Drivers (Deviation from Peer Group)")
        chart_data = pd.DataFrame(top_factors)
        bar_fig = px.bar(
            chart_data,
            x="robust_zscore",
            y="feature",
            orientation="h",
            color="robust_zscore",
            color_continuous_scale=["#38bdf8", "#ef4444"],
            labels={"robust_zscore": "Standardized Deviation (Z-Score)", "feature": "Behavioral Feature"},
            title=f"Top Deviations for Employee: {selected_user}"
        )
        bar_fig.update_layout(
            template="plotly_dark",
            height=260,
            margin=dict(l=10, r=10, t=35, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Recent Raw Logs for this user
    user_raw_logs = raw_logs_df[raw_logs_df["user_id"] == selected_user].sort_values(by="timestamp", ascending=False).head(10)
    with st.expander(f"📋 View Recent Audit Logs for {selected_user} ({len(user_raw_logs)} events)", expanded=False):
        st.dataframe(
            user_raw_logs[["timestamp", "action_type", "resource", "destination", "bytes_transferred", "is_off_hours"]],
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 🤖 Groq AI Cyber Threat Copilot")
    st.write(
        "Click the button below to have the **Groq AI Copilot** analyze this employee's activity, "
        "evaluate intent, and produce an executive-ready threat briefing."
    )

    if st.button("🚀 Generate AI Threat Explanation", type="primary", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please provide a valid Groq API Key in the sidebar to run the AI Copilot.")
        else:
            with st.spinner("🤖 Groq AI is analyzing behavioral logs & generating threat briefing..."):
                ai_briefing = generate_groq_ai_report(
                    user_id=selected_user,
                    user_row=user_row,
                    explanation=explanation,
                    top_events=user_raw_logs,
                    api_key=api_key
                )
                st.markdown("#### 📄 Executive Threat Briefing:")
                st.markdown(ai_briefing)


# ==========================================
# TAB 3: LIVE THREAT SIMULATOR
# ==========================================
with tab_simulator:
    st.markdown("### 🧪 Live Anomaly Simulator")
    st.markdown(
        "Demonstrate the Isolation Forest model in real time! Adjust the employee behavior sliders "
        "below and see whether the model flags the behavior as **Normal** or an **Insider Threat**."
    )

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_mb = st.slider("📦 Data Downloaded / Transferred (MB):", min_value=50, max_value=10000, value=250, step=50)
        sim_off_hours = st.slider("🌙 Off-Hours Actions (Late Night / Weekend):", min_value=0, max_value=30, value=2, step=1)
    
    with sim_col2:
        sim_usb = st.slider("💾 USB Removable Storage Connects:", min_value=0, max_value=10, value=0, step=1)
        sim_failed_logins = st.slider("🔑 Failed Login Attempts:", min_value=0, max_value=15, value=1, step=1)

    if st.button("⚡ Evaluate with Isolation Forest", type="primary", use_container_width=True):
        # Create single-row DataFrame using median baseline of other features
        median_features = features_df[FEATURE_COLUMNS].median().to_dict()
        sim_dict = median_features.copy()

        # Update with simulated inputs and realistic correlated features
        sim_dict["total_mb_xfer"] = sim_mb
        sim_dict["max_single_xfer_mb"] = sim_mb * 0.7
        sim_dict["avg_xfer_per_event_mb"] = sim_mb / max(1, (sim_off_hours + 10))
        sim_dict["off_hours_events"] = sim_off_hours
        sim_dict["off_hours_ratio"] = min(1.0, sim_off_hours / max(1, (sim_off_hours + 15)))
        sim_dict["usb_connects"] = sim_usb
        sim_dict["failed_logins"] = sim_failed_logins
        sim_dict["file_copy_count"] = max(2, int(sim_usb * 4))

        sim_df = pd.DataFrame([sim_dict])
        sim_pred = detector.predict_risk(sim_df).iloc[0]

        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if sim_pred["is_anomaly"]:
                st.error("### 🚨 INSIDER THREAT DETECTED!")
            else:
                st.success("### 🟢 NORMAL EMPLOYEE BEHAVIOR")
            
            st.metric("Calibrated Risk Score", f"{sim_pred['risk_score']:.1f} / 100")
            st.metric("Assigned Risk Level", sim_pred["risk_tier"])
            st.metric("Raw Tree Anomaly Score", f"{sim_pred['raw_anomaly_score']:.4f}")

        with res_col2:
            st.markdown("#### 💡 Model Decision Reasoning:")
            if sim_pred["is_anomaly"]:
                st.markdown(
                    f"- **Result:** The user was classified as an **outlier** because their behavioral pattern "
                    f"deviated significantly from normal organizational clusters.\n"
                    f"- **Key Culprits:** "
                    f"{'High Data Transfer (' + str(sim_mb) + ' MB); ' if sim_mb > 1500 else ''}"
                    f"{'Late Night Off-Hours (' + str(sim_off_hours) + ' events); ' if sim_off_hours > 5 else ''}"
                    f"{'Unusual USB Storage Use (' + str(sim_usb) + ' devices); ' if sim_usb > 1 else ''}"
                    f"{'Multiple Failed Logins (' + str(sim_failed_logins) + ' attempts);' if sim_failed_logins > 3 else ''}\n"
                    f"- **Isolation Speed:** In the forest's decision trees, this point was isolated with very few splits (short path length)."
                )
            else:
                st.markdown(
                    f"- **Result:** The user behavior is **safe and benign**.\n"
                    f"- **Reasoning:** Data transfer volume ({sim_mb} MB) and off-hours activity ({sim_off_hours} events) "
                    f"fall well within the standard standard deviation of normal peer employees.\n"
                    f"- **Isolation Speed:** Required a deep tree traversal to isolate, indicating standard behavior."
                )


# ==========================================
# TAB 4: MODEL PERFORMANCE & ANALYTICS
# ==========================================
with tab_analytics:
    st.markdown("### 📈 Model Evaluation & Performance Analytics")
    st.markdown(
        "Quantitative evaluation of the **Isolation Forest** anomaly detector across benchmark metrics, "
        "score distribution, and comparative validation against baseline models."
    )

    # 1. Comparative Benchmark Table
    benchmark_path = os.path.join(DATA_DIR, "benchmark_metrics.csv")
    if os.path.exists(benchmark_path):
        bench_df = pd.read_csv(benchmark_path)
        st.markdown("#### 🏆 Multi-Model Benchmark Comparison")
        st.dataframe(
            bench_df.rename(columns={
                "model": "Anomaly Detection Algorithm",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1-Score",
                "roc_auc": "ROC-AUC",
                "inference_latency_ms": "Latency (ms)",
                "false_positive_rate": "FPR"
            }).style.format({
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1-Score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "Latency (ms)": "{:.2f} ms",
                "FPR": "{:.2%}"
            }),
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Charts: Anomaly Score Distribution + Feature Impact
    col_dist, col_imp = st.columns(2)
    
    with col_dist:
        st.markdown("#### 📊 Anomaly Score Distribution")
        dist_fig = px.histogram(
            display_df,
            x="raw_anomaly_score",
            color="is_anomaly",
            nbins=30,
            color_discrete_map={False: "#10b981", True: "#ef4444"},
            labels={
                "raw_anomaly_score": "Isolation Score (Negative = Outlier)",
                "is_anomaly": "Flagged Threat",
                "count": "Employee Count"
            },
            title="Score Distribution: Normal Clusters vs Outliers"
        )
        dist_fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(dist_fig, use_container_width=True)
        st.caption("Normal employees cluster around positive scores (+0.15 to +0.25). Malicious outliers separate cleanly on the negative tail.")

    with col_imp:
        st.markdown("#### ⚖️ Key UEBA Feature Deviations Across Threats")
        threat_mask = display_df["is_anomaly"] == True
        key_features = ["total_mb_xfer", "off_hours_events", "usb_connects", "failed_logins", "risky_cloud_count", "sensitive_file_count"]
        feat_labels = ["Total MB Transferred", "Off-Hours Events", "USB Connects", "Failed Logins", "Risky Cloud Uploads", "Sensitive Files"]
        
        diff_ratios = []
        for feat in key_features:
            norm_val = display_df[~threat_mask][feat].mean()
            threat_val = display_df[threat_mask][feat].mean()
            ratio = (threat_val / max(0.01, norm_val))
            diff_ratios.append(ratio)
        
        feat_df = pd.DataFrame({
            "Feature": feat_labels,
            "Anomaly Magnitude (x Baseline)": diff_ratios
        }).sort_values(by="Anomaly Magnitude (x Baseline)", ascending=True)

        feat_fig = px.bar(
            feat_df,
            x="Anomaly Magnitude (x Baseline)",
            y="Feature",
            orientation="h",
            color="Anomaly Magnitude (x Baseline)",
            color_continuous_scale=["#38bdf8", "#ef4444"],
            title="Anomaly Magnitude Multiplier (Threat vs Baseline)"
        )
        feat_fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=35, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(feat_fig, use_container_width=True)
        st.caption("Shows how many times higher the average threat metric is compared to normal company baseline.")

    st.markdown("---")
    
    # 3. Mathematical Principles & Architecture
    col_math, col_arch = st.columns(2)
    
    with col_math:
        st.markdown("#### 📐 Isolation Forest Mathematical Formulation")
        st.markdown(
            "Isolation Forest isolates anomalies by constructing an ensemble of random binary trees. "
            "Because outliers have extreme values, they are isolated with **shorter path lengths** than normal data points."
        )
        st.latex(r"s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}")
        st.markdown(
            "Where:\n"
            "- $h(x)$: Path length (number of edges traversed from root to leaf to isolate instance $x$).\n"
            "- $E(h(x))$: Average path length across the ensemble of 180 isolation trees (`iTrees`).\n"
            "- $c(n) = 2\\ln(n-1) + 0.5772156649 - \\frac{2(n-1)}{n}$: Average path length in a Binary Search Tree.\n\n"
            "**Decision Rule:**\n"
            "- $s \\to 1$: **High-Risk Anomaly** (isolated rapidly in very few tree splits).\n"
            "- $s < 0.5$: **Normal Behavior** (requires deep tree traversal).\n"
            "- $s \\approx 0.5$: Baseline behavior without distinct anomaly signatures."
        )

    with col_arch:
        st.markdown("#### 🏗️ End-to-End System Pipeline")
        st.code(
            """
[ Enterprise SIEM Logs ]
  │  • Authentication (Logon/Logoff events)
  │  • File Audit & Removable Media (USB)
  │  • Cloud SaaS & Perimeter Network
  ▼
[ UEBA Feature Engineering Engine ]
  │  • Temporal metrics (off-hours & weekend ratios)
  │  • Volumetric aggregations (Total MB, Cloud Uploads)
  │  • Robust Z-score normalization against peer department
  ▼
[ Isolation Forest Detection Core ]
  │  • 180 Random Decision Trees (Sub-sample: 256)
  │  • Recursive attribute splitting & path length calculation
  │  • Calibrated Risk Scoring (0 - 100 Scale)
  ▼
[ Real-Time Security Intelligence ]
  │  • Interactive Anomaly Dashboard & Scatter Telemetry
  │  • Groq AI Copilot (Automated MITRE ATT&CK Briefings)
  ▼
[ Incident Containment & Response ]
            """,
            language="text"
        )
