"""
investigation.py
================
Security Analyst Insider-Risk Queue & Investigation Management.
Provides triage queue state tracking, forensic dossier generation,
containment playbooks, and analyst reporting.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


STATUS_OPTIONS = [
    "NEW_ALERT",
    "UNDER_INVESTIGATION",
    "ESCALATED_TO_SOC",
    "ESCALATED_LEGAL_HR",
    "RESOLVED_FALSE_POSITIVE",
    "RESOLVED_CONFIRMED_THREAT"
]

PLAYBOOKS = {
    "DATA_EXFILTRATION_DEPARTING": [
        "1. Immediately suspend corporate cloud storage and USB mass-storage write permissions on host.",
        "2. Isolate host endpoint via EDR (prevent further network exfiltration).",
        "3. Cross-reference departing employee list with HR to confirm resignation or performance review status.",
        "4. Contact Cloud Access Security Broker (CASB) team to revoke SaaS API tokens and block destination IPs.",
        "5. Preserve forensic snapshot of endpoint disk and volatile memory for legal chain of custody."
    ],
    "PRIVILEGE_ESCALATION_ROGUE_ADMIN": [
        "1. Temporarily disable compromised Active Directory credential and revoke Kerberos tickets.",
        "2. Rotate all Domain Controller service account keys and production database passwords.",
        "3. Audit security log event IDs 4672 (special privileges assigned) and 4625 (failed logon attempts).",
        "4. Review firewall and internal jump host logs for lateral movement.",
        "5. Escalate immediately to Tier-3 Incident Commander."
    ],
    "IP_HARVESTING_SABOTAGE": [
        "1. Revoke access tokens to Git repositories, code hosting servers, and internal document drives.",
        "2. Clone and inspect local developer git cache for unauthorized branch creation or stashing.",
        "3. Audit file integrity monitoring (FIM) logs on sensitive source repos.",
        "4. Schedule formal interview with department head and employee."
    ],
    "OFF_HOURS_COVERT_LEAK": [
        "1. Quarantine outbound email messages at Secure Email Gateway (SEG).",
        "2. Inspect encrypted attachment hashes and evaluate DLP classification signatures.",
        "3. Block recipient external email address and domain on corporate perimeter.",
        "4. Review employee's VPN connection IP geolocation and multi-factor authentication logs."
    ],
    "GENERAL_ANOMALY": [
        "1. Review recent access requests in Identity & Access Management (IAM) portal.",
        "2. Check with department manager if user was approved for scheduled maintenance or overtime.",
        "3. Monitor user behavior over the next 48-hour observation window."
    ]
}


class InsiderRiskQueueManager:
    """
    Manages the security analyst triage queue, persisting incident statuses,
    analyst notes, and review history.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.state: Dict[str, Dict[str, Any]] = {}
        self.load()
        
    def load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}
        else:
            self.state = {}
            
    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.state, f, indent=2)
            
    def get_user_status(self, user_id: str) -> Dict[str, Any]:
        return self.state.get(user_id, {
            "status": "NEW_ALERT",
            "analyst_notes": "",
            "assigned_analyst": "Unassigned",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    def update_user_status(
        self,
        user_id: str,
        status: str,
        analyst_notes: str,
        assigned_analyst: str = "SOC_Analyst_1"
    ) -> None:
        self.state[user_id] = {
            "status": status,
            "analyst_notes": analyst_notes,
            "assigned_analyst": assigned_analyst,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save()


def _format_dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Safely converts a DataFrame to a Markdown table without requiring third-party libraries."""
    try:
        return df.to_markdown(index=False)
    except (ImportError, Exception):
        headers = list(df.columns)
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        row_lines = []
        for _, row in df.iterrows():
            row_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join([header_line, sep_line] + row_lines) + "\n"


def generate_investigation_report(
    user_id: str,
    user_row: pd.Series,
    explanation: Dict[str, Any],
    queue_info: Dict[str, Any],
    top_events: pd.DataFrame
) -> str:
    """
    Renders a comprehensive, executive-ready Markdown Incident Response Dossier.
    """
    tier = user_row.get("risk_tier", "UNKNOWN")
    score = user_row.get("risk_score", 0.0)
    dept = user_row.get("department", "Unknown")
    
    # Determine probable playbook
    prob_scenario = user_row.get("threat_scenario", "GENERAL_ANOMALY")
    if prob_scenario not in PLAYBOOKS:
        prob_scenario = "GENERAL_ANOMALY"
    playbook_steps = PLAYBOOKS.get(prob_scenario, PLAYBOOKS["GENERAL_ANOMALY"])
    
    report = f"""# 🛡️ SOC INCIDENT INVESTIGATION REPORT: {user_id}
**Classification:** STRICTLY CONFIDENTIAL // INSIDER RISK INVESTIGATION  
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Assigned Analyst:** {queue_info.get('assigned_analyst', 'Unassigned')}  
**Triage Status:** `{queue_info.get('status', 'NEW_ALERT')}`  

---

## 1. Executive Incident Summary
- **Target Employee:** `{user_id}`
- **Department:** {dept}
- **Assigned Risk Tier:** **{tier}**
- **Isolation Forest Calibrated Risk Score:** **{score}/100**
- **Primary Model Decision:** `{'ANOMALY / THREAT' if user_row.get('is_anomaly', False) else 'BENIGN / NORMAL'}`

### Model Threat Narrative
> {explanation.get('analyst_narrative', 'No anomalies detected.')}

---

## 2. Key Behavioral Deviations vs Enterprise Baseline
| Metric / Feature | Observed Value | Dept Baseline | Std Dev Deviation |
| :--- | :--- | :--- | :--- |
"""
    for item in explanation.get("top_contributions", []):
        f = item["feature"]
        obs = item["user_value"]
        base = item["baseline_mean"]
        z = item["robust_zscore"]
        report += f"| `{f}` | **{obs:,.2f}** | {base:,.2f} | **+{z:.1f}σ** |\n"

    report += f"""
---

## 3. Recommended Containment & Remediation Playbook
*Prescribed response workflow for pattern `{prob_scenario}`:*
"""
    for step in playbook_steps:
        report += f"- {step}\n"

    report += f"""
---

## 4. Analyst Case Notes
```text
{queue_info.get('analyst_notes', 'No analyst notes recorded yet.')}
```

---

## 5. Correlated Event Audit Trail (Last {len(top_events)} Critical Events)
"""
    if len(top_events) > 0:
        event_sample = top_events[["timestamp", "action_type", "resource", "sensitivity", "destination", "status", "bytes_transferred"]].copy()
        report += _format_dataframe_to_markdown(event_sample)
    else:
        report += "_No raw events attached._\n"

    return report


def query_ai_analyst(
    user_id: str,
    user_row: pd.Series,
    explanation: Dict[str, Any],
    top_events: pd.DataFrame,
    provider: str = "groq",
    api_key: str = ""
) -> str:
    """
    Queries an LLM (Groq or Google Gemini API) to act as a Senior Cyber Threat Intelligence Analyst,
    producing an advanced MITRE ATT&CK forensic analysis of the flagged employee.
    """
    import requests

    if not api_key:
        return "⚠️ No API Key provided. Please enter a valid Groq API Key or Google Gemini API Key in the sidebar."

    # Construct prompt
    top_factors = explanation.get("top_contributions", [])
    factors_summary = "\n".join([
        f"- {item['feature']}: {item['user_value']} (Peer Baseline: {item['baseline_mean']:.2f}, Standardized Deviation: +{item['robust_zscore']:.1f}σ)"
        for item in top_factors
    ])
    
    events_str = ""
    if len(top_events) > 0:
        events_str = top_events[["timestamp", "action_type", "resource", "destination", "bytes_transferred", "is_off_hours"]].head(15).to_string(index=False)
    
    prompt = f"""You are a Lead Security Operations Center (SOC) Cyber Forensics & Insider Threat Investigator.
Analyze the following anomalous employee activity flagged by our enterprise Isolation Forest anomaly detector.

EMPLOYEE METADATA:
- User ID: {user_id}
- Department: {user_row.get('department', 'Unknown')}
- Calibrated Risk Score: {user_row.get('risk_score', 'N/A')}/100 ({user_row.get('risk_tier', 'CRITICAL')} RISK)
- Threat Signature / Scenario: {user_row.get('threat_scenario', 'Unknown')}

TOP ABNORMAL BEHAVIORAL INDICATORS (Z-SCORE DEVIATIONS):
{factors_summary}

MODEL NARRATIVE:
{explanation.get('analyst_narrative', 'N/A')}

SAMPLE RECENT RAW SIEM AUDIT EVENTS:
{events_str}

TASK:
Provide a concise, professional SOC Incident Briefing with the following exact sections:
1. 🎯 Threat Summary & Intent Assessment (Is this deliberate data theft, compromised credentials/account takeover, or policy negligence?)
2. 🗺️ MITRE ATT&CK Mapping (List applicable techniques, e.g., T1048 Exfiltration Over Alternative Protocol, T1078 Valid Accounts, T1052 Exfiltration Over Physical Medium, T1098, etc.)
3. 🔎 Forensic Evidence Highlights (Highlight specific timestamps, destinations, data volumes, and suspicious commands)
4. 🚨 Immediate Containment Recommendations (Numbered concrete steps for SOC / Incident Commander, e.g., token revocation, EDR network isolation, legal preservation)

Keep the briefing crisp, authoritative, and formatted in clean GitHub markdown."""

    try:
        if provider.lower() == "groq":
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            candidate_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]
            last_err = ""
            for model_name in candidate_models:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are an elite enterprise SOC Incident Commander and Cyber Threat Intelligence expert."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1200
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    last_err = f"Groq API Error ({resp.status_code}): {resp.text}"
            return f"❌ {last_err}"

        elif provider.lower() in ["gemini", "google"]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1200
                }
            }
            resp = requests.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    return candidates[0]["content"]["parts"][0]["text"]
                return "No analysis returned from Gemini."
            else:
                return f"❌ Google Gemini API Error ({resp.status_code}): {resp.text}"
        else:
            return f"Unsupported provider: {provider}"

    except Exception as e:
        return f"❌ Error querying AI Analyst: {str(e)}"

