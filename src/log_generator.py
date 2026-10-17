"""
log_generator.py
================
Enterprise User Activity and Access Log Generator.
Generates realistic SOC SIEM / CERT-style audit logs with both normal operational
traffic and multi-stage insider threat attack vectors.
"""

import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd


# Set deterministic random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEPARTMENTS = ["Engineering", "Finance", "Human_Resources", "Sales", "IT_Admin", "Legal"]
DEVICE_TYPES = ["PC-Workstation", "Laptop-Remote", "Server-DC", "Cloud-VM"]
ACTION_TYPES = ["LOGON", "LOGOFF", "FILE_ACCESS", "FILE_COPY", "DEVICE_CONNECT", "HTTP_REQUEST", "EMAIL_SENT"]
SENSITIVITY_LEVELS = ["UNCLASSIFIED", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]

SENSITIVE_PATHS = [
    "/confidential/q4_merger_acquisition_draft.pdf",
    "/source_code/core_quantum_algos.tar.gz",
    "/credentials/master_prod_keys.pem",
    "/hr/salaries_and_severance_2026.xlsx",
    "/finance/customer_credit_vault_dump.sql",
    "/legal/pending_patent_disclosures.docx"
]

NORMAL_PATHS = [
    "/docs/general_handbook.pdf",
    "/shared/team_standup_notes.docx",
    "/projects/daily_task_tracker.xlsx",
    "/wiki/engineering_standards.md",
    "/marketing/q3_social_assets.png",
    "/public/press_release_template.txt"
]

CLOUD_DESTINATIONS_APPROVED = ["sharepoint.enterprise.corp", "github.enterprise.corp", "confluence.enterprise.corp"]
CLOUD_DESTINATIONS_RISKY = ["mega.nz/upload", "dropbox.com/api/v2/files", "wetransfer.com/send", "pastebin.com/post"]


def generate_enterprise_logs(
    num_users: int = 120,
    days: int = 14,
    threat_ratio: float = 0.05,
    start_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic enterprise user activity logs.
    
    Parameters:
        num_users: Total number of enterprise employees.
        days: Number of days to simulate.
        threat_ratio: Fraction of users exhibiting anomalous insider threat patterns.
        start_date: Starting datetime for logs.
        
    Returns:
        (raw_logs_df, user_ground_truth_df)
    """
    if start_date is None:
        start_date = datetime(2026, 9, 1, 8, 0, 0)
        
    num_threat_users = max(2, int(num_users * threat_ratio))
    threat_user_ids = {f"USR_{i:04d}" for i in range(1, num_threat_users + 1)}
    
    users = []
    for i in range(1, num_users + 1):
        uid = f"USR_{i:04d}"
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        is_threat = uid in threat_user_ids
        threat_scenario = None
        if is_threat:
            scenarios = [
                "DATA_EXFILTRATION_DEPARTING",
                "PRIVILEGE_ESCALATION_ROGUE_ADMIN",
                "IP_HARVESTING_SABOTAGE",
                "OFF_HOURS_COVERT_LEAK"
            ]
            threat_scenario = scenarios[(i - 1) % len(scenarios)]
        
        users.append({
            "user_id": uid,
            "department": dept,
            "primary_device": f"DEV-{dept[:3].upper()}-{i:03d}",
            "is_threat": is_threat,
            "threat_scenario": threat_scenario or "NORMAL"
        })
        
    users_df = pd.DataFrame(users)
    
    all_events: List[Dict] = []
    log_id_counter = 1
    
    for day_idx in range(days):
        current_day = start_date + timedelta(days=day_idx)
        is_weekend = current_day.weekday() >= 5
        
        for user in users:
            uid = user["user_id"]
            dept = user["department"]
            is_threat = user["is_threat"]
            scenario = user["threat_scenario"]
            dev = user["primary_device"]
            
            # Weekend filter for normal users
            if is_weekend and not is_threat:
                # Normal users rarely work on weekends (5% probability, 1-3 actions)
                if random.random() > 0.05:
                    continue
                num_normal_actions = random.randint(1, 4)
            else:
                num_normal_actions = random.randint(25, 60)
                
            if not is_threat:
                # ----------------- NORMAL OPERATIONAL BEHAVIOR -----------------
                # Regular 9-to-5 working hours with slight Gaussian jitter
                login_hour = int(np.clip(np.random.normal(9, 0.75), 7, 11))
                login_min = random.randint(0, 59)
                login_time = current_day.replace(hour=login_hour, minute=login_min, second=random.randint(0, 59))
                
                # Successful logon
                all_events.append({
                    "log_id": f"LOG_{log_id_counter:08d}",
                    "timestamp": login_time,
                    "user_id": uid,
                    "department": dept,
                    "device_id": dev,
                    "action_type": "LOGON",
                    "resource": "Local_Console",
                    "sensitivity": "UNCLASSIFIED",
                    "bytes_transferred": 0,
                    "status": "SUCCESS",
                    "destination": "Internal_Network",
                    "is_off_hours": login_time.hour < 8 or login_time.hour >= 18 or is_weekend
                })
                log_id_counter += 1
                
                # Occasional rare failed login (e.g. mistyped password ~ 1% chance)
                if random.random() < 0.03:
                    all_events.append({
                        "log_id": f"LOG_{log_id_counter:08d}",
                        "timestamp": login_time - timedelta(minutes=1),
                        "user_id": uid,
                        "department": dept,
                        "device_id": dev,
                        "action_type": "LOGON",
                        "resource": "Local_Console",
                        "sensitivity": "UNCLASSIFIED",
                        "bytes_transferred": 0,
                        "status": "FAILURE",
                        "destination": "Internal_Network",
                        "is_off_hours": login_time.hour < 8 or login_time.hour >= 18 or is_weekend
                    })
                    log_id_counter += 1
                
                # Normal daytime events
                current_time = login_time
                for _ in range(num_normal_actions):
                    gap = random.randint(3, 20)
                    current_time += timedelta(minutes=gap)
                    if current_time.hour >= 18:
                        break
                        
                    action = random.choices(
                        ["FILE_ACCESS", "HTTP_REQUEST", "EMAIL_SENT"],
                        weights=[0.55, 0.35, 0.10]
                    )[0]
                    
                    if action == "FILE_ACCESS":
                        res = random.choice(NORMAL_PATHS)
                        sens = "INTERNAL" if random.random() < 0.3 else "UNCLASSIFIED"
                        bytes_xfer = random.randint(5_000, 250_000)
                        dest = "Internal_File_Share"
                    elif action == "HTTP_REQUEST":
                        res = random.choice(CLOUD_DESTINATIONS_APPROVED)
                        sens = "UNCLASSIFIED"
                        bytes_xfer = random.randint(10_000, 1_000_000)
                        dest = res
                    else: # EMAIL_SENT
                        res = f"colleague_{random.randint(1, 50)}@enterprise.corp"
                        sens = "INTERNAL"
                        bytes_xfer = random.randint(20_000, 500_000)
                        dest = "Exchange_Internal"
                        
                    all_events.append({
                        "log_id": f"LOG_{log_id_counter:08d}",
                        "timestamp": current_time,
                        "user_id": uid,
                        "department": dept,
                        "device_id": dev,
                        "action_type": action,
                        "resource": res,
                        "sensitivity": sens,
                        "bytes_transferred": bytes_xfer,
                        "status": "SUCCESS",
                        "destination": dest,
                        "is_off_hours": False
                    })
                    log_id_counter += 1
                    
                # End of day logoff
                all_events.append({
                    "log_id": f"LOG_{log_id_counter:08d}",
                    "timestamp": current_time + timedelta(minutes=10),
                    "user_id": uid,
                    "department": dept,
                    "device_id": dev,
                    "action_type": "LOGOFF",
                    "resource": "Local_Console",
                    "sensitivity": "UNCLASSIFIED",
                    "bytes_transferred": 0,
                    "status": "SUCCESS",
                    "destination": "Internal_Network",
                    "is_off_hours": False
                })
                log_id_counter += 1
                
            else:
                # ----------------- ANOMALOUS INSIDER THREAT BEHAVIOR -----------------
                # Threat behavior kicks in especially heavily in the second half of the timeline
                is_attack_day = (day_idx >= days // 2) or (day_idx % 2 == 1)
                
                if not is_attack_day:
                    # Act somewhat normal before launching attack
                    login_time = current_day.replace(hour=9, minute=random.randint(0, 30))
                    all_events.append({
                        "log_id": f"LOG_{log_id_counter:08d}",
                        "timestamp": login_time,
                        "user_id": uid,
                        "department": dept,
                        "device_id": dev,
                        "action_type": "LOGON",
                        "resource": "Local_Console",
                        "sensitivity": "UNCLASSIFIED",
                        "bytes_transferred": 0,
                        "status": "SUCCESS",
                        "destination": "Internal_Network",
                        "is_off_hours": is_weekend
                    })
                    log_id_counter += 1
                    continue
                    
                if scenario == "DATA_EXFILTRATION_DEPARTING":
                    # Attack Pattern: 01:00 AM - 04:00 AM, USB insertion, massive confidential file copy, cloud upload to Mega/Dropbox
                    attack_time = current_day.replace(hour=random.choice([1, 2, 3, 23]), minute=random.randint(5, 45))
                    
                    # Off-hours logon
                    all_events.append({
                        "log_id": f"LOG_{log_id_counter:08d}",
                        "timestamp": attack_time,
                        "user_id": uid,
                        "department": dept,
                        "device_id": dev,
                        "action_type": "LOGON",
                        "resource": "Remote_VPN",
                        "sensitivity": "INTERNAL",
                        "bytes_transferred": 0,
                        "status": "SUCCESS",
                        "destination": "Corporate_VPN_Gateway",
                        "is_off_hours": True
                    })
                    log_id_counter += 1
                    
                    # Connect USB device
                    attack_time += timedelta(minutes=2)
                    all_events.append({
                        "log_id": f"LOG_{log_id_counter:08d}",
                        "timestamp": attack_time,
                        "user_id": uid,
                        "department": dept,
                        "device_id": dev,
                        "action_type": "DEVICE_CONNECT",
                        "resource": "USB_SANDISK_ULTRA_64GB",
                        "sensitivity": "RESTRICTED",
                        "bytes_transferred": 0,
                        "status": "SUCCESS",
                        "destination": "USB_Removable_Bus",
                        "is_off_hours": True
                    })
                    log_id_counter += 1
                    
                    # Rapid access and copy of sensitive files
                    for path in SENSITIVE_PATHS:
                        attack_time += timedelta(seconds=random.randint(15, 60))
                        bytes_copied = random.randint(50_000_000, 450_000_000) # 50 MB - 450 MB each
                        all_events.append({
                            "log_id": f"LOG_{log_id_counter:08d}",
                            "timestamp": attack_time,
                            "user_id": uid,
                            "department": dept,
                            "device_id": dev,
                            "action_type": "FILE_COPY",
                            "resource": path,
                            "sensitivity": "RESTRICTED",
                            "bytes_transferred": bytes_copied,
                            "status": "SUCCESS",
                            "destination": "E:/Backups/Exfil",
                            "is_off_hours": True
                        })
                        log_id_counter += 1
                        
                    # Cloud upload to unauthorized storage
                    attack_time += timedelta(minutes=5)
                    all_events.append({
                        "log_id": f"LOG_{log_id_counter:08d}",
                        "timestamp": attack_time,
                        "user_id": uid,
                        "department": dept,
                        "device_id": dev,
                        "action_type": "HTTP_REQUEST",
                        "resource": random.choice(CLOUD_DESTINATIONS_RISKY),
                        "sensitivity": "CONFIDENTIAL",
                        "bytes_transferred": random.randint(800_000_000, 2_500_000_000), # ~1 - 2.5 GB exfil
                        "status": "SUCCESS",
                        "destination": "External_Public_Cloud",
                        "is_off_hours": True
                    })
                    log_id_counter += 1

                elif scenario == "PRIVILEGE_ESCALATION_ROGUE_ADMIN":
                    # Attack Pattern: Repeated brute-force / lateral logon failures across multiple servers, followed by DC access
                    attack_time = current_day.replace(hour=random.choice([0, 22, 23]), minute=random.randint(10, 50))
                    
                    target_servers = [f"SRV-DC-{i:02d}" for i in range(1, 6)]
                    for srv in target_servers:
                        for _ in range(random.randint(3, 7)):
                            attack_time += timedelta(seconds=random.randint(5, 20))
                            all_events.append({
                                "log_id": f"LOG_{log_id_counter:08d}",
                                "timestamp": attack_time,
                                "user_id": uid,
                                "department": dept,
                                "device_id": srv,
                                "action_type": "LOGON",
                                "resource": "Admin_Share_C$",
                                "sensitivity": "RESTRICTED",
                                "bytes_transferred": 0,
                                "status": "FAILURE",
                                "destination": srv,
                                "is_off_hours": True
                            })
                            log_id_counter += 1
                            
                    # Eventually successful privilege bypass
                    attack_time += timedelta(minutes=2)
                    all_events.append({
                        "log_id": f"LOG_{log_id_counter:08d}",
                        "timestamp": attack_time,
                        "user_id": uid,
                        "department": dept,
                        "device_id": "SRV-DC-01",
                        "action_type": "LOGON",
                        "resource": "Domain_Admin_Console",
                        "sensitivity": "RESTRICTED",
                        "bytes_transferred": 0,
                        "status": "SUCCESS",
                        "destination": "SRV-DC-01",
                        "is_off_hours": True
                    })
                    log_id_counter += 1
                    
                    # Dumping credential databases
                    all_events.append({
                        "log_id": f"LOG_{log_id_counter:08d}",
                        "timestamp": attack_time + timedelta(minutes=1),
                        "user_id": uid,
                        "department": dept,
                        "device_id": "SRV-DC-01",
                        "action_type": "FILE_ACCESS",
                        "resource": "/credentials/master_prod_keys.pem",
                        "sensitivity": "RESTRICTED",
                        "bytes_transferred": 50_000_000,
                        "status": "SUCCESS",
                        "destination": "Local_Admin_Buffer",
                        "is_off_hours": True
                    })
                    log_id_counter += 1

                elif scenario == "IP_HARVESTING_SABOTAGE":
                    # Attack Pattern: High-frequency bulk reading of source repositories & confidential files
                    attack_time = current_day.replace(hour=random.choice([19, 20, 21]), minute=10)
                    for path in SENSITIVE_PATHS * 3:
                        attack_time += timedelta(seconds=random.randint(3, 10))
                        all_events.append({
                            "log_id": f"LOG_{log_id_counter:08d}",
                            "timestamp": attack_time,
                            "user_id": uid,
                            "department": dept,
                            "device_id": dev,
                            "action_type": "FILE_ACCESS",
                            "resource": path,
                            "sensitivity": "RESTRICTED",
                            "bytes_transferred": random.randint(15_000_000, 100_000_000),
                            "status": "SUCCESS",
                            "destination": "Local_Disk",
                            "is_off_hours": True
                        })
                        log_id_counter += 1

                elif scenario == "OFF_HOURS_COVERT_LEAK":
                    # Attack Pattern: Weekend/late night emails sent to personal external domains with high attachment sizes
                    attack_time = current_day.replace(hour=random.choice([2, 4, 22]), minute=15)
                    for _ in range(random.randint(4, 9)):
                        attack_time += timedelta(minutes=random.randint(2, 8))
                        all_events.append({
                            "log_id": f"LOG_{log_id_counter:08d}",
                            "timestamp": attack_time,
                            "user_id": uid,
                            "department": dept,
                            "device_id": dev,
                            "action_type": "EMAIL_SENT",
                            "resource": "shadow_ops@protonmail.com",
                            "sensitivity": "CONFIDENTIAL",
                            "bytes_transferred": random.randint(25_000_000, 120_000_000),
                            "status": "SUCCESS",
                            "destination": "External_Mail_Relay",
                            "is_off_hours": True
                        })
                        log_id_counter += 1
                        
    raw_df = pd.DataFrame(all_events)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values(by="timestamp").reset_index(drop=True)
    
    return raw_df, users_df


if __name__ == "__main__":
    out_dir = r"C:\Users\DELL\.gemini\antigravity\scratch\insider_threat_detector\data"
    os.makedirs(out_dir, exist_ok=True)
    print("Generating enterprise user activity logs...")
    raw_logs, user_meta = generate_enterprise_logs(num_users=150, days=14, threat_ratio=0.06)
    
    raw_logs_path = os.path.join(out_dir, "raw_user_activity_logs.csv")
    meta_path = os.path.join(out_dir, "users_metadata.csv")
    
    raw_logs.to_csv(raw_logs_path, index=False)
    user_meta.to_csv(meta_path, index=False)
    
    print(f"Generated {len(raw_logs):,} events across {len(user_meta)} enterprise users.")
    print(f"Threat users: {user_meta['is_threat'].sum()} ({user_meta['is_threat'].mean()*100:.1f}%)")
    print(f"Saved to: {raw_logs_path}")
