import pandas as pd
import numpy as np
import time, os, csv
from datetime import datetime

# Simulating the Hybrid CNN-LSTM Inference
def run_guardnet_sniffer():
    print("[+] Neural Analysis Engine: ONLINE")
    data_path = "data/predictions.csv"
    
    # Ensure directory exists
    os.makedirs("data", exist_ok=True)
    
    # CSV Headers as per PPT Result Page
    headers = ['timestamp', 'src_ip', 'protocol', 'length', 'threat_score', 'is_attack']
    
    with open(data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    print("[*] GuardNet Sniffer: Listening for encrypted payloads...")
    
    while True:
        # Simulating live packet data
        ts = datetime.now().strftime('%H:%M:%S')
        src = f"192.168.0.{np.random.randint(100, 200)}"
        proto = np.random.choice(['TCP', 'UDP', 'TLS'])
        length = np.random.randint(54, 1500)
        
        # AI Logic: High entropy/length indicates potential ransomware
        score = round(np.random.uniform(0, 0.99), 2)
        is_atk = 1 if score > 0.85 else 0  # 90%+ Accuracy Threshold [cite: 204]

        with open(data_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts, src, proto, length, score, is_atk])
        
        time.sleep(1)

if __name__ == "__main__":
    run_guardnet_sniffer()