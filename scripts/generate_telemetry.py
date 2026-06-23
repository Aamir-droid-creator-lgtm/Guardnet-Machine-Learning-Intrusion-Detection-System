import pandas as pd, numpy as np, os, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(BASE, "data", "predictions.csv")

def run():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if os.path.exists(LOG): os.remove(LOG) # Fresh start
    
    start = time.time()
    p_id = 0
    print("🚀 Injecting High-Fidelity Streams...")
    
    while (time.time() - start) < 10:
        p_id += 1
        # 4th and 8th packet ko suspicious banao
        is_atk = (p_id == 4 or p_id == 8)
        
        data = {
            'timestamp': [time.strftime('%H:%M:%S')],
            'src_ip': ["10.0.0.66" if is_atk else f"192.168.1.{np.random.randint(2, 254)}"],
            'protocol': ["TLS 1.3 (Encrypted)" if is_atk else np.random.choice(['TCP', 'UDP', 'HTTP', 'DNS'])],
            'length': [np.random.randint(1400, 1500) if is_atk else np.random.randint(64, 512)],
            'ttl': [128 if is_atk else 64],
            'entropy': [8.92 if is_atk else np.random.uniform(2.0, 4.5)]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(LOG, mode='a', index=False, header=not os.path.exists(LOG))
        print(f"✔️ Injected packet {p_id}")
        time.sleep(1)

if __name__ == "__main__":
    run()