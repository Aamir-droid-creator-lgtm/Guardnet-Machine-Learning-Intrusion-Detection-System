import os
import sys
import subprocess
import pandas as pd
import psutil
import numpy as np
import datetime
import time
import threading
from flask import Flask, render_template, jsonify

# Reuse the genuine model + preprocessing pipeline from the detector.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from threat_detector import load_artifacts, preprocess, COLUMNS  # noqa: E402

# --- 🎯 DYNAMIC PATH LOGIC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
# Data folder ko automate kiya hai
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "predictions.csv")

os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, 
            template_folder=TEMPLATE_DIR, 
            static_folder=STATIC_DIR)

# Global flag to control the live detection loop
sim_running = False

# Stream source: real NSL-KDD test records scored by the trained model.
STREAM_PATH = os.path.join(DATA_DIR, "NSL-KDD Dataset", "KDDTest+.txt")

# Load the genuine model + preprocessor once at import time.
try:
    _MODEL, _SCALER, _FEATURE_COLS, _THRESHOLD = load_artifacts()
    print(f"[*] GuardNet model loaded | decision threshold = {_THRESHOLD}")
except SystemExit:
    _MODEL = None
    print("[!] Model not found - run scripts/model_trainer.py before live capture.")


def run_live_detection():
    """Stream real NSL-KDD records through the trained Hybrid CNN-LSTM and
    publish genuine model predictions to the dashboard CSV."""
    global sim_running
    if _MODEL is None:
        print("[!] No model loaded; cannot start live detection.")
        sim_running = False
        return

    print("[!] SOC Terminal: Starting GuardNet live detection (real model)...")
    source = pd.read_csv(STREAM_PATH, header=None, names=COLUMNS)
    source = source.sample(frac=1, random_state=None).reset_index(drop=True)  # shuffle order
    idx = 0

    while sim_running:
        try:
            record = source.iloc[[idx % len(source)]].reset_index(drop=True)
            idx += 1

            # Genuine model inference on this record.
            X = preprocess(record, _SCALER, _FEATURE_COLS)
            prob = float(_MODEL.predict(X, verbose=0).flatten()[0])
            is_atk = int(prob >= _THRESHOLD)
            actual_atk = int(record["label"].iloc[0] != "normal")

            new_entry = {
                'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
                # NSL-KDD has no IP; synthesize a stable-looking source address.
                'src_ip': f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(2,254)}",
                'protocol': str(record["protocol_type"].iloc[0]).upper(),
                'service': str(record["service"].iloc[0]),
                'entropy': round(prob * 100, 1),   # displayed metric = model threat %
                'threat_score': round(prob * 100, 1),
                'is_attack': is_atk,
                'actual_attack': actual_atk,
                'lat': np.random.uniform(-30, 60),
                'lng': np.random.uniform(-100, 130)
            }

            df_new = pd.DataFrame([new_entry])
            if not os.path.exists(CSV_PATH):
                df_new.to_csv(CSV_PATH, index=False)
            else:
                try:
                    df_old = pd.read_csv(CSV_PATH)
                    df_final = pd.concat([df_old, df_new], ignore_index=True).tail(100)
                    df_final.to_csv(CSV_PATH, index=False)
                except Exception:
                    df_new.to_csv(CSV_PATH, index=False)

            time.sleep(1.5)
        except Exception as e:
            print(f"Detection Error: {e}")
            time.sleep(2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_capture')
def start():
    global sim_running
    try:
        # Stop existing and clear old logs
        sim_running = False
        time.sleep(0.5)
        
        # Fresh start: launch real model-driven detection loop
        sim_running = True
        threading.Thread(target=run_live_detection, daemon=True).start()

        return jsonify({"status": "success", "message": "Neural SOC Capture Online"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/restart')
def restart_system():
    global sim_running
    sim_running = False
    time.sleep(0.5)
    try:
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        return jsonify({"status": "success", "message": "Neural Buffer Cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stats')
def stats():
    # Empty check
    if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size < 10:
        return jsonify({
            "total": 0, "anomalies": 0, "intensity": 0, 
            "cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent, "logs": []
        })

    try:
        # Pandas reading with try-except for file access conflicts
        df = pd.read_csv(CSV_PATH)
        df['entropy'] = pd.to_numeric(df['entropy'], errors='coerce').fillna(0)
        
        total = len(df)
        anomalies = int(df['is_attack'].sum())
        
        # Latest model threat % (chart) and rolling average (stat card)
        intensity = round(float(df['entropy'].iloc[-1]), 2)
        avg_intensity = round(float(df['entropy'].mean()), 2)

        # Logs for table (Reversed for newest on top)
        logs = df.tail(15).to_dict('records')[::-1]

        return jsonify({
            "total": total,
            "anomalies": anomalies,
            "intensity": intensity,
            "avg_intensity": avg_intensity,
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "logs": logs
        })
    except Exception as e:
        return jsonify({"error": "Data stream busy"}), 200

if __name__ == "__main__":
    print(f"[*] GuardNet Server Online | Port: 5005")
    app.run(debug=True, port=5005, use_reloader=False)