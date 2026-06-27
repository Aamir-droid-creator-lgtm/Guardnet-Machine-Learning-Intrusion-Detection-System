"""
GuardNet - Threat Detector (genuine inference)
==============================================
Loads the trained Hybrid CNN-LSTM model + fitted preprocessor and scores
NSL-KDD-format network records, producing real model probabilities.

No label leakage, no hardcoded entropy rules, no random threat scores -
every prediction comes from the model.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "backend", "models", "ransomware_lstm.h5")
PREP_PATH = os.path.join(BASE, "backend", "models", "preprocessor.joblib")

# NSL-KDD schema (must match model_trainer.py)
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
# Must match SKEWED in model_trainer.py (same log1p transform at inference time).
SKEWED = [
    "duration", "src_bytes", "dst_bytes", "hot", "num_compromised", "num_root",
    "num_file_creations", "count", "srv_count", "dst_host_count",
    "dst_host_srv_count",
]


def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)):
        print("[!] Model or preprocessor not found. Run scripts/model_trainer.py first.")
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH)
    prep = joblib.load(PREP_PATH)
    threshold = prep.get("threshold", 0.5)
    return model, prep["scaler"], prep["feature_columns"], threshold


def preprocess(df_raw, scaler, feature_columns):
    """Transform raw NSL-KDD records into the exact feature space the model
    was trained on (same one-hot columns, same scaling)."""
    X = df_raw.copy()
    for col in ("label", "difficulty"):
        if col in X.columns:
            X = X.drop(columns=[col])
    for col in SKEWED:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))
    X = pd.get_dummies(X, columns=CATEGORICAL)
    # Align columns to training schema: add missing, drop unseen, fix order.
    X = X.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(X.values.astype("float32"))


def run(input_path=None, threshold=None, limit=20):
    model, scaler, feature_columns, saved_threshold = load_artifacts()
    if threshold is None:
        threshold = saved_threshold

    if input_path is None:
        input_path = os.path.join(BASE, "data", "NSL-KDD Dataset", "KDDTest+.txt")
    print(f"[*] GuardNet AI Engine scoring: {input_path}")

    df_raw = pd.read_csv(input_path, header=None, names=COLUMNS)
    if limit:
        df_raw = df_raw.head(limit)

    X = preprocess(df_raw, scaler, feature_columns)
    probs = model.predict(X, verbose=0).flatten()

    out = pd.DataFrame({
        "protocol": df_raw["protocol_type"].values,
        "service": df_raw["service"].values,
        "threat_score": np.round(probs * 100, 1),
        "prediction": np.where(probs >= threshold, "ATTACK", "BENIGN"),
    })
    if "label" in df_raw.columns:
        out["actual"] = np.where(df_raw["label"].values == "normal", "BENIGN", "ATTACK")

    print(out.to_string(index=False))
    print(f"\n[*] Scored {len(out)} records | Threats flagged: {(probs >= threshold).sum()}")
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run(input_path=path)
