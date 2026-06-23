"""
GuardNet - Hybrid CNN-LSTM Trainer (NSL-KDD)
=============================================
Trains a genuine Hybrid CNN-LSTM intrusion-detection model on the full
NSL-KDD feature set (no label leakage, no synthetic shortcut features).

Pipeline:
    KDDTrain+  -> clean + encode + scale -> Conv1D (spatial) -> LSTM (temporal)
               -> Dense -> sigmoid (BENIGN vs ATTACK)
Artifacts saved:
    backend/models/ransomware_lstm.h5   (trained Keras model)
    backend/models/preprocessor.joblib  (scaler + feature columns)
Evaluation on the held-out KDDTest+ split is reported at the end.
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, Dropout, LSTM, Reshape, Conv1D, MaxPooling1D, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report,
)

# --- Paths -----------------------------------------------------------------
SCRIPT_LOCATION = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_LOCATION)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "NSL-KDD Dataset")
TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")
MODEL_OUT = os.path.join(PROJECT_ROOT, "backend", "models", "ransomware_lstm.h5")
PREP_OUT = os.path.join(PROJECT_ROOT, "backend", "models", "preprocessor.joblib")

# --- NSL-KDD column schema (41 features + label + difficulty) ---------------
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


def load_dataset(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df.drop(columns=["difficulty"])  # NSL-KDD difficulty score, not a feature
    # Binary target: anything that is not 'normal' is an attack.
    y = (df["label"] != "normal").astype(int).values
    X = df.drop(columns=["label"])
    return X, y


def build_model(n_features):
    """Genuine Hybrid CNN-LSTM: Conv1D extracts local feature patterns,
    LSTM models sequential dependencies across the feature vector."""
    model = Sequential([
        Input(shape=(n_features,)),
        Reshape((n_features, 1)),

        # CNN block - spatial feature extraction
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),

        # LSTM block - temporal / sequential pattern analysis
        LSTM(64, return_sequences=False),

        # Dense classification head
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),  # BENIGN (0) vs ATTACK (1)
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
    )
    return model


def initiate_model_training():
    print("--- [Status] GuardNet Hybrid CNN-LSTM Training (NSL-KDD) ---")
    for p in (TRAIN_PATH, TEST_PATH):
        if not os.path.exists(p):
            print(f"[-] Error: Dataset missing at {p}")
            return

    # 1. Load train + test
    X_train_raw, y_train = load_dataset(TRAIN_PATH)
    X_test_raw, y_test = load_dataset(TEST_PATH)

    # 2. One-hot encode categoricals. Fit on the union so train/test columns
    #    align (KDDTest+ contains service values unseen in training).
    combined = pd.concat([X_train_raw, X_test_raw], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=CATEGORICAL)
    X_train_df = combined.xs("train")
    X_test_df = combined.xs("test")
    feature_columns = list(X_train_df.columns)

    # 3. Scale all features to comparable ranges
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values.astype("float32"))
    X_test = scaler.transform(X_test_df.values.astype("float32"))

    print(f"[*] Train: {X_train.shape}  Test: {X_test.shape}  Features: {len(feature_columns)}")
    print(f"[*] Train attack ratio: {y_train.mean():.2%} | Test attack ratio: {y_test.mean():.2%}")

    # 4. Hold out a validation split from training data (used to choose the
    #    decision threshold WITHOUT ever touching the test set).
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # Class weighting: counter the train/test attack-ratio shift and push the
    # model toward higher attack recall (the project's stated objective).
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = {int(c): w for c, w in zip(classes, weights)}
    print(f"[*] Class weights: {class_weight}")

    # 5. Build + train
    model = build_model(X_train.shape[1])
    model.summary()
    early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[early],
        verbose=1,
    )

    # 6. Choose the decision threshold on the VALIDATION set (no test leakage),
    #    maximizing F1 so recall improves without collapsing precision.
    threshold = select_threshold(model, X_val, y_val)
    print(f"[*] Selected decision threshold (from validation): {threshold:.2f}")

    # 7. Honest evaluation on the held-out KDDTest+ split
    evaluate(model, X_test, y_test, threshold)

    # 8. Persist model + preprocessor (threshold travels with the artifacts)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    joblib.dump(
        {"scaler": scaler, "feature_columns": feature_columns, "threshold": threshold},
        PREP_OUT,
    )
    print(f"\nSuccess: Model saved at {MODEL_OUT}")
    print(f"Success: Preprocessor saved at {PREP_OUT}")


def select_threshold(model, X_val, y_val):
    """Pick the probability cutoff that maximizes F1 on the validation set."""
    probs = model.predict(X_val, verbose=0).flatten()
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_val, (probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)


def evaluate(model, X_test, y_test, threshold=0.5):
    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)
    print("\n=========== Evaluation on KDDTest+ (held-out) ===========")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall   : {recall_score(y_test, preds):.4f}")
    print(f"F1-score : {f1_score(y_test, preds):.4f}")
    print("\nConfusion Matrix [rows=actual, cols=pred] (0=BENIGN, 1=ATTACK):")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["BENIGN", "ATTACK"]))
    print("========================================================")


if __name__ == "__main__":
    initiate_model_training()
