"""
GuardNet - Results Visualizer
=============================
Loads the trained model + preprocessor, evaluates on the held-out KDDTest+
split, and renders a polished PNG (metrics panel + confusion matrix) suitable
for a report or LinkedIn post.

Output: backend/models/guardnet_results.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from threat_detector import load_artifacts, preprocess, COLUMNS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE, "data", "NSL-KDD Dataset", "KDDTest+.txt")
OUT_PNG = os.path.join(BASE, "backend", "models", "guardnet_results.png")

# Theme
BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#2f81f7"
DANGER = "#f85149"
OK = "#3fb950"


def main():
    model, scaler, feature_columns, threshold = load_artifacts()
    df = pd.read_csv(TEST_PATH, header=None, names=COLUMNS)
    y_true = (df["label"] != "normal").astype(int).values

    X = preprocess(df, scaler, feature_columns)
    probs = model.predict(X, verbose=0).flatten()
    y_pred = (probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(11, 5.2), facecolor=BG)
    gs = GridSpec(1, 2, width_ratios=[1, 1.05], wspace=0.25)
    fig.suptitle("GuardNet  |  Hybrid CNN-LSTM on NSL-KDD (KDDTest+)",
                 color=FG, fontsize=16, fontweight="bold", y=0.98)

    # --- Left: metric cards ---
    axm = fig.add_subplot(gs[0, 0])
    axm.set_facecolor(BG)
    axm.axis("off")
    metrics = [("Accuracy", acc, ACCENT), ("Precision", prec, OK),
               ("Recall", rec, "#d29922"), ("F1-score", f1, ACCENT)]
    for i, (name, val, color) in enumerate(metrics):
        row, col = divmod(i, 2)
        x, y = 0.04 + col * 0.5, 0.55 - row * 0.42
        card = plt.Rectangle((x, y), 0.44, 0.34, transform=axm.transAxes,
                             facecolor="#161b22", edgecolor=color, linewidth=2,
                             zorder=1)
        axm.add_patch(card)
        axm.text(x + 0.22, y + 0.225, f"{val*100:.1f}%", transform=axm.transAxes,
                 ha="center", va="center", color=color, fontsize=24,
                 fontweight="bold")
        axm.text(x + 0.22, y + 0.07, name, transform=axm.transAxes,
                 ha="center", va="center", color=FG, fontsize=11)
    axm.text(0.04, 0.02, f"Held-out test records: {len(y_true):,}   |   "
             f"Decision threshold: {threshold}",
             transform=axm.transAxes, color="#8b949e", fontsize=9)

    # --- Right: confusion matrix ---
    axc = fig.add_subplot(gs[0, 1])
    axc.set_facecolor(BG)
    im = axc.imshow(cm, cmap="Blues")
    labels = ["BENIGN", "ATTACK"]
    axc.set_xticks([0, 1]); axc.set_yticks([0, 1])
    axc.set_xticklabels(labels, color=FG); axc.set_yticklabels(labels, color=FG)
    axc.set_xlabel("Predicted", color=FG, fontsize=11)
    axc.set_ylabel("Actual", color=FG, fontsize=11)
    axc.set_title("Confusion Matrix", color=FG, fontsize=13, pad=10)
    thresh = cm.max() / 2.0
    for r in range(2):
        for c in range(2):
            axc.text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                     color="white" if cm[r, c] > thresh else "#0d1117",
                     fontsize=18, fontweight="bold")
    for spine in axc.spines.values():
        spine.set_color("#30363d")
    axc.tick_params(colors=FG)

    fig.savefig(OUT_PNG, dpi=160, facecolor=BG, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    print(f"Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f}")


if __name__ == "__main__":
    main()
