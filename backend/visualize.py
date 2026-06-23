import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Path Configuration ---
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_FILE = os.path.join(BASE_PATH, "data", "predictions.csv")

def generate_visual_audit():
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"[-] Error: {PREDICTIONS_FILE} not found. Run threat_detector.py first.")
        return

    # Load the latest predictions
    df = pd.read_csv(PREDICTIONS_FILE)

    # Confirming data structure
    print("[*] Accessing Forensic Data...")
    print(df[['length', 'entropy', 'is_ransomware', 'threat_score']].head())

    # Styling
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Chart 1: Distribution of Traffic ---
    counts = df["is_ransomware"].value_counts()
    labels = ['Normal Traffic', 'Ransomware']
    colors = ['#2ecc71', '#e74c3c'] # Green and Red
    
    # Mapping actual counts to labels present in data
    plot_labels = [labels[i] for i in counts.index]
    plot_colors = [colors[i] for i in counts.index]

    ax1.bar(plot_labels, counts.values, color=plot_colors)
    ax1.set_title("Neural Audit: Threat Distribution")
    ax1.set_ylabel("Packet Count")

    # --- Chart 2: Entropy vs Threat Score (Forensic Analysis) ---
    # Ransomware usually has high entropy and high threat score
    sns.scatterplot(x='entropy', y='threat_score', hue='is_ransomware', 
                    data=df, palette={0: '#2ecc71', 1: '#e74c3c'}, ax=ax2)
    ax2.set_title("Forensic Analysis: Entropy vs AI Confidence")
    ax2.axhline(0.5, ls='--', color='black', alpha=0.5) # Threshold line

    plt.tight_layout()
    print("[+] Visual Audit Report Generated.")
    plt.show()

if __name__ == "__main__":
    generate_visual_audit()