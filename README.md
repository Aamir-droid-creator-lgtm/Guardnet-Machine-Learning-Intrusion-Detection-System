# 🛡️ GuardNet: ML-Powered Network Intrusion Detection System

**GuardNet** is a Network Intrusion Detection System (NIDS) that classifies network
traffic as **BENIGN** or **ATTACK** using a genuine **Hybrid CNN-LSTM** deep-learning
model trained on the **NSL-KDD** dataset. Predictions are streamed to a live
SOC-style dashboard for monitoring and forensic review.

The model combines a **CNN** stage (spatial feature extraction across the network
feature vector) with an **LSTM** stage (sequential pattern analysis), followed by a
dense classifier — matching the architecture described in the project report.

---

## 📊 Evaluation Results

Evaluated on the held-out **KDDTest+** split (22,544 records the model never saw
during training):

| Metric     | Score  |
| :--------- | :----- |
| Accuracy   | 79.9%  |
| Precision  | 92.6%  |
| Recall     | 70.4%  |
| F1-score   | 80.0%  |

**Confusion matrix** (rows = actual, cols = predicted):

|              | Pred BENIGN | Pred ATTACK |
| :----------- | :---------- | :---------- |
| **BENIGN**   | 8,985       | 726         |
| **ATTACK**   | 3,803       | 9,030       |

> These are real, reproducible numbers — run `python scripts/model_trainer.py` to
> regenerate them. ~80% on KDDTest+ is a strong, honest result: that test set
> deliberately contains attack types absent from training to measure generalization.

---

## 🚀 Key Features

* **Genuine Hybrid CNN-LSTM model** trained on the full NSL-KDD feature set (no
  label leakage, no synthetic shortcut features).
* **Honest evaluation** on a held-out test split with accuracy, precision, recall,
  F1, and confusion matrix.
* **Class weighting + validation-tuned decision threshold** to improve attack
  recall without test-set leakage.
* **Live SOC Dashboard** (Flask + Chart.js) that streams real NSL-KDD records
  through the trained model and shows the model's actual predictions.
* **Forensic logging** of predictions to CSV for incident review.

---

## 🏗️ Methodology / Pipeline

```
Network Traffic (NSL-KDD records)
        ↓
Preprocessing  (clean → one-hot encode protocol/service/flag → standard-scale)
        ↓
CNN Layers     (Conv1D → BatchNorm → MaxPool — spatial feature extraction)
        ↓
LSTM Layer     (temporal / sequential pattern analysis)
        ↓
Dense + Dropout → Sigmoid
        ↓
Output: BENIGN / ATTACK  (+ threat score %)
```

Training uses `KDDTrain+`; evaluation uses the held-out `KDDTest+`. The fitted
scaler, feature columns, and decision threshold are saved alongside the model so
inference reproduces the exact training-time feature space.

---

## 🛠️ Tech Stack

| Component         | Technology                       |
| :---------------- | :--------------------------------|
| **Language**      | Python 3.11                      |
| **Deep Learning** | TensorFlow / Keras               |
| **ML / Data**     | scikit-learn, Pandas, NumPy, joblib |
| **Backend**       | Flask                            |
| **Frontend**      | HTML5, CSS3, Chart.js            |
| **Dataset**       | NSL-KDD (KDDTrain+ / KDDTest+)   |
| **System Metrics**| psutil                           |

---

## 📂 Directory Structure

```text
Cybersecurity Threat Detection System/
├── app.py                       # Flask app + live model-driven detection loop
├── templates/
│   └── index.html               # SOC Dashboard frontend
├── backend/
│   └── models/
│       ├── ransomware_lstm.h5    # Trained CNN-LSTM model
│       └── preprocessor.joblib   # Fitted scaler + feature columns + threshold
├── scripts/
│   ├── model_trainer.py         # Trains + evaluates the CNN-LSTM on NSL-KDD
│   └── threat_detector.py       # Genuine model inference on NSL-KDD records
├── data/
│   └── NSL-KDD Dataset/         # KDDTrain+.txt / KDDTest+.txt
│       predictions.csv          # Live dashboard output (model predictions)
└── README.md
```

---

## 🚦 Execution Guide

**1. Install dependencies**
```bash
pip install tensorflow scikit-learn pandas numpy joblib flask flask-cors psutil
```

**2. Train + evaluate the model**
```bash
python scripts/model_trainer.py
```
Trains the CNN-LSTM on `KDDTrain+`, prints metrics on `KDDTest+`, and saves the
model + preprocessor to `backend/models/`.

**3. (Optional) Batch inference on test records**
```bash
python scripts/threat_detector.py
```
Scores real NSL-KDD records and prints the model's predictions vs. actual labels.

**4. Launch the SOC Dashboard**
```bash
python app.py
```
Open **http://127.0.0.1:5005** and click *Start Capture*. The dashboard streams
real NSL-KDD records through the trained model and displays live predictions.

---

## ⚠️ Scope & Honesty Notes

* The dashboard demonstrates the model on **real NSL-KDD records**, not on live
  raw packet capture. Mapping live Scapy packets to NSL-KDD's 41 flow features is
  a separate, non-trivial feature-engineering problem and is out of scope here.
* Classification is **binary** (BENIGN vs ATTACK); multi-class attack
  categorization is a possible future extension.

---

## 🔒 Disclaimer

This project is intended for **educational and cybersecurity research purposes
only**.

~ Developed by Ritesh Pawar, Mohd Kaif Khan, Aamir Khan
