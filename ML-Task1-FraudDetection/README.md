# 🔐 UAS Machine Learning — Task 1: End-to-End Fraud Detection Pipeline

> **UAS Machine Learning — Individual Task | Hands-On End-to-End Models**

---

## 📋 Overview

Proyek ini membangun **pipeline deteksi fraud transaksi online end-to-end** menggunakan dataset IEEE-CIS Fraud Detection. Pipeline mengimplementasikan dua pendekatan:

- **Baseline ML**: LightGBM (Gradient Boosting)
- **Deep Learning**: Multi-Layer Perceptron (MLP) dengan PyTorch

### Task Summary

| Item | Detail |
|------|--------|
| **Problem** | Binary Classification (isFraud: 0 or 1) |
| **Dataset** | IEEE-CIS Fraud Detection (`train_transaction.csv`) |
| **ML Model** | LightGBM (Baseline) |
| **DL Model** | MLP Neural Network (PyTorch) |
| **Challenge** | Severe class imbalance (~3.5% fraud rate) |
| **Metrics** | AUC-ROC, F1-Score, Precision, Recall |

---

## 🏗️ Repository Structure

```
ML-Task1-FraudDetection/
├── README.md
├── requirements.txt
├── notebooks/
│   └── fraud_detection_pipeline.ipynb   # Main notebook
└── reports/
    ├── eda_overview.png
    ├── mlp_training_curves.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── lgbm_model.txt                   # Saved LightGBM model
    └── mlp_model.pth                    # Saved MLP weights
```

---

## 🚀 How to Run

### Google Colab (Recommended — GPU for faster training)
1. Mount Google Drive dan upload `train_transaction.csv`
2. Uncomment Google Drive mounting cell
3. Run all cells

### Local
```bash
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection_pipeline.ipynb
```

---

## 📊 Pipeline

1. **EDA** — Class imbalance analysis, TransactionAmt distribution
2. **Preprocessing** — Label encoding kategorical, median imputation
3. **Feature Engineering** — Log transform, decimal part, temporal features (hour, day_of_week)
4. **LightGBM** — scale_pos_weight untuk handle imbalance, early stopping
5. **MLP (PyTorch)** — StandardScaler, BatchNorm, Dropout, weighted BCE loss
6. **Comparison** — ROC curves, bar chart metrics, confusion matrices

---

## 🧠 MLP Architecture

```
Input (N features)
    → Linear(512) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(64)  → ReLU
    → Linear(1)   → Sigmoid → Fraud Probability
```

---

## 📈 Results

| Model | AUC-ROC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| LightGBM | 0.9604 | 0.4930 | 0.3479 | 0.8456 |
| **MLP (DL)** | 0.9186 | 0.3444 | 0.2212 | 0.7769 |

---

## 👤 Identifikasi

| Item | Info |
|------|------|
| **Nama** | Rakha Primindra Danuatmaja |
| **NIM** | 1103223001 |
| **Kelas** | TK-46-GAB (Machine Learning) |
| **Mata Kuliah** | Machine Learning — UAS |
