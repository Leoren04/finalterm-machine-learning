# 🎵 UAS Machine Learning — Task 2: Song Release Year Regression

> **UAS Machine Learning — Individual Task | Hands-On End-to-End Models**

---

## 📋 Overview

Pipeline **regresi end-to-end** untuk memprediksi **tahun rilis lagu** dari fitur audio numerik (Million Song Dataset). Mengimplementasikan dua pendekatan:

- **Baseline ML**: LightGBM Regressor
- **Deep Learning**: Multi-Layer Perceptron (MLP) Regressor dengan PyTorch

### Task Summary

| Item | Detail |
|------|--------|
| **Problem** | Regression (predict release year — continuous value) |
| **Dataset** | `midterm-regresi-dataset.csv` (~500K songs, 90 features) |
| **ML Model** | LightGBM Regressor (Baseline) |
| **DL Model** | MLP Neural Network (PyTorch) |
| **Metrics** | RMSE, MAE, R² |

---

## 🏗️ Repository Structure

```
ML-Task2-SongRegression/
├── README.md
├── requirements.txt
├── notebooks/
│   └── song_year_regression.ipynb       # Main notebook
└── reports/
    ├── target_distribution.png
    ├── mlp_training_curves.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    └── best_mlp.pth                     # Best MLP model weights
```

---

## 🚀 How to Run

### Google Colab (Recommended)
1. Mount Google Drive dan upload `midterm-regresi-dataset.csv`
2. Uncomment Google Drive mounting cell
3. Enable GPU Runtime (T4) untuk training MLP yang lebih cepat
4. Run all cells

### Local
```bash
pip install -r requirements.txt
jupyter notebook notebooks/song_year_regression.ipynb
```

---

## 📊 Dataset Structure

Dataset tidak memiliki header:
- **Column 0**: `year` — tahun rilis lagu (target)
- **Columns 1–90**: `feature_1` s/d `feature_90` — fitur timbre audio

> Contoh: `2001, 49.94357, 21.47114, 73.0775, ...`

---

## 🧠 MLP Architecture

```
Input (90 features)
    → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(64)  → BatchNorm → ReLU → Dropout(0.1)
    → Linear(32)  → ReLU
    → Linear(1)   → Predicted Year (denormalized)
```

**Key training techniques:**
- Target normalization (z-score) untuk stabilitas training
- ReduceLROnPlateau scheduler
- Gradient clipping (max norm = 1.0)
- Best model checkpoint saving

---

## 📈 Results

| Model | RMSE (years) | MAE (years) | R² |
|-------|-------------|-------------|-----|
| LightGBM |  8.9169  | 6.2181 | 0.3401 |
| **MLP (DL)** | 8.7274 | 6.0240 | 0.3679 |

---

## 👤 Identifikasi

| Item | Info |
|------|------|
| **Nama** | Rakha Primindra Danuatmaja |
| **NIM** | 1103223001 |
| **Kelas** | TK-46-GAB (Machine Learning) |
| **Mata Kuliah** | Machine Learning — UAS |
