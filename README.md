# 🎵 Spotify Track Popularity Predictor

> An end-to-end machine learning project exploring what audio features drive track popularity on Spotify — featuring EDA, feature scaling, and a comparison of SVM vs. Decision Tree classifiers.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com)

---

## 📖 Overview

This project uses the **Spotify Tracks Dataset** from Kaggle (~114,000 tracks) to investigate the relationship between a song's audio features and its popularity score. The notebook walks through a complete ML pipeline:

1. **Data loading & cleaning** — null removal, deduplication
2. **Exploratory Data Analysis** — distributions, correlation heatmap
3. **Feature engineering** — selecting key audio features
4. **Preprocessing** — StandardScaler normalization
5. **Modeling** — SVM and Decision Tree classifiers
6. **Evaluation** — accuracy comparison and analysis

📊**[View the live interactive notebook →](https://dchildr23.github.io/Spotify/Spotify%20ML%20Notebook.html)**

---

## 🎯 Key Findings

| Finding | Detail |
|---|---|
| **Energy ↔ Loudness** | Strongest correlation (r ≈ 0.76) — louder tracks feel more energetic |
| **Popularity predictors** | Weak individual correlations — popularity is multi-feature, not single-driver |
| **Best model** | Decision Tree edges SVM (7.1% vs 6.2% exact-match accuracy) |
| **Takeaway** | Exact score prediction is hard — binning into tiers would be more practical |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas** — data manipulation and sampling
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualizations
- **scikit-learn** — StandardScaler, SVM, DecisionTreeClassifier, train_test_split

---
open `spotify_ml_notebook.html` directly in your browser for the pre-rendered version.

---

## 📂 Project Structure

```
spotify-ml-portfolio/
├── spotify_ml_notebook.html   # Rendered interactive notebook (portfolio view)   
├── Popularity Predictor.py        # Python script version
├── Spotify Tracks Dataset.csv # Raw data
└── README.md
```

