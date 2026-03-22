# 🎵 Spotify Track Popularity Predictor

> An end-to-end machine learning project exploring what audio features drive track popularity on Spotify — featuring EDA, popularity tier classification, regression modeling, ensemble methods, and 5-fold cross-validation.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=flat)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com)

---

## 📖 Overview

This project uses the **Spotify Tracks Dataset** from Kaggle (~114,000 tracks) to investigate what drives track popularity. It runs two parallel modeling tracks — classification and regression — across an end-to-end ML pipeline:

1. **Data loading & cleaning** — null removal, deduplication
2. **Exploratory Data Analysis** — distributions, correlation heatmap
3. **Feature engineering** — 9 audio features + genre one-hot encoding (top 20 genres)
4. **Popularity binning** — continuous score → Low / Mid / High tiers
5. **Preprocessing** — StandardScaler on numeric features only
6. **Classification** — Random Forest, Gradient Boosting, XGBoost with 5-fold CV
7. **Regression** — Random Forest & XGBoost predicting exact score (R² + RMSE)
8. **Feature importance** — identifying the top drivers of popularity

📊 **[View the live interactive notebook →](https://dchildr23.github.io/Spotify/Spotify%20ML%20Notebook.html)**

---

## 🎯 Key Findings

| Finding | Detail |
|---|---|
| **Energy ↔ Loudness** | Strongest correlation (r ≈ 0.76) — louder tracks feel more energetic |
| **Popularity predictors** | Weak individual audio correlations — genre and combined features matter more |
| **Best classifier** | Random Forest / XGBoost on 3-tier bins — far stronger than exact-score classification |
| **Regression insight** | RMSE and R² give a much more meaningful view of model performance than accuracy alone |
| **Top features** | Genre, acousticness, and instrumentalness tend to rank highest in feature importance |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data manipulation and sampling
- **Matplotlib / Seaborn** — visualizations
- **scikit-learn** — StandardScaler, RandomForest, GradientBoosting, cross_val_score
- **XGBoost** *(optional)* — XGBClassifier, XGBRegressor

## 📂 Project Structure

```
spotify-ml-portfolio/
├── spotify_ml_notebook.html    # Rendered interactive notebook (portfolio view)
├── spotify analysis v2.py      # Random Forest and XGBoost — v2 with all improvements
├── spotify analysis v1.py   # Original v1 script (SVM + Decision Tree)
├── spotify_ml_results.png      # Model results chart
├── feature_importance.png      # Feature importance chart 
├── Spotify Tracks Dataset.csv  # Raw data 
└── README.md
```

## 🔭 Potential Next Steps

- [ ] **Hyperparameter tuning** — GridSearchCV / RandomizedSearchCV on all models
- [ ] **SHAP values** — explainability layer on top of XGBoost predictions
- [ ] **Artist-level features** — follower count, genre diversity as additional signals
- [ ] **Time-based analysis** — does release year affect popularity trends?
- [ ] **Streamlit app** — interactive popularity predictor from audio feature inputs
