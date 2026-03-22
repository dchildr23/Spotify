import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor
XGBOOST_AVAILABLE = True

sns.set_theme(style="darkgrid")
print("✅ Libraries loaded\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("Spotify Tracks Dataset.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"Dataset shape after cleaning: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

# 2a. Bin popularity into 3 tiers
#     Low: 0–33  |  Mid: 34–66  |  High: 67–100
def bin_popularity(score):
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Mid"
    else:
        return "High"

df["popularity_tier"] = df["popularity"].apply(bin_popularity)

tier_counts = df["popularity_tier"].value_counts()
print("\nPopularity tier distribution:")
print(tier_counts.to_string())

# 2b. One-hot encode genre (track_genre column)
#     Keep top N genres to avoid too many sparse columns
TOP_N_GENRES = 20
top_genres = df["track_genre"].value_counts().nlargest(TOP_N_GENRES).index
df["track_genre_filtered"] = df["track_genre"].where(
    df["track_genre"].isin(top_genres), other="Other"
)
genre_dummies = pd.get_dummies(df["track_genre_filtered"], prefix="genre")
print(f"\nGenre columns added: {genre_dummies.shape[1]}")

# 2c. Assemble feature matrix
audio_features = [
    "danceability", "energy", "loudness",
    "tempo", "liveness", "valence", "speechiness",
    "acousticness", "instrumentalness"
]

df_features = pd.concat([df[audio_features], genre_dummies], axis=1)

# 2d. Sample 10% for speed (remove or increase frac for full training)
df_sample = df_features.sample(frac=0.1, random_state=42)
y_tier   = df.loc[df_sample.index, "popularity_tier"]   # classification target
y_score  = df.loc[df_sample.index, "popularity"]         # regression target

print(f"\nWorking sample size: {len(df_sample):,} rows")
print(f"Feature count:       {df_sample.shape[1]} (audio + genre dummies)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING — StandardScaler on numeric columns only
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
audio_cols   = [c for c in df_sample.columns if c in audio_features]
genre_cols   = [c for c in df_sample.columns if c.startswith("genre_")]

X_audio_scaled = pd.DataFrame(
    scaler.fit_transform(df_sample[audio_cols]),
    columns=audio_cols,
    index=df_sample.index
)
X = pd.concat([X_audio_scaled, df_sample[genre_cols]], axis=1)

print(f"\n✅ Features scaled. Final shape: {X.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION — Popularity Tiers (Low / Mid / High)
#    Using cross-validation instead of a single train/test split
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CLASSIFICATION: Predicting Popularity Tier")
print("="*60)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
}
if XGBOOST_AVAILABLE:
    classifiers["XGBoost"] = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0
    )

clf_results = {}
for name, model in classifiers.items():
    scores = cross_val_score(
        model, X, y_tier,
        cv=cv_strategy, scoring="accuracy", n_jobs=-1
    )
    clf_results[name] = scores
    print(f"\n{name}")
    print(f"  Fold accuracies: {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean accuracy:   {scores.mean():.4f} ({scores.mean()*100:.1f}%)")
    print(f"  Std deviation:   ±{scores.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. REGRESSION — Predicting Exact Popularity Score
#    Metrics: R² and RMSE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("REGRESSION: Predicting Exact Popularity Score")
print("="*60)

cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

regressors = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
}
if XGBOOST_AVAILABLE:
    regressors["XGBoost"] = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42, verbosity=0
    )

reg_results = {}
for name, model in regressors.items():
    r2_scores   = cross_val_score(model, X, y_score, cv=cv_reg, scoring="r2", n_jobs=-1)
    rmse_scores = np.sqrt(-cross_val_score(
        model, X, y_score, cv=cv_reg,
        scoring="neg_mean_squared_error", n_jobs=-1
    ))
    reg_results[name] = {"r2": r2_scores, "rmse": rmse_scores}
    print(f"\n{name}")
    print(f"  R²   — Mean: {r2_scores.mean():.4f}  |  Std: ±{r2_scores.std():.4f}")
    print(f"  RMSE — Mean: {rmse_scores.mean():.2f}   |  Std: ±{rmse_scores.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Spotify ML — Model Results", fontsize=16, fontweight="bold", y=1.01)

# ── 6a. Popularity tier distribution ──
ax = axes[0, 0]
colors = ["#ef4444", "#f59e0b", "#1DB954"]
tier_counts[["Low", "Mid", "High"]].plot(
    kind="bar", ax=ax, color=colors, edgecolor="none"
)
ax.set_title("Popularity Tier Distribution")
ax.set_xlabel("Tier")
ax.set_ylabel("Track Count")
ax.tick_params(axis="x", rotation=0)

# ── 6b. Classification accuracy by model (box plot across folds) ──
ax = axes[0, 1]
names = list(clf_results.keys())
data  = [clf_results[n] for n in names]
bp = ax.boxplot(data, labels=names, patch_artist=True)
colors_box = ["#3b82f6", "#a855f7", "#f59e0b"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Classification Accuracy (5-Fold CV)")
ax.set_ylabel("Accuracy")
ax.tick_params(axis="x", rotation=15)

# ── 6c. R² scores across folds ──
ax = axes[1, 0]
for i, (name, res) in enumerate(reg_results.items()):
    ax.plot(range(1, 6), res["r2"], marker="o", label=name, linewidth=2)
    ax.axhline(res["r2"].mean(), linestyle="--", alpha=0.4)
ax.set_title("Regression R² Across Folds")
ax.set_xlabel("Fold")
ax.set_ylabel("R²")
ax.legend()

# ── 6d. RMSE scores across folds ──
ax = axes[1, 1]
for name, res in reg_results.items():
    ax.plot(range(1, 6), res["rmse"], marker="s", label=name, linewidth=2)
    ax.axhline(res["rmse"].mean(), linestyle="--", alpha=0.4)
ax.set_title("Regression RMSE Across Folds")
ax.set_xlabel("Fold")
ax.set_ylabel("RMSE (popularity points)")
ax.legend()

plt.tight_layout()
plt.savefig("spotify_ml_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Plot saved as spotify_ml_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE (best classifier: Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FEATURE IMPORTANCE — Random Forest Classifier")
print("="*60)

rf_final = RandomForestClassifier(
    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
)
rf_final.fit(X, y_tier)

importance_df = pd.DataFrame({
    "feature":   X.columns,
    "importance": rf_final.feature_importances_
}).sort_values("importance", ascending=False).head(15)

print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df, x="importance", y="feature",
    palette="Greens_r", edgecolor="none"
)
plt.title("Top 15 Feature Importances (Random Forest)", fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Feature importance plot saved as feature_importance.png")

print("\n✅ All done!")
