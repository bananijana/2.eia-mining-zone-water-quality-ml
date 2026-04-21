"""
ML Classification — Mining Zone Impact Assessment
---------------------------------------------------
Trains a Random Forest classifier to predict EQI-based
impact class directly from raw water quality parameters,
bypassing the index computation step.

Compares ML predictions against index-derived classes
to validate both approaches.

Authors:
  Lead   : Banani Jana
  Support: Anikate Chowdhury (ML pipeline)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "outputs/eqi_results.csv"   # use EQI-computed results
OUTPUT_DIR = "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = ["pH", "EC", "TDS", "Temp", "SO4", "NO3",
            "Cl", "HCO3", "Fe", "Mn", "As", "Pb", "Cu", "Zn"]
TARGET   = "ImpactClass"

RANDOM_STATE = 42
TEST_SIZE    = 0.25

IMPACT_PALETTE = {
    "Low":       "#388E3C",
    "Moderate":  "#F9A825",
    "High":      "#F57C00",
    "Very High": "#D32F2F",
}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} samples")
print(f"Impact class distribution:\n{df[TARGET].value_counts()}\n")

X  = df[FEATURES].values
le = LabelEncoder()
y  = le.fit_transform(df[TARGET])
classes = le.classes_
print(f"Classes: {list(classes)}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

# ── Cross-Validation ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Classification Report ─────────────────────────────────────────────────────
report = classification_report(y_test, y_pred,
                                target_names=classes, output_dict=True)
pd.DataFrame(report).T.round(3).to_csv(
    f"{OUTPUT_DIR}/classification_report.csv")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# ── Plot 1: Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=classes).plot(
    ax=ax, cmap="Oranges", colorbar=False,
    text_kw={"fontsize": 12, "fontweight": "bold"})
ax.set_title("Confusion Matrix — Impact Class Prediction\n(Random Forest)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("True", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=180)
plt.close()
print("Saved: confusion_matrix.png")

# ── Plot 2: Feature Importance ────────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#B71C1C" if v >= importances.quantile(0.75) else
          "#EF9A9A" if v >= importances.median() else
          "#FFCDD2" for v in importances.values]
bars = ax.barh(importances.index, importances.values,
               color=colors, edgecolor="white", height=0.65)
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
ax.axvline(importances.median(), color="navy", linestyle="--",
           linewidth=1.3, alpha=0.7, label="Median importance")
ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
ax.set_title("Random Forest — Parameter Importance\nfor Impact Class Prediction",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=180)
plt.close()
print("Saved: feature_importance.png")

# ── Plot 3: ROC Curves ────────────────────────────────────────────────────────
y_test_bin  = label_binarize(y_test, classes=range(len(classes)))
class_cols  = list(IMPACT_PALETTE.values())

fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls, col) in enumerate(zip(classes, class_cols)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, linewidth=2,
            label=f"{cls}  (AUC = {roc_auc:.2f})")
ax.plot([0,1],[0,1],"k--", linewidth=1, alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — Impact Class (One-vs-Rest)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=180)
plt.close()
print("Saved: roc_curves.png")

# ── Plot 4: CV Scores ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
folds   = [f"Fold {i+1}" for i in range(len(cv_scores))]
bcols   = ["#B71C1C" if s >= cv_scores.mean() else "#EF9A9A"
           for s in cv_scores]
ax.bar(folds, cv_scores, color=bcols, edgecolor="white", width=0.5)
ax.axhline(cv_scores.mean(), color="navy", linestyle="--",
           linewidth=1.5, label=f"Mean = {cv_scores.mean():.4f}")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=12,
             fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, val in zip(ax.patches, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/cv_scores.png", dpi=180)
plt.close()
print("Saved: cv_scores.png")

# ── Plot 5: ML vs Index Agreement ────────────────────────────────────────────
df_test = df.iloc[
    pd.read_csv(DATA_PATH).index[
        np.where(np.isin(np.arange(len(df)),
                 np.where(np.ones(len(df)))[0][-len(y_test):]))[0]
    ]
].copy() if False else df.copy()

# Simpler: add predictions to full dataset via cross_val_predict
from sklearn.model_selection import cross_val_predict
y_cv_pred  = cross_val_predict(rf, X, y, cv=cv)
df["ML_Predicted"] = le.inverse_transform(y_cv_pred)
df["Agreement"]    = df[TARGET] == df["ML_Predicted"]

agree_by_zone = df.groupby("Zone")["Agreement"].mean() * 100
fig, ax = plt.subplots(figsize=(7, 4))
zone_order = ["Impact", "Buffer", "Background"]
zone_cols  = ["#D32F2F", "#F57C00", "#388E3C"]
bars = ax.bar(zone_order,
              [agree_by_zone[z] for z in zone_order],
              color=zone_cols, edgecolor="white", width=0.5)
ax.set_ylim(0, 115)
ax.set_ylabel("Agreement (%)", fontsize=11)
ax.set_title("ML Prediction vs Index-Based Classification\nAgreement by Zone",
             fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, z in zip(bars, zone_order):
    val = agree_by_zone[z]
    ax.text(bar.get_x() + bar.get_width()/2, val + 2,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/ml_vs_index_agreement.png", dpi=180)
plt.close()
print("Saved: ml_vs_index_agreement.png")

print(f"\n✓ All ML outputs saved to: {OUTPUT_DIR}")
print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
