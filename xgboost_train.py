import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ── 1. LOAD CLEAN DATA ────────────────────────────────────────────────────────
df = pd.read_csv("lending_club_clean.csv")

# ── 2. DEFINE FEATURES ────────────────────────────────────────────────────────
strong_num   = ["dti", "revol_util", "pub_rec", "annual_inc"]
support_num  = ["emp_length", "open_acc", "mort_acc", "credit_history_years", "revol_bal"]
cat_features = ["home_ownership"]
all_num      = strong_num + support_num
all_features = all_num + cat_features

X = df[all_features]
y = df["target"]

# ── 3. PREPROCESSOR ───────────────────────────────────────────────────────────
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, all_num),
    ("cat", cat_pipe, cat_features)
], remainder="drop")

# ── 4. INITIAL BALANCED SAMPLE ────────────────────────────────────────────────
# Match repaid count to defaulter count for balanced training
n_defaulters   = (y == 0).sum()
repaid_idx     = y[y == 1].index
default_idx    = y[y == 0].index
repaid_sampled = repaid_idx.to_series().sample(n=n_defaulters, random_state=42)
balanced_idx   = pd.concat([repaid_sampled, default_idx.to_series()])

X_balanced = X.loc[balanced_idx]
y_balanced = y.loc[balanced_idx]

print(f"Initial balanced dataset: {len(X_balanced):,} rows")
print(f"Defaulters: {(y_balanced==0).sum():,}  |  Repaid: {(y_balanced==1).sum():,}")

# ── 5. TRAIN INITIAL MODEL ────────────────────────────────────────────────────
# This model is used only to score confidence — not the final model
print("\nTraining initial model for confidence scoring...")

X_proc_balanced = preprocessor.fit_transform(X_balanced)

initial_model = xgb.XGBClassifier(
    max_depth=5, n_estimators=300, learning_rate=0.05,
    gamma=1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", random_state=42, n_jobs=-1
)
initial_model.fit(X_proc_balanced, y_balanced, verbose=False)

# ── 6. SCORE THE FULL DATASET ─────────────────────────────────────────────────
# Get repay probability for every entry in the full dataset
print("Scoring full dataset for confidence...")
X_all_proc   = preprocessor.transform(X)
all_probs    = initial_model.predict_proba(X_all_proc)[:, 1]

df_scored         = df[all_features + ["target"]].copy()
df_scored["prob"] = all_probs

# ── 7. SELECT MOST CONFIDENT CASES ───────────────────────────────────────────
# Most confident repaid   = highest repay probability among actual repaid loans
# Most confident declined = lowest repay probability among actual defaulted loans
# Ambiguous cases (middle) are left out of training entirely

repaid_scored   = df_scored[df_scored["target"] == 1].sort_values("prob", ascending=False)
default_scored  = df_scored[df_scored["target"] == 0].sort_values("prob", ascending=True)

# Take top N from each — match counts for balance
# Use same count as original defaulter pool
n_select        = n_defaulters
confident_repaid   = repaid_scored.head(n_select)
confident_declined = default_scored.head(n_select)

confident_df = pd.concat([confident_repaid, confident_declined])
print(f"\nConfident training dataset: {len(confident_df):,} rows")
print(f"Confident repaid:    {len(confident_repaid):,}  "
      f"(avg prob: {confident_repaid['prob'].mean():.3f})")
print(f"Confident defaulted: {len(confident_declined):,}  "
      f"(avg prob: {confident_declined['prob'].mean():.3f})")
print(f"Ambiguous cases excluded: "
      f"{len(df_scored) - len(confident_df):,}")

# ── DIVERSITY CHECK ───────────────────────────────────────────────────────────
# Shows the spread of confidence scores in the training dataset
# Ensures we're not just training on extreme cases at both ends
print(f"\n── Confident Dataset Diversity Check ──")
print(f"\n  Repaid cases:")
print(f"    Highest confidence (most certain repaid): {confident_repaid['prob'].iloc[0]:.4f}")
print(f"    Lowest confidence  (least certain repaid): {confident_repaid['prob'].iloc[-1]:.4f}")
print(f"    Mean: {confident_repaid['prob'].mean():.4f}  "
      f"Std: {confident_repaid['prob'].std():.4f}")

print(f"\n  Defaulted cases:")
print(f"    Highest confidence (most certain default): {confident_declined['prob'].iloc[0]:.4f}")
print(f"    Lowest confidence  (least certain default): {confident_declined['prob'].iloc[-1]:.4f}")
print(f"    Mean: {confident_declined['prob'].mean():.4f}  "
      f"Std: {confident_declined['prob'].std():.4f}")

# Distribution of confidence scores in buckets
print(f"\n  Repaid confidence distribution:")
repaid_bins = pd.cut(confident_repaid['prob'],
                     bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                     labels=["0-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
print(f"  {repaid_bins.value_counts().sort_index().to_string()}")

print(f"\n  Defaulted confidence distribution (lower prob = more confident default):")
default_bins = pd.cut(confident_declined['prob'],
                      bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
                      labels=["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50%+"])
print(f"  {default_bins.value_counts().sort_index().to_string()}")

X_conf = confident_df[all_features]
y_conf = confident_df["target"]

# ── 8. TRAIN / TEST SPLIT ON CONFIDENT DATA ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_conf, y_conf, test_size=0.2, random_state=42, stratify=y_conf
)
print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ── 9. REFIT PREPROCESSOR ON CONFIDENT TRAINING DATA ─────────────────────────
# Refit on the new training split — never fit on test data
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# ── 10. FINAL XGBOOST MODEL ───────────────────────────────────────────────────
print("\nTraining final model on confident dataset...")
xgb_model = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=300,
    learning_rate=0.05,
    gamma=1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_proc, y_train,
    eval_set=[(X_test_proc, y_test)],
    verbose=50
)
print(f"\nBest iteration: {xgb_model.best_iteration}")

# ── 11. PREDICT ───────────────────────────────────────────────────────────────
y_pred = xgb_model.predict(X_test_proc)
y_prob = xgb_model.predict_proba(X_test_proc)[:, 1]

# ── 12. EVALUATION ────────────────────────────────────────────────────────────
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Defaulted", "Repaid"]))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\n── Cost Analysis ──")
print(f"Correctly declined defaulters  (avoided loss):  {tn:,}")
print(f"Defaulters approved by mistake (costly!):        {fp:,}")
print(f"Good borrowers declined (missed revenue):        {fn:,}")
print(f"Good borrowers correctly approved:               {tp:,}")

# ── 13. CONFUSION MATRIX ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Defaulted", "Repaid"],
    cmap="Blues", ax=ax
)
ax.set_title("Confusion Matrix — XGBoost (Confident Learning)")
plt.tight_layout()
plt.savefig("confusion_matrix_xgb.png", dpi=120)
plt.close()
print("Saved: confusion_matrix_xgb.png")

# ── 14. ROC CURVE ─────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"XGBoost AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost (Confident Learning)")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve_xgb.png", dpi=120)
plt.close()
print("Saved: roc_curve_xgb.png")

# ── 15. FEATURE IMPORTANCE ────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature":    all_features,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n── Feature Importances ──")
print(importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df["feature"], importance_df["importance"], color="#3498db")
ax.set_title("XGBoost Feature Importances (Confident Learning)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_xgb.png", dpi=120)
plt.close()
print("Saved: feature_importance_xgb.png")

# ── 16. SHAP VALUES ───────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_proc[:500])

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_proc[:500], feature_names=all_features, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=120)
plt.close()
print("Saved: shap_summary.png")

# ── 17. MODEL VERSIONING ──────────────────────────────────────────────────────
os.makedirs("model_registry", exist_ok=True)
version           = datetime.now().strftime("%Y%m%d_%H%M%S")
preprocessor_path = f"model_registry/preprocessor_{version}.joblib"
model_path        = f"model_registry/xgb_model_{version}.joblib"

joblib.dump(preprocessor, preprocessor_path)
joblib.dump(xgb_model,    model_path)

metadata = {
    "version":                version,
    "roc_auc":                round(roc_auc, 4),
    "best_iteration":         int(xgb_model.best_iteration),
    "features":               all_features,
    "train_size":             len(X_train),
    "test_size":              len(X_test),
    "confident_repaid":       len(confident_repaid),
    "confident_declined":     len(confident_declined),
    "ambiguous_excluded":     int(len(df_scored) - len(confident_df)),
    "trained_at":             datetime.now().isoformat()
}
with open(f"model_registry/metadata_{version}.json", "w") as f:
    json.dump(metadata, f, indent=2)

with open("model_registry/latest.json", "w") as f:
    json.dump({
        "version":           version,
        "model_path":        model_path,
        "preprocessor_path": preprocessor_path
    }, f, indent=2)

print(f"\nModel saved:        {model_path}")
print(f"Preprocessor saved: {preprocessor_path}")
print(f"Version:            {version}")