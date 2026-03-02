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

# ── 4. BUILD TRAINING DATASET ─────────────────────────────────────────────────
# Strategy:
# - Defaulter side: confident defaulters only (prob < 0.35)
#   gives the model a sharp view of what bad looks like
# - Repaid side: random sample matching defaulter count
#   gives the model full diversity of good borrowers
print("Building training dataset...")

# Step 1 — quick balanced initial model to score defaulters
n_def        = (y == 0).sum()
def_idx      = y[y == 0].index
rep_idx      = y[y == 1].index.to_series().sample(n=n_def, random_state=42)
bal_idx      = pd.concat([def_idx.to_series(), rep_idx])

X_bal_proc   = preprocessor.fit_transform(X.loc[bal_idx])
y_bal        = y.loc[bal_idx]

initial_model = xgb.XGBClassifier(
    max_depth=4, n_estimators=100, learning_rate=0.1,
    random_state=42, n_jobs=-1, eval_metric="auc"
)
initial_model.fit(X_bal_proc, y_bal, verbose=False)

# Step 2 — score all defaulters and keep only confident ones (prob < 0.35)
X_all_proc   = preprocessor.transform(X)
all_probs    = initial_model.predict_proba(X_all_proc)[:, 1]

df_scored             = df[all_features + ["target"]].copy()
df_scored["prob"]     = all_probs

confident_defaulters  = df_scored[
    (df_scored["target"] == 0) &
    (df_scored["prob"] < 0.35)
]
print(f"Confident defaulters: {len(confident_defaulters):,} "
      f"(avg prob: {confident_defaulters['prob'].mean():.3f})")

# Step 3 — random repaid sample matching defaulter count
n_defaulters   = len(confident_defaulters)
repaid_sampled = df[df["target"] == 1].sample(n=n_defaulters, random_state=42)
print(f"Random repaid sample: {len(repaid_sampled):,}")

training_df  = pd.concat([repaid_sampled, confident_defaulters])
print(f"Final training size:  {len(training_df):,}")

X_train_data = training_df[all_features]
y_train_data = training_df["target"]

# ── 5. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_train_data, y_train_data,
    test_size=0.2, random_state=42, stratify=y_train_data
)
print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ── 6. FIT PREPROCESSOR ON TRAINING DATA ONLY ────────────────────────────────
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# ── 7. TRAIN XGBOOST ─────────────────────────────────────────────────────────
print("\nTraining model...")
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

# ── 8. PREDICT ────────────────────────────────────────────────────────────────
y_pred = xgb_model.predict(X_test_proc)
y_prob = xgb_model.predict_proba(X_test_proc)[:, 1]

# ── 9. EVALUATION ─────────────────────────────────────────────────────────────
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

# ── 10. FULL DATASET COVERAGE ─────────────────────────────────────────────────
print("\n── Full Dataset Coverage ──")
X_full_proc   = preprocessor.transform(X)
full_probs    = xgb_model.predict_proba(X_full_proc)[:, 1]
full_approved = (full_probs >= 0.50).sum()
print(f"Total applicants:  {len(df):,}")
print(f"Would approve:     {full_approved:,} ({full_approved/len(df):.1%})")
print(f"Would decline:     {len(df)-full_approved:,} ({(len(df)-full_approved)/len(df):.1%})")

approved_mask = full_probs >= 0.50
print(f"Actual repay rate among approved: {y[approved_mask].mean():.1%}")
print(f"Actual repay rate among declined: {y[~approved_mask].mean():.1%}")

# ── 11. CONFUSION MATRIX ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Defaulted", "Repaid"],
    cmap="Blues", ax=ax
)
ax.set_title("Confusion Matrix — XGBoost")
plt.tight_layout()
plt.savefig("confusion_matrix_xgb.png", dpi=120)
plt.close()
print("Saved: confusion_matrix_xgb.png")

# ── 12. ROC CURVE ─────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"XGBoost AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve_xgb.png", dpi=120)
plt.close()
print("Saved: roc_curve_xgb.png")

# ── 13. FEATURE IMPORTANCE ────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature":    all_features,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n── Feature Importances ──")
print(importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df["feature"], importance_df["importance"], color="#3498db")
ax.set_title("XGBoost Feature Importances")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_xgb.png", dpi=120)
plt.close()
print("Saved: feature_importance_xgb.png")

# ── 14. SHAP VALUES ───────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_proc[:500])

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_proc[:500], feature_names=all_features, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=120)
plt.close()
print("Saved: shap_summary.png")

# ── 15. MODEL VERSIONING ──────────────────────────────────────────────────────
os.makedirs("model_registry", exist_ok=True)
version           = datetime.now().strftime("%Y%m%d_%H%M%S")
preprocessor_path = f"model_registry/preprocessor_{version}.joblib"
model_path        = f"model_registry/xgb_model_{version}.joblib"

joblib.dump(preprocessor, preprocessor_path)
joblib.dump(xgb_model,    model_path)

metadata = {
    "version":          version,
    "roc_auc":          round(roc_auc, 4),
    "best_iteration":   int(xgb_model.best_iteration),
    "features":         all_features,
    "train_size":       len(X_train),
    "test_size":        len(X_test),
    "n_defaulters":     n_defaulters,
    "n_repaid_sampled": len(repaid_sampled),
    "trained_at":       datetime.now().isoformat()
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