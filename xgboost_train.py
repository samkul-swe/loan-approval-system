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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ── 1. LOAD CLEAN DATA ────────────────────────────────────────────────────────
df = pd.read_csv("lending_club_clean.csv")

# ── 2. DEFINE FEATURES ────────────────────────────────────────────────────────
strong_num   = ["dti", "revol_util", "pub_rec", "annual_inc"]
support_num  = ["emp_length", "open_acc", "mort_acc", "credit_history_years", "revol_bal"]
cat_features = ["home_ownership", "verification_status"]
all_num      = strong_num + support_num

X = df[all_num + cat_features]
y = df["target"]

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ── 4. PREPROCESSING PIPELINE ─────────────────────────────────────────────────
# XGBoost can handle ordinally encoded categories natively
# We use OrdinalEncoder instead of OneHotEncoder to keep feature names clean
# and make SHAP values easier to interpret later
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
    # No scaling needed — XGBoost is tree-based, scale doesn't affect splits
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, all_num),
    ("cat", cat_pipe, cat_features)
])

# ── 5. COMPUTE CLASS WEIGHT ───────────────────────────────────────────────────
# Reflects the 3x cost asymmetry — missing a defaulter costs 3x more
# scale_pos_weight = count(negative) / count(positive)
# We then multiply by 3 to reflect the business cost
n_repaid   = (y_train == 1).sum()
n_default  = (y_train == 0).sum()
scale_pos  = round((n_repaid / n_default) * 3, 2)
print(f"scale_pos_weight: {scale_pos}")
# Why 3x? The problem states defaults cost ClearLend 3x more than missed revenue.
# Without this the model would naturally optimise for the majority class (repaid).
# Setting this higher makes the model more conservative — it penalises approving
# a defaulter much more than declining a good borrower.

# ── 6. XGBOOST MODEL ─────────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(

    # max_depth=5: each tree can ask up to 5 questions deep.
    # Too shallow (2-3): misses important feature interactions.
    # Too deep (8-10): memorises training data, won't generalise.
    # 5 gives enough depth for layered decisions without overfitting.
    max_depth=5,

    # n_estimators=300: number of trees to build sequentially.
    # Too few (50): underfits, not enough correction passes.
    # Too many (1000+): very slow, marginal gains, risk of overfitting.
    # 300 is a solid middle ground for this dataset size.
    n_estimators=300,

    # learning_rate=0.05: how much each new tree corrects the previous one.
    # Too high (0.3+): trees overcorrect, model becomes unstable.
    # Too low (0.01): needs many more trees to converge, slow.
    # 0.05 with 300 trees gives stable, gradual improvement.
    learning_rate=0.05,

    # gamma=1: minimum improvement required to create a new branch.
    # gamma=0: splits freely, risks overfitting on noise.
    # gamma=5+: too restrictive, model becomes too shallow.
    # gamma=1 ensures splits only happen when they genuinely help —
    # aligns with your intuition of "only branch on meaningful new relations".
    gamma=1,

    # subsample=0.8: each tree is trained on 80% of training rows (randomly sampled).
    # This introduces variety between trees, reducing overfitting.
    # 1.0 = use all rows every time (more overfit risk).
    # 0.5 = too little data per tree, underfits.
    subsample=0.8,

    # colsample_bytree=0.8: each tree only sees 80% of features.
    # Forces trees to find different combinations of features,
    # making the ensemble more robust and diverse.
    colsample_bytree=0.8,

    # scale_pos_weight: handles class imbalance + 3x cost asymmetry.
    # Computed above based on actual class counts in training data.
    scale_pos_weight=scale_pos,

    # eval_metric: what to optimise during training.
    # auc = ROC-AUC, directly matches our evaluation metric.
    eval_metric="auc",

    # early_stopping_rounds=30: stop training if AUC hasn't improved
    # in the last 30 trees. Prevents unnecessary training and overfitting.
    early_stopping_rounds=30,

    use_label_encoder=False,
    random_state=42,        # reproducibility — same input always gives same output
    n_jobs=-1               # use all CPU cores
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   xgb_model)
])

# ── 7. TRAIN ──────────────────────────────────────────────────────────────────
# Prepare eval set for early stopping
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

xgb_model.fit(
    X_train_proc, y_train,
    eval_set=[(X_test_proc, y_test)],
    verbose=50   # print progress every 50 trees
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

# ── 10. CONFUSION MATRIX ──────────────────────────────────────────────────────
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

# ── 11. ROC CURVE ─────────────────────────────────────────────────────────────
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

# ── 12. FEATURE IMPORTANCE ────────────────────────────────────────────────────
feature_names = all_num + cat_features
importance_df = pd.DataFrame({
    "feature":    feature_names,
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

# ── 13. SHAP VALUES (explainability) ─────────────────────────────────────────
# SHAP explains each individual prediction — which features pushed
# this specific applicant toward approval or decline and by how much.
# This is what generates the 3 human-readable decline reasons later.
explainer  = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_proc[:500])  # sample for speed

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_test_proc[:500],
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=120)
plt.close()
print("Saved: shap_summary.png")

# ── 14. MODEL VERSIONING ──────────────────────────────────────────────────────
# Every saved model gets a timestamp version so you can roll back at any time.
os.makedirs("model_registry", exist_ok=True)
version    = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"model_registry/xgb_model_{version}.joblib"

# Save preprocessor and model separately for clean inference
joblib.dump(preprocessor, f"model_registry/preprocessor_{version}.joblib")
joblib.dump(xgb_model,    model_path)

# Save metadata alongside the model
metadata = {
    "version":         version,
    "roc_auc":         round(roc_auc, 4),
    "best_iteration":  int(xgb_model.best_iteration),
    "features":        feature_names,
    "train_size":      len(X_train),
    "test_size":       len(X_test),
    "scale_pos_weight": scale_pos,
    "trained_at":      datetime.now().isoformat()
}
with open(f"model_registry/metadata_{version}.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Always keep a pointer to the latest model for inference
with open("model_registry/latest.json", "w") as f:
    json.dump({"version": version, "model_path": model_path}, f, indent=2)

print(f"\nModel saved: {model_path}")
print(f"Version: {version}")
print(f"Metadata: model_registry/metadata_{version}.json")