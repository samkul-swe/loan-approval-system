import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ── 1. LOAD CLEAN DATA ────────────────────────────────────────────────────────
df = pd.read_csv("lending_club_clean.csv")

# ── 2. DEFINE FEATURES ────────────────────────────────────────────────────────
# Strong features — primary decision drivers
strong_num   = ["dti", "revol_util", "pub_rec", "pub_rec_bankruptcies", "annual_inc"]

# Supporting numerical features
support_num  = ["emp_length", "open_acc", "total_acc", "mort_acc",
                "credit_history_years", "revol_bal"]

# Categorical features
cat_features = ["home_ownership", "verification_status"]

all_num = strong_num + support_num
X = df[all_num + cat_features]
y = df["target"]

# ── 3. TRAIN / TEST SPLIT (stratified to preserve class balance) ──────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"Train default rate: {(y_train == 0).mean():.2%}")
print(f"Test  default rate: {(y_test  == 0).mean():.2%}")

# ── 4. PREPROCESSING PIPELINE ─────────────────────────────────────────────────
# Numerical: impute missing with median, then scale
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Categorical: impute missing with most frequent, then one-hot encode
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, all_num),
    ("cat", cat_pipe, cat_features)
])

# ── 5. MODEL ──────────────────────────────────────────────────────────────────
# class_weight="balanced" handles the 3x cost asymmetry —
# the model penalises missing a defaulter more than declining a good borrower
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# ── 6. TRAIN ──────────────────────────────────────────────────────────────────
model.fit(X_train, y_train)
print("\nModel trained.")

# ── 7. PREDICT ────────────────────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]  # probability of repaying

# ── 8. EVALUATION ─────────────────────────────────────────────────────────────
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Defaulted", "Repaid"]))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ── 9. CONFUSION MATRIX ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Defaulted", "Repaid"],
    cmap="Blues", ax=ax
)
ax.set_title("Confusion Matrix — Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png", dpi=120)
plt.show()

# Breakdown of costly mistakes
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\n── Cost Analysis ──")
print(f"Correctly declined defaulters  (avoided loss):  {tn:,}")
print(f"Defaulters approved by mistake (costly!):        {fp:,}")
print(f"Good borrowers declined (missed revenue):        {fn:,}")
print(f"Good borrowers correctly approved:               {tp:,}")

# ── 10. ROC CURVE ─────────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#2ecc71", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Logistic Regression")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve_lr.png", dpi=120)
plt.show()

# ── 11. FEATURE IMPORTANCE (coefficients) ─────────────────────────────────────
# Get feature names after encoding
ohe_cats = model.named_steps["preprocessor"] \
               .named_transformers_["cat"] \
               .named_steps["encoder"] \
               .get_feature_names_out(cat_features)
feature_names = all_num + list(ohe_cats)

coefficients = model.named_steps["classifier"].coef_[0]
coef_df = pd.DataFrame({
    "feature":     feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", key=abs, ascending=False)

print("\n── Top Feature Importances (by coefficient magnitude) ──")
print(coef_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top = coef_df.head(15)
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in top["coefficient"]]
ax.barh(top["feature"], top["coefficient"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Top 15 Feature Coefficients\n(green = increases repay probability, red = decreases it)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_lr.png", dpi=120)
plt.show()