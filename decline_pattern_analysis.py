import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── 1. SETUP ──────────────────────────────────────────────────────────────────
df = pd.read_csv("lending_club_clean.csv")

with open("model_registry/latest.json") as f:
    latest = json.load(f)
main_model        = joblib.load(latest["model_path"])
main_preprocessor = joblib.load(latest["preprocessor_path"])

strong_num   = ["dti", "revol_util", "pub_rec", "annual_inc"]
support_num  = ["emp_length", "open_acc", "mort_acc", "credit_history_years", "revol_bal"]
cat_features = ["home_ownership"]
all_num      = strong_num + support_num
all_features = all_num + cat_features

APPROVAL_THRESHOLD     = 0.50
REPAID_MODEL_THRESHOLD = 0.50

# ── 2. REBUILD REPAID MODEL ───────────────────────────────────────────────────
print("Rebuilding repaid-only model...")
repaid_only = df[df["target"] == 1][all_features].copy()
repaid_only["repaid_prob"] = main_model.predict_proba(
    main_preprocessor.transform(repaid_only)
)[:, 1]
median_prob              = repaid_only["repaid_prob"].median()
repaid_only["repaid_label"] = (repaid_only["repaid_prob"] >= median_prob).astype(int)

repaid_preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), all_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), cat_features)
], remainder="drop")

X_repaid_proc = repaid_preprocessor.fit_transform(repaid_only[all_features])
repaid_model  = xgb.XGBClassifier(
    max_depth=4, n_estimators=200, learning_rate=0.05,
    gamma=1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", random_state=42, n_jobs=-1
)
repaid_model.fit(X_repaid_proc, repaid_only["repaid_label"], verbose=False)
print("Repaid model ready.\n")

# ── 3. SCORE ALL APPLICANTS ───────────────────────────────────────────────────
print("Scoring all applicants...")
X      = df[all_features]
X_main = main_preprocessor.transform(X)
X_rep  = repaid_preprocessor.transform(X)

main_probs    = main_model.predict_proba(X_main)[:, 1]
repaid_scores = repaid_model.predict_proba(X_rep)[:, 1]

df_scored = df[all_features + ["target"]].copy()
df_scored["main_prob"]    = main_probs
df_scored["repaid_score"] = repaid_scores
df_scored["main_dec"]     = (main_probs   >= APPROVAL_THRESHOLD).astype(int)
df_scored["repaid_dec"]   = (repaid_scores >= REPAID_MODEL_THRESHOLD).astype(int)

# ── 4. CATEGORISE DECLINED APPLICANTS ────────────────────────────────────────
# For declined applicants, figure out which model is responsible
declined = df_scored[df_scored["main_dec"] == 0].copy()

def decline_source(row):
    # Both models agree they're risky — clear decline
    if row["repaid_dec"] == 0:
        return "Both models decline"
    # Main model declines but repaid model thinks they look okay — main model is strict
    else:
        return "Main model strict"

declined["decline_source"] = declined.apply(decline_source, axis=1)

print(f"── Declined Applicant Breakdown ──")
print(f"Total declined: {len(declined):,}")
source_counts = declined["decline_source"].value_counts()
for source, count in source_counts.items():
    pct = count / len(declined) * 100
    print(f"  {source}: {count:,} ({pct:.1f}%)")

# ── 5. SHAP FOR TOP DECLINE REASONS ──────────────────────────────────────────
print("\nComputing SHAP values for declined applicants (sample of 2000)...")

# Sample for speed — stratified by decline source
sample_both   = declined[declined["decline_source"] == "Both models decline"].sample(
    min(1000, len(declined[declined["decline_source"] == "Both models decline"])),
    random_state=42
)
sample_strict = declined[declined["decline_source"] == "Main model strict"].sample(
    min(1000, len(declined[declined["decline_source"] == "Main model strict"])),
    random_state=42
)

explainer = shap.TreeExplainer(main_model)

def top_negative_feature(X_proc):
    """Returns the feature with the most negative SHAP value per row"""
    shap_vals = explainer.shap_values(X_proc)
    top_neg   = np.argmin(shap_vals, axis=1)
    return [all_features[i] for i in top_neg]

# Top decline reason for "both models decline" group
X_both_proc   = main_preprocessor.transform(sample_both[all_features])
X_strict_proc = main_preprocessor.transform(sample_strict[all_features])

sample_both   = sample_both.copy()
sample_strict = sample_strict.copy()

sample_both["top_reason"]   = top_negative_feature(X_both_proc)
sample_strict["top_reason"] = top_negative_feature(X_strict_proc)

# ── 6. TOP REASONS BY GROUP ───────────────────────────────────────────────────
print("\n── Top Decline Reasons — Both Models Agree ──")
both_reasons = sample_both["top_reason"].value_counts(normalize=True) * 100
print(both_reasons.round(1).to_string())

print("\n── Top Decline Reasons — Main Model Strict (repaid model disagrees) ──")
strict_reasons = sample_strict["top_reason"].value_counts(normalize=True) * 100
print(strict_reasons.round(1).to_string())

# ── 7. FEATURE AVERAGES BY DECLINE SOURCE ────────────────────────────────────
print("\n── Average Feature Values by Decline Source ──")
approved = df_scored[df_scored["main_dec"] == 1]
comparison = pd.DataFrame({
    "Approved":           approved[all_num].mean(),
    "Both Decline":       sample_both[all_num].mean(),
    "Main Strict":        sample_strict[all_num].mean(),
}).round(2)
print(comparison.to_string())

# ── 8. PLOT TOP REASONS ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

both_reasons.plot(kind="bar", ax=axes[0], color="#e74c3c")
axes[0].set_title("Top Decline Reasons\nBoth Models Agree")
axes[0].set_ylabel("% of Declines")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

strict_reasons.plot(kind="bar", ax=axes[1], color="#f39c12")
axes[1].set_title("Top Decline Reasons\nMain Model Strict Only")
axes[1].set_ylabel("% of Declines")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

plt.suptitle("What Is Driving Declines?", fontsize=13)
plt.tight_layout()
plt.savefig("decline_patterns.png", dpi=120)
plt.close()
print("\nSaved: decline_patterns.png")

# ── 9. ACTUAL REPAYMENT RATE BY DECLINE SOURCE ────────────────────────────────
print("\n── Actual Repayment Rate by Decline Source ──")
print(f"Both models decline — actually repaid: "
      f"{(declined[declined['decline_source']=='Both models decline']['target']==1).mean():.1%}")
print(f"Main model strict  — actually repaid: "
      f"{(declined[declined['decline_source']=='Main model strict']['target']==1).mean():.1%}")