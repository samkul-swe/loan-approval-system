import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

# ── 1. LOAD DATA AND MODEL ────────────────────────────────────────────────────
df = pd.read_csv("lending_club_clean.csv")

with open("model_registry/latest.json") as f:
    latest = json.load(f)

model        = joblib.load(latest["model_path"])
preprocessor = joblib.load(latest["preprocessor_path"])

with open(f"model_registry/metadata_{latest['version']}.json") as f:
    metadata = json.load(f)

strong_num   = ["dti", "revol_util", "pub_rec", "annual_inc"]
engineered   = ["debt_burden", "credit_utilization_pressure",
                "has_public_record", "high_dti_low_income"]
support_num  = ["emp_length", "open_acc", "mort_acc", "credit_history_years", "revol_bal"]
cat_features = ["home_ownership"]
all_num      = strong_num + engineered + support_num
all_features = all_num + cat_features

X = df[all_features]
y = df["target"]

# ── 2. SCORE FULL DATASET ─────────────────────────────────────────────────────
print("Scoring full dataset...")
X_proc = preprocessor.transform(X)
probs  = model.predict_proba(X_proc)[:, 1]

# ── 3. CALIBRATION CURVE — BEFORE ────────────────────────────────────────────
# Split into 10 equal-sized buckets by predicted probability
# For each bucket: what is the actual repay rate?
# A perfectly calibrated model has predicted prob == actual repay rate
print("\n── Calibration Check: Predicted vs Actual Repay Rate ──\n")
print(f"  {'Prob Bucket':>12} {'Count':>8} {'Pred Prob':>10} {'Actual Rate':>12} {'Gap':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")

df_cal          = pd.DataFrame({"prob": probs, "actual": y})
df_cal["bucket"] = pd.qcut(df_cal["prob"], q=10, duplicates="drop")

calibration_data = []
for bucket, group in df_cal.groupby("bucket", observed=True):
    pred_prob   = group["prob"].mean()
    actual_rate = group["actual"].mean()
    gap         = actual_rate - pred_prob
    calibration_data.append({
        "bucket":      str(bucket),
        "count":       len(group),
        "pred_prob":   pred_prob,
        "actual_rate": actual_rate,
        "gap":         gap
    })
    print(f"  {str(bucket):>12} {len(group):>8,} {pred_prob:>10.3f} "
          f"{actual_rate:>12.3f} {gap:>+8.3f}")

cal_df      = pd.DataFrame(calibration_data)
mean_abs_gap = cal_df["gap"].abs().mean()
print(f"\n  Mean absolute gap: {mean_abs_gap:.4f}")
print(f"  {'Well calibrated' if mean_abs_gap < 0.05 else 'Needs calibration'}")

# Brier score — lower is better, 0 = perfect, 0.25 = random
brier = brier_score_loss(y, probs)
print(f"  Brier score: {brier:.4f} (lower = better, 0.25 = random)")

# ── 4. PLOT CALIBRATION CURVE — BEFORE ───────────────────────────────────────
fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
axes[0].plot(mean_pred, fraction_pos, "o-", color="#3498db", label="Current model")
axes[0].set_xlabel("Mean Predicted Probability")
axes[0].set_ylabel("Fraction of Positives (Actual Repay Rate)")
axes[0].set_title("Calibration Curve — Before Calibration")
axes[0].legend()
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# ── 5. APPLY ISOTONIC REGRESSION CALIBRATION ─────────────────────────────────
# Isotonic regression is non-parametric — it finds the best monotonic mapping
# from predicted probabilities to actual repay rates
# More flexible than Platt scaling for XGBoost
print("\nApplying isotonic regression calibration...")

# Use a fresh split for calibration — never calibrate on training data
X_cal_train, X_cal_test, y_cal_train, y_cal_test = train_test_split(
    X, y, test_size=0.3, random_state=99, stratify=y
)

X_cal_train_proc = preprocessor.transform(X_cal_train)
X_cal_test_proc  = preprocessor.transform(X_cal_test)

# Wrap the existing model with isotonic calibration
calibrated_model = CalibratedClassifierCV(
    model, method="isotonic", cv="prefit"
)
calibrated_model.fit(X_cal_train_proc, y_cal_train)

# ── 6. CALIBRATION CURVE — AFTER ─────────────────────────────────────────────
probs_cal = calibrated_model.predict_proba(X_cal_test_proc)[:, 1]

fraction_pos_cal, mean_pred_cal = calibration_curve(
    y_cal_test, probs_cal, n_bins=10, strategy="quantile"
)

axes[1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
axes[1].plot(mean_pred_cal, fraction_pos_cal, "o-", color="#2ecc71",
             label="Calibrated model")
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("Fraction of Positives (Actual Repay Rate)")
axes[1].set_title("Calibration Curve — After Isotonic Calibration")
axes[1].legend()
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

plt.suptitle("Probability Calibration: Before vs After", fontsize=13)
plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=120)
plt.close()
print("Saved: calibration_curve.png")

# ── 7. COMPARE METRICS BEFORE AND AFTER ──────────────────────────────────────
print("\n── Metrics Comparison ──")
probs_before = model.predict_proba(X_cal_test_proc)[:, 1]
auc_before   = roc_auc_score(y_cal_test, probs_before)
auc_after    = roc_auc_score(y_cal_test, probs_cal)
brier_before = brier_score_loss(y_cal_test, probs_before)
brier_after  = brier_score_loss(y_cal_test, probs_cal)

print(f"  ROC-AUC  — Before: {auc_before:.4f}  After: {auc_after:.4f}")
print(f"  Brier    — Before: {brier_before:.4f}  After: {brier_after:.4f} "
      f"({'improved' if brier_after < brier_before else 'no improvement'})")

# ── 8. CALIBRATED BUCKET CHECK ────────────────────────────────────────────────
print("\n── Calibration Check After: Predicted vs Actual Repay Rate ──\n")
print(f"  {'Prob Bucket':>12} {'Count':>8} {'Pred Prob':>10} {'Actual Rate':>12} {'Gap':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")

df_cal2           = pd.DataFrame({"prob": probs_cal, "actual": y_cal_test.values})
df_cal2["bucket"] = pd.qcut(df_cal2["prob"], q=10, duplicates="drop")

gaps_after = []
for bucket, group in df_cal2.groupby("bucket", observed=True):
    pred_prob   = group["prob"].mean()
    actual_rate = group["actual"].mean()
    gap         = actual_rate - pred_prob
    gaps_after.append(abs(gap))
    print(f"  {str(bucket):>12} {len(group):>8,} {pred_prob:>10.3f} "
          f"{actual_rate:>12.3f} {gap:>+8.3f}")

print(f"\n  Mean absolute gap before: {mean_abs_gap:.4f}")
print(f"  Mean absolute gap after:  {np.mean(gaps_after):.4f}")

# ── 9. SAVE CALIBRATED MODEL ──────────────────────────────────────────────────
version = metadata["version"]
cal_path = f"model_registry/calibrated_model_{version}.joblib"
joblib.dump(calibrated_model, cal_path)

# Update latest.json to point to calibrated model
with open("model_registry/latest.json") as f:
    latest = json.load(f)

latest["calibrated_model_path"] = cal_path
with open("model_registry/latest.json", "w") as f:
    json.dump(latest, f, indent=2)

print(f"\nCalibrated model saved: {cal_path}")
print("latest.json updated with calibrated_model_path")
print("\nNext step: update xgboost_predict.py to load calibrated model")