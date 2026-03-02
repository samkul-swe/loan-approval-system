import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json

# ── 1. LOAD ORIGINAL DATASET WITH GRADE ──────────────────────────────────────
# We need the original dataset which has the grade column
# lending_club_clean.csv dropped it, so we load the raw file
df_raw = pd.read_csv("lending_club_loan_two.csv")

# Keep only rows with known outcomes
known_statuses = ["Fully Paid", "Charged Off"]
df_raw = df_raw[df_raw["loan_status"].isin(known_statuses)].copy()
df_raw["target"] = (df_raw["loan_status"] == "Fully Paid").astype(int)

print(f"Total rows with known outcome: {len(df_raw):,}")
print(f"Grade distribution:\n{df_raw['grade'].value_counts().sort_index()}\n")

# ── 2. LOAD MODEL ─────────────────────────────────────────────────────────────
with open("model_registry/latest.json") as f:
    latest = json.load(f)

model        = joblib.load(latest["model_path"])
preprocessor = joblib.load(latest["preprocessor_path"])

with open(f"model_registry/metadata_{latest['version']}.json") as f:
    metadata = json.load(f)

# ── 3. PREPARE FEATURES ───────────────────────────────────────────────────────
from datetime import datetime
import numpy as np

strong_num   = ["dti", "revol_util", "pub_rec", "annual_inc"]
support_num  = ["emp_length", "open_acc", "mort_acc", "credit_history_years", "revol_bal"]
cat_features = ["home_ownership"]
all_num      = strong_num + support_num
all_features = all_num + cat_features

# Engineer credit_history_years from earliest_cr_line
def parse_credit_date(val):
    try:
        return datetime.strptime(val, "%b-%Y")
    except:
        return np.nan

ref_date = datetime(2020, 1, 1)
df_raw["earliest_cr_line"] = df_raw["earliest_cr_line"].apply(parse_credit_date)
df_raw["credit_history_years"] = df_raw["earliest_cr_line"].apply(
    lambda x: (ref_date - x).days / 365 if pd.notnull(x) else np.nan
)

# Convert emp_length to numeric
emp_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8,  "9 years": 9, "10+ years": 10
}
df_raw["emp_length"] = df_raw["emp_length"].map(emp_map)

# ── 4. SCORE ALL APPLICANTS ───────────────────────────────────────────────────
print("Scoring all applicants...")
X         = df_raw[all_features]
X_proc    = preprocessor.transform(X)
probs     = model.predict_proba(X_proc)[:, 1]

df_raw["repay_prob"] = probs
df_raw["approved"]   = (probs >= 0.50).astype(int)

# ── 5. APPROVAL RATE BY GRADE ─────────────────────────────────────────────────
print("── Approval Rate by Grade ──\n")
print(f"  {'Grade':>6} {'Total':>8} {'Approved':>10} {'Approval%':>11} "
      f"{'Actual Repay%':>15} {'Default Rate':>13}")
print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*11} {'-'*15} {'-'*13}")

grade_stats = []
for grade in ["A", "B", "C", "D", "E", "F", "G"]:
    subset       = df_raw[df_raw["grade"] == grade]
    if len(subset) == 0:
        continue
    total        = len(subset)
    approved     = subset["approved"].sum()
    approval_pct = approved / total * 100
    repay_rate   = subset["target"].mean() * 100
    default_rate = 100 - repay_rate

    grade_stats.append({
        "grade": grade, "total": total,
        "approved": approved, "approval_pct": approval_pct,
        "repay_rate": repay_rate, "default_rate": default_rate
    })
    print(f"  {grade:>6} {total:>8,} {approved:>10,} {approval_pct:>10.1f}% "
          f"{repay_rate:>14.1f}% {default_rate:>12.1f}%")

# ── 6. WHAT IS THE MODEL DECLINING WITHIN EACH GRADE ─────────────────────────
print(f"\n── Among Declined: Actual Repay Rate by Grade ──\n")
print(f"  {'Grade':>6} {'Declined':>9} {'Actually Repaid':>16} {'Repay Rate':>12}")
print(f"  {'-'*6} {'-'*9} {'-'*16} {'-'*12}")

for grade in ["A", "B", "C", "D", "E", "F", "G"]:
    subset   = df_raw[(df_raw["grade"] == grade) & (df_raw["approved"] == 0)]
    if len(subset) == 0:
        continue
    declined    = len(subset)
    act_repaid  = subset["target"].sum()
    repay_rate  = subset["target"].mean() * 100
    print(f"  {grade:>6} {declined:>9,} {act_repaid:>16,} {repay_rate:>11.1f}%")

# ── 7. PLOT ───────────────────────────────────────────────────────────────────
stats_df = pd.DataFrame(grade_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Approval rate by grade
axes[0].bar(stats_df["grade"], stats_df["approval_pct"],
            color=["#2ecc71" if g in ["A","B","C"] else
                   "#f39c12" if g in ["D","E"] else "#e74c3c"
                   for g in stats_df["grade"]])
axes[0].set_title("Our Model — Approval Rate by Grade")
axes[0].set_xlabel("Lending Club Grade")
axes[0].set_ylabel("Approval Rate %")
axes[0].set_ylim(0, 100)
for i, row in stats_df.iterrows():
    axes[0].text(i, row["approval_pct"] + 1, f"{row['approval_pct']:.0f}%",
                 ha="center", fontsize=9)

# Actual repay rate by grade
axes[1].bar(stats_df["grade"], stats_df["repay_rate"],
            color=["#2ecc71" if g in ["A","B","C"] else
                   "#f39c12" if g in ["D","E"] else "#e74c3c"
                   for g in stats_df["grade"]])
axes[1].set_title("Actual Repayment Rate by Grade")
axes[1].set_xlabel("Lending Club Grade")
axes[1].set_ylabel("Repay Rate %")
axes[1].set_ylim(0, 100)
for i, row in stats_df.iterrows():
    axes[1].text(i, row["repay_rate"] + 1, f"{row['repay_rate']:.0f}%",
                 ha="center", fontsize=9)

plt.suptitle("Grade Validation — Does Our Model Agree with Lending Club?", fontsize=13)
plt.tight_layout()
plt.savefig("grade_validation.png", dpi=120)
plt.close()
print("\nSaved: grade_validation.png")