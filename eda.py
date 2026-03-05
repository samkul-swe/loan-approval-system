import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("lending_club_loan_two.csv")
print(f"Shape: {df.shape}")

# ── 2. DROP LEAKY / IRRELEVANT COLUMNS ───────────────────────────────────────
drop_cols = [
    "grade", "sub_grade",       # Lending Club's own underwriting — leakage
    "int_rate",                  # derived from underwriting
    "installment",               # derived from loan amount + rate
    "loan_amnt",                 # decision output, not applicant input
    "initial_list_status",       # marketplace mechanic
    "application_type",          # always Individual for POS lending
    "emp_title",                 # high cardinality free text
    "title",                     # loan purpose free text, noisy
    "zip_code",                  # geography — fair lending risk
    "addr_state",                # geography — fair lending risk
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── 3. TARGET VARIABLE ────────────────────────────────────────────────────────
known_statuses = ["Fully Paid", "Charged Off"]
df = df[df["loan_status"].isin(known_statuses)].copy()
df["target"] = (df["loan_status"] == "Fully Paid").astype(int)
df.drop(columns=["loan_status"], inplace=True)

print(f"\nClass balance:\n{df['target'].value_counts()}")
print(f"Default rate: {(df['target'] == 0).mean():.2%}")

# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────
def parse_date(val, fmt):
    try:
        return datetime.strptime(val, fmt)
    except:
        return np.nan

# Parse dates
df["earliest_cr_line_dt"] = df["earliest_cr_line"].apply(
    lambda x: parse_date(x, "%b-%Y")
)
df["issue_d_dt"] = df["issue_d"].apply(
    lambda x: parse_date(x, "%b-%Y")
)

# credit_history_years — use issue_d as reference date per row
# This avoids the fixed 2020 reference date leakage issue
df["credit_history_years"] = df.apply(
    lambda row: (row["issue_d_dt"] - row["earliest_cr_line_dt"]).days / 365
    if pd.notnull(row["issue_d_dt"]) and pd.notnull(row["earliest_cr_line_dt"])
    else np.nan,
    axis=1
)

# Drop raw date columns — issue_d is post-decision, earliest_cr_line now engineered
df.drop(columns=["earliest_cr_line", "issue_d",
                  "earliest_cr_line_dt", "issue_d_dt"], inplace=True)

# emp_length to numeric
emp_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8,  "9 years": 9, "10+ years": 10
}
df["emp_length"] = df["emp_length"].map(emp_map)

# ── 5. ENGINEERED FEATURES ────────────────────────────────────────────────────
# These encode domain knowledge directly into features
# so the model doesn't have to discover these relationships through splits

# Actual dollar amount of monthly debt payments
# More informative than dti alone — 20% dti on 30k income is very different
# from 20% dti on 200k income
df["debt_burden"] = (df["dti"] / 100) * df["annual_inc"] / 12

# How much of annual income is tied up in revolving debt
# Captures financial stress independent of credit limit
df["credit_utilization_pressure"] = df["revol_bal"] / (df["annual_inc"] + 1)

# Binary flag — even one public record is a meaningful signal
# Separates clean from any negative history
df["has_public_record"] = (df["pub_rec"] > 0).astype(int)

# Thin file flag — limited credit history AND few accounts
# These applicants are harder to assess, not necessarily risky
df["thin_file"] = (
    (df["credit_history_years"] < 5) &
    (df["open_acc"] < 4)
).astype(int)

# Dangerous combination — high debt burden AND low income
# Neither alone is as risky as both together
df["high_dti_low_income"] = (
    (df["dti"] > 30) &
    (df["annual_inc"] < 50000)
).astype(int)

print(f"\n── Engineered Feature Summary ──")
print(f"debt_burden (monthly $):          mean={df['debt_burden'].mean():.0f}, "
      f"median={df['debt_burden'].median():.0f}")
print(f"credit_utilization_pressure:      mean={df['credit_utilization_pressure'].mean():.3f}, "
      f"median={df['credit_utilization_pressure'].median():.3f}")
print(f"has_public_record = 1:            {df['has_public_record'].mean():.1%} of applicants")
print(f"thin_file = 1:                    {df['thin_file'].mean():.1%} of applicants")
print(f"high_dti_low_income = 1:          {df['high_dti_low_income'].mean():.1%} of applicants")

# ── 6. DEFINE FEATURE SETS ────────────────────────────────────────────────────
strong_features = ["dti", "revol_util", "pub_rec", "annual_inc"]
engineered_features = [
    "debt_burden", "credit_utilization_pressure",
    "has_public_record", "thin_file", "high_dti_low_income"
]
supporting_features = [
    "emp_length", "home_ownership",
    "open_acc", "mort_acc", "credit_history_years", "revol_bal"
]
cat_features = ["home_ownership"]
num_features = [f for f in strong_features + engineered_features + supporting_features
                if f not in cat_features]

# ── 7. MISSING VALUES ─────────────────────────────────────────────────────────
print("\n── Missing Values ──")
all_feats = strong_features + engineered_features + supporting_features
missing     = df[all_feats].isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"missing_count": missing, "missing_%": missing_pct}))

# ── 8. CLASS BALANCE PLOT ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
df["target"].value_counts().plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
ax.set_xticklabels(["Defaulted (0)", "Repaid (1)"], rotation=0)
ax.set_title("Class Balance")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("class_balance.png", dpi=120)
plt.close()
print("Saved: class_balance.png")

# ── 9. DISTRIBUTIONS OF STRONG + ENGINEERED FEATURES BY TARGET ───────────────
plot_features = strong_features + engineered_features
n_cols = 3
n_rows = (len(plot_features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()

for i, feat in enumerate(plot_features):
    ax = axes[i]
    for label, color, name in [(0, "#e74c3c", "Defaulted"), (1, "#2ecc71", "Repaid")]:
        vals = df[df["target"] == label][feat].dropna()
        ax.hist(vals, bins=50, alpha=0.5, color=color, label=name, density=True)
    ax.set_title(f"{feat}")
    ax.legend()

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Strong + Engineered Feature Distributions: Repaid vs Defaulted", fontsize=14)
plt.tight_layout()
plt.savefig("strong_feature_distributions.png", dpi=120)
plt.close()
print("Saved: strong_feature_distributions.png")

# ── 10. DISTRIBUTIONS OF SUPPORTING FEATURES BY TARGET ───────────────────────
num_supporting = [f for f in supporting_features if f not in cat_features]
n_rows = (len(num_supporting) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()

for i, feat in enumerate(num_supporting):
    ax = axes[i]
    for label, color, name in [(0, "#e74c3c", "Defaulted"), (1, "#2ecc71", "Repaid")]:
        vals = df[df["target"] == label][feat].dropna()
        ax.hist(vals, bins=50, alpha=0.5, color=color, label=name, density=True)
    ax.set_title(f"{feat}")
    ax.legend()

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Supporting Feature Distributions: Repaid vs Defaulted", fontsize=14)
plt.tight_layout()
plt.savefig("supporting_feature_distributions.png", dpi=120)
plt.close()
print("Saved: supporting_feature_distributions.png")

# ── 11. CATEGORICAL FEATURES BY TARGET ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ct     = df.groupby(["home_ownership", "target"]).size().unstack(fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0)
ct_pct.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
ax.set_title("home_ownership — Default Rate by Category")
ax.set_ylabel("Proportion")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(["Defaulted", "Repaid"])
plt.tight_layout()
plt.savefig("categorical_features.png", dpi=120)
plt.close()
print("Saved: categorical_features.png")

# ── 12. CORRELATION MATRIX ────────────────────────────────────────────────────
corr_matrix = df[num_features + ["target"]].corr()

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, ax=ax, linewidths=0.5
)
ax.set_title("Feature Correlation Matrix (including engineered features)")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=120)
plt.close()
print("Saved: correlation_matrix.png")

# ── 13. SUMMARY STATS BY TARGET ───────────────────────────────────────────────
print("\n── Mean values: Repaid vs Defaulted ──")
print(df.groupby("target")[num_features].mean().T.rename(
    columns={0: "Defaulted", 1: "Repaid"}
))

# ── 14. SAVE CLEAN DATASET ────────────────────────────────────────────────────
df.to_csv("lending_club_clean.csv", index=False)
print("\nClean dataset saved to lending_club_clean.csv")
print(f"Final shape: {df.shape}")
print(f"\nFeatures saved:")
print(f"  Strong:     {strong_features}")
print(f"  Engineered: {engineered_features}")
print(f"  Supporting: {supporting_features}")