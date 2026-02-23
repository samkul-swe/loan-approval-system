import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no rendering, just saving
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("lending_club_loan_two.csv")
print(f"Shape: {df.shape}")
print(df.dtypes)

# ── 2. DROP LEAKY / IRRELEVANT COLUMNS ───────────────────────────────────────
drop_cols = [
    "grade", "sub_grade",       # Lending Club's own underwriting — leakage
    "int_rate",                  # derived from underwriting
    "installment",               # derived from loan amount + rate
    "loan_amnt",                 # decision output, not applicant input
    "issue_d",                   # post-decision date
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
df["target"] = (df["loan_status"] == "Fully Paid").astype(int)  # 1=repaid, 0=defaulted
df.drop(columns=["loan_status"], inplace=True)

print(f"\nClass balance:\n{df['target'].value_counts()}")
print(f"Default rate: {(df['target'] == 0).mean():.2%}")

# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────
def parse_credit_date(val):
    try:
        return datetime.strptime(val, "%b-%Y")
    except:
        return np.nan

ref_date = datetime(2020, 1, 1)
df["earliest_cr_line"] = df["earliest_cr_line"].apply(parse_credit_date)
df["credit_history_years"] = df["earliest_cr_line"].apply(
    lambda x: (ref_date - x).days / 365 if pd.notnull(x) else np.nan
)
df.drop(columns=["earliest_cr_line"], inplace=True)

emp_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8,  "9 years": 9, "10+ years": 10
}
df["emp_length"] = df["emp_length"].map(emp_map)

# ── 5. DEFINE FEATURE SETS ────────────────────────────────────────────────────
strong_features  = ["dti", "revol_util", "pub_rec", "annual_inc"]
supporting_features = [
    "emp_length", "home_ownership", "verification_status",
    "open_acc", "mort_acc", "credit_history_years", "revol_bal"
]
cat_features = ["home_ownership", "verification_status"]
num_features = [f for f in strong_features + supporting_features if f not in cat_features]

# ── 6. MISSING VALUES ─────────────────────────────────────────────────────────
print("\n── Missing Values ──")
missing     = df[strong_features + supporting_features].isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"missing_count": missing, "missing_%": missing_pct}))

# ── 7. CLASS BALANCE PLOT ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
df["target"].value_counts().plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
ax.set_xticklabels(["Defaulted (0)", "Repaid (1)"], rotation=0)
ax.set_title("Class Balance")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("class_balance.png", dpi=120)
plt.close()
print("Saved: class_balance.png")

# ── 8. DISTRIBUTIONS OF STRONG FEATURES BY TARGET ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(strong_features):
    ax = axes[i]
    for label, color, name in [(0, "#e74c3c", "Defaulted"), (1, "#2ecc71", "Repaid")]:
        vals = df[df["target"] == label][feat].dropna()
        ax.hist(vals, bins=50, alpha=0.5, color=color, label=name, density=True)
    ax.set_title(f"{feat} distribution")
    ax.legend()

axes[-1].set_visible(False)
plt.suptitle("Strong Feature Distributions: Repaid vs Defaulted", fontsize=14)
plt.tight_layout()
plt.savefig("strong_feature_distributions.png", dpi=120)
plt.close()
print("Saved: strong_feature_distributions.png")

# ── 9. DISTRIBUTIONS OF SUPPORTING FEATURES BY TARGET ────────────────────────
num_supporting = [f for f in supporting_features if f not in cat_features]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(num_supporting):
    ax = axes[i]
    for label, color, name in [(0, "#e74c3c", "Defaulted"), (1, "#2ecc71", "Repaid")]:
        vals = df[df["target"] == label][feat].dropna()
        ax.hist(vals, bins=50, alpha=0.5, color=color, label=name, density=True)
    ax.set_title(f"{feat} distribution")
    ax.legend()

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Supporting Feature Distributions: Repaid vs Defaulted", fontsize=14)
plt.tight_layout()
plt.savefig("supporting_feature_distributions.png", dpi=120)
plt.close()
print("Saved: supporting_feature_distributions.png")

# ── 10. CATEGORICAL FEATURES BY TARGET ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, feat in zip(axes, cat_features):
    ct     = df.groupby([feat, "target"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    ct_pct.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
    ax.set_title(f"{feat} — Default Rate by Category")
    ax.set_ylabel("Proportion")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(["Defaulted", "Repaid"])

plt.tight_layout()
plt.savefig("categorical_features.png", dpi=120)
plt.close()
print("Saved: categorical_features.png")

# ── 11. CORRELATION MATRIX ────────────────────────────────────────────────────
corr_matrix = df[num_features + ["target"]].corr()

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, ax=ax, linewidths=0.5
)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=120)
plt.close()
print("Saved: correlation_matrix.png")

# ── 12. SUMMARY STATS BY TARGET ───────────────────────────────────────────────
print("\n── Mean values: Repaid vs Defaulted ──")
print(df.groupby("target")[num_features].mean().T.rename(columns={0: "Defaulted", 1: "Repaid"}))

# ── 13. SAVE CLEAN DATASET ────────────────────────────────────────────────────
df.to_csv("lending_club_clean.csv", index=False)
print("\nClean dataset saved to lending_club_clean.csv")
print(f"Final shape: {df.shape}")