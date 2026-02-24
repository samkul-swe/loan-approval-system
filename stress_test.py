import pandas as pd
from xgboost_predict import decide

# ── APPLICANT PROFILES ────────────────────────────────────────────────────────
# 33% clearly should be APPROVED (strong profiles)
# 33% clearly should be DECLINED (risky profiles)
# 33% ambiguous (borderline — reasonable people could disagree)

applicants = [

    # ── CLEARLY APPROVED ─────────────────────────────────────────────────────
    {
        "label":    "Strong #1 — High income, low debt, long history",
        "expected": "APPROVED",
        "data": {
            "dti": 10.0, "revol_util": 15.0, "pub_rec": 0, "annual_inc": 120000,
            "emp_length": 10, "open_acc": 7, "mort_acc": 2,
            "credit_history_years": 18.0, "revol_bal": 4000, "home_ownership": "MORTGAGE"
        }
    },
    {
        "label":    "Strong #2 — Solid middle income, very low utilisation",
        "expected": "APPROVED",
        "data": {
            "dti": 8.0, "revol_util": 10.0, "pub_rec": 0, "annual_inc": 75000,
            "emp_length": 8, "open_acc": 6, "mort_acc": 1,
            "credit_history_years": 14.0, "revol_bal": 2000, "home_ownership": "MORTGAGE"
        }
    },
    {
        "label":    "Strong #3 — Own home, zero public records, long employment",
        "expected": "APPROVED",
        "data": {
            "dti": 12.0, "revol_util": 22.0, "pub_rec": 0, "annual_inc": 95000,
            "emp_length": 10, "open_acc": 9, "mort_acc": 1,
            "credit_history_years": 20.0, "revol_bal": 5000, "home_ownership": "OWN"
        }
    },
    {
        "label":    "Strong #4 — Very high income, moderate dti",
        "expected": "APPROVED",
        "data": {
            "dti": 18.0, "revol_util": 25.0, "pub_rec": 0, "annual_inc": 200000,
            "emp_length": 10, "open_acc": 12, "mort_acc": 3,
            "credit_history_years": 22.0, "revol_bal": 8000, "home_ownership": "MORTGAGE"
        }
    },

    # ── CLEARLY DECLINED ──────────────────────────────────────────────────────
    {
        "label":    "Risky #1 — High dti, maxed credit, bankruptcy",
        "expected": "DECLINED",
        "data": {
            "dti": 42.0, "revol_util": 90.0, "pub_rec": 3, "annual_inc": 28000,
            "emp_length": 0, "open_acc": 16, "mort_acc": 0,
            "credit_history_years": 2.0, "revol_bal": 22000, "home_ownership": "RENT"
        }
    },
    {
        "label":    "Risky #2 — Very low income, high utilisation, public records",
        "expected": "DECLINED",
        "data": {
            "dti": 38.0, "revol_util": 85.0, "pub_rec": 2, "annual_inc": 22000,
            "emp_length": 1, "open_acc": 14, "mort_acc": 0,
            "credit_history_years": 3.0, "revol_bal": 18000, "home_ownership": "RENT"
        }
    },
    {
        "label":    "Risky #3 — Maxed out credit, short history, renting",
        "expected": "DECLINED",
        "data": {
            "dti": 35.0, "revol_util": 92.0, "pub_rec": 1, "annual_inc": 35000,
            "emp_length": 2, "open_acc": 18, "mort_acc": 0,
            "credit_history_years": 4.0, "revol_bal": 25000, "home_ownership": "RENT"
        }
    },
    {
        "label":    "Risky #4 — High debt burden across the board",
        "expected": "DECLINED",
        "data": {
            "dti": 45.0, "revol_util": 78.0, "pub_rec": 2, "annual_inc": 30000,
            "emp_length": 0, "open_acc": 20, "mort_acc": 0,
            "credit_history_years": 1.5, "revol_bal": 30000, "home_ownership": "RENT"
        }
    },

    # ── AMBIGUOUS ─────────────────────────────────────────────────────────────
    {
        "label":    "Ambiguous #1 — Good income but high utilisation",
        "expected": "AMBIGUOUS",
        "data": {
            "dti": 22.0, "revol_util": 68.0, "pub_rec": 0, "annual_inc": 80000,
            "emp_length": 6, "open_acc": 8, "mort_acc": 1,
            "credit_history_years": 10.0, "revol_bal": 12000, "home_ownership": "MORTGAGE"
        }
    },
    {
        "label":    "Ambiguous #2 — Low income but clean record",
        "expected": "AMBIGUOUS",
        "data": {
            "dti": 25.0, "revol_util": 40.0, "pub_rec": 0, "annual_inc": 38000,
            "emp_length": 4, "open_acc": 5, "mort_acc": 0,
            "credit_history_years": 7.0, "revol_bal": 5000, "home_ownership": "RENT"
        }
    },
    {
        "label":    "Ambiguous #3 — One public record but otherwise solid",
        "expected": "AMBIGUOUS",
        "data": {
            "dti": 20.0, "revol_util": 45.0, "pub_rec": 1, "annual_inc": 70000,
            "emp_length": 7, "open_acc": 7, "mort_acc": 1,
            "credit_history_years": 12.0, "revol_bal": 7000, "home_ownership": "MORTGAGE"
        }
    },
    {
        "label":    "Ambiguous #4 — Short employment but high income",
        "expected": "AMBIGUOUS",
        "data": {
            "dti": 28.0, "revol_util": 55.0, "pub_rec": 0, "annual_inc": 110000,
            "emp_length": 1, "open_acc": 10, "mort_acc": 0,
            "credit_history_years": 8.0, "revol_bal": 9000, "home_ownership": "RENT"
        }
    },
]

# ── RUN DECISIONS ─────────────────────────────────────────────────────────────
results = []
for applicant in applicants:
    result = decide(applicant["data"])
    results.append({
        "label":        applicant["label"],
        "expected":     applicant["expected"],
        "decision":     result["decision"],
        "repay_prob":   result["repay_probability"],
    })

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
df = pd.DataFrame(results)

def outcome(row):
    if row["expected"] == "AMBIGUOUS":
        return "—"
    return "✓ CORRECT" if row["decision"] == row["expected"] else "✗ WRONG"

df["outcome"] = df.apply(outcome, axis=1)

print("\n── Stress Test Results ──\n")
print(df[["label", "expected", "decision", "repay_prob", "outcome"]].to_string(index=False))

# ── SCORE ─────────────────────────────────────────────────────────────────────
clear_cases  = df[df["expected"] != "AMBIGUOUS"]
correct      = (clear_cases["outcome"] == "✓ CORRECT").sum()
total_clear  = len(clear_cases)

print(f"\nClear cases correct: {correct}/{total_clear}")
print(f"Ambiguous cases (no right answer): {len(df[df['expected'] == 'AMBIGUOUS'])}")
print(f"\nRepay probability ranges:")
print(f"  Approved cases:   {df[df['expected']=='APPROVED']['repay_prob'].mean():.3f} avg")
print(f"  Declined cases:   {df[df['expected']=='DECLINED']['repay_prob'].mean():.3f} avg")
print(f"  Ambiguous cases:  {df[df['expected']=='AMBIGUOUS']['repay_prob'].mean():.3f} avg")