import pandas as pd
import numpy as np
import joblib
import json
import shap
from datetime import datetime

# ── 1. LOAD LATEST MODEL ──────────────────────────────────────────────────────
# Always loads whatever version is marked as latest.
# To roll back: change "version" in model_registry/latest.json to any past version.
with open("model_registry/latest.json") as f:
    latest = json.load(f)

version      = latest["version"]
model        = joblib.load(latest["model_path"])
preprocessor = joblib.load(latest["preprocessor_path"])

with open(f"model_registry/metadata_{version}.json") as f:
    metadata = json.load(f)

print(f"Loaded model version: {version}")
print(f"Trained ROC-AUC: {metadata['roc_auc']}")

# ── 2. APPROVAL THRESHOLD ─────────────────────────────────────────────────────
# predict_proba returns probability of REPAYING (class 1).
# We approve if repay probability >= threshold.
# 0.6 means "I need at least 60% confidence they'll repay before approving."
# Higher threshold = more conservative = fewer approvals but fewer defaults.
# Lower threshold = more approvals but more defaults.
# Tune this based on business feedback after deployment.
APPROVAL_THRESHOLD = 0.5

# ── 3. OFFER SELECTION LOGIC ─────────────────────────────────────────────────
# Offers available to approved applicants.
# Selection is based on repay probability + monthly payment affordability.
OFFERS = {
    "A": {"amount": 500,  "term_months": 6,  "apr": 0.09},
    "B": {"amount": 1000, "term_months": 12, "apr": 0.12},
    "C": {"amount": 2500, "term_months": 24, "apr": 0.15},
    "D": {"amount": 5000, "term_months": 36, "apr": 0.19},
}

def monthly_payment(amount, apr, term_months):
    """Standard amortisation formula."""
    r = apr / 12
    return amount * r / (1 - (1 + r) ** -term_months)

def select_offer(repay_prob, annual_inc, dti):
    """
    Pick the best offer the applicant can afford and is likely to repay.
    Logic:
    - Compute monthly income headroom after existing debt obligations
    - Find the highest offer whose monthly payment fits within that headroom
    - Further constrain by repay probability — lower confidence = smaller offer
    """
    monthly_inc     = annual_inc / 12
    existing_debt   = monthly_inc * (dti / 100)
    # Allow up to 40% DTI total — headroom is what's left before hitting that ceiling
    dti_ceiling     = monthly_inc * 0.40
    headroom        = max(0, dti_ceiling - existing_debt)

    # Probability bands map to maximum offer tier
    if repay_prob >= 0.85:
        max_tier = "D"
    elif repay_prob >= 0.75:
        max_tier = "C"
    elif repay_prob >= 0.65:
        max_tier = "B"
    else:
        max_tier = "A"

    tier_order = ["A", "B", "C", "D"]
    max_idx    = tier_order.index(max_tier)

    # Walk down from max allowed tier, pick highest that fits in headroom
    for tier in reversed(tier_order[:max_idx + 1]):
        offer = OFFERS[tier]
        pmt   = monthly_payment(offer["amount"], offer["apr"], offer["term_months"])
        if pmt <= headroom:
            return tier, offer, round(pmt, 2)

    # If nothing fits, return smallest offer anyway with a note
    offer = OFFERS["A"]
    pmt   = monthly_payment(offer["amount"], offer["apr"], offer["term_months"])
    return "A", offer, round(pmt, 2)

# ── 4. DECLINE REASONS ────────────────────────────────────────────────────────
# Uses SHAP values to find which features pushed this applicant toward default.
# Returns the top 3 most impactful negative factors in human-readable form.

REASON_TEMPLATES = {
    "dti":                  "Your debt-to-income ratio of {val:.1f}% is too high — too much of your income is already committed to existing debt.",
    "revol_util":           "Your revolving credit utilisation of {val:.1f}% is too high — you are using too much of your available credit limit.",
    "pub_rec":              "You have {val:.0f} derogatory public record(s) on file, such as liens or judgements.",
    "annual_inc":           "Your annual income of ${val:,.0f} is insufficient to support this loan given your existing obligations.",
    "emp_length":           "Your employment history of {val:.0f} year(s) is too short to demonstrate sufficient income stability.",
    "open_acc":             "You have {val:.0f} open credit accounts, suggesting a high level of active credit obligations.",
    "mort_acc":             "Your mortgage account profile does not meet our current lending criteria.",
    "credit_history_years": "Your credit history of {val:.1f} years is too short to assess repayment reliability.",
    "revol_bal":            "Your revolving balance of ${val:,.0f} indicates a high level of outstanding credit card debt.",
    "home_ownership":       "Your home ownership status does not meet our current lending criteria.",
    "verification_status":  "We were unable to verify your income, which increases uncertainty in our assessment.",
}

def get_decline_reasons(shap_vals, feature_names, applicant_row, top_n=3):
    """
    shap_vals: 1D array of SHAP values for this applicant (one per feature)
    Negative SHAP = pushed toward default = reason for decline
    """
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap":    shap_vals,
        "value":   applicant_row
    })

    # Only keep features that pushed toward default (negative SHAP)
    negative = shap_df[shap_df["shap"] < 0].sort_values("shap")

    reasons = []
    for _, row in negative.head(top_n).iterrows():
        feat = row["feature"]
        val  = row["value"]
        if feat in REASON_TEMPLATES:
            try:
                reasons.append(REASON_TEMPLATES[feat].format(val=val))
            except:
                reasons.append(f"{feat} did not meet our lending criteria.")
        else:
            reasons.append(f"{feat} did not meet our lending criteria.")

    # Guarantee at least 3 reasons even if fewer negative SHAP features
    while len(reasons) < 3:
        reasons.append("Your overall credit profile does not meet our current lending criteria.")

    return reasons[:3]

# ── 5. MAIN DECISION FUNCTION ─────────────────────────────────────────────────
def decide(applicant: dict) -> dict:
    """
    Given a new applicant's data as a dict, return a full decision:
    - approved / declined
    - if approved: which offer
    - if declined: 3 human-readable reasons
    - model version used (for audit)
    - timestamp
    """
    feature_names = metadata["features"]
    X_input       = pd.DataFrame([applicant])[feature_names]
    X_proc        = preprocessor.transform(X_input)

    repay_prob    = float(model.predict_proba(X_proc)[0][1])

    # SHAP explanation for this applicant
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_proc)[0]

    decision = {
        "applicant":      applicant,
        "repay_probability": round(repay_prob, 4),
        "model_version":  version,
        "decided_at":     datetime.now().isoformat(),
    }

    if repay_prob >= APPROVAL_THRESHOLD:
        tier, offer, pmt = select_offer(
            repay_prob,
            applicant["annual_inc"],
            applicant["dti"]
        )
        decision.update({
            "decision":        "APPROVED",
            "offer_tier":      tier,
            "offer_amount":    offer["amount"],
            "offer_term":      f"{offer['term_months']} months",
            "offer_apr":       f"{offer['apr']*100:.0f}%",
            "monthly_payment": pmt,
        })
    else:
        reasons = get_decline_reasons(shap_vals, feature_names, X_proc[0])
        decision.update({
            "decision":        "DECLINED",
            "decline_reasons": reasons,
        })

    return decision

# ── 6. EXAMPLE USAGE ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example 1: Strong applicant
    good_applicant = {
        "dti":                   12.5,
        "revol_util":            18.0,
        "pub_rec":               0,
        "annual_inc":            85000,
        "emp_length":            7,
        "open_acc":              8,
        "mort_acc":              1,
        "credit_history_years":  12.0,
        "revol_bal":             3200,
        "home_ownership":        "MORTGAGE",
        "verification_status":   "Verified",
    }

    # Example 2: Risky applicant
    risky_applicant = {
        "dti":                   38.0,
        "revol_util":            82.0,
        "pub_rec":               2,
        "annual_inc":            32000,
        "emp_length":            1,
        "open_acc":              14,
        "mort_acc":              0,
        "credit_history_years":  3.0,
        "revol_bal":             18000,
        "home_ownership":        "RENT",
        "verification_status":   "Not Verified",
    }

    for label, applicant in [("Strong applicant", good_applicant), ("Risky applicant", risky_applicant)]:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
        result = decide(applicant)
        print(f"Decision:          {result['decision']}")
        print(f"Repay probability: {result['repay_probability']}")
        print(f"Model version:     {result['model_version']}")

        if result["decision"] == "APPROVED":
            print(f"Offer:             {result['offer_tier']} — ${result['offer_amount']} "
                  f"over {result['offer_term']} at {result['offer_apr']} APR")
            print(f"Monthly payment:   ${result['monthly_payment']}")
        else:
            print("Decline reasons:")
            for i, r in enumerate(result["decline_reasons"], 1):
                print(f"  {i}. {r}")