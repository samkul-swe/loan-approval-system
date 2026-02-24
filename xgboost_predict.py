import pandas as pd
import numpy as np
import joblib
import json
import shap
from datetime import datetime

# ── 1. LOAD LATEST MODEL ──────────────────────────────────────────────────────
with open("model_registry/latest.json") as f:
    latest = json.load(f)

version      = latest["version"]
model        = joblib.load(latest["model_path"])
preprocessor = joblib.load(latest["preprocessor_path"])

with open(f"model_registry/metadata_{version}.json") as f:
    metadata = json.load(f)

print(f"Loaded model version: {version}")
print(f"Trained ROC-AUC: {metadata['roc_auc']}")

APPROVAL_THRESHOLD = 0.50

OFFERS = {
    "A": {"amount": 500,  "term_months": 6,  "apr": 0.09},
    "B": {"amount": 1000, "term_months": 12, "apr": 0.12},
    "C": {"amount": 2500, "term_months": 24, "apr": 0.15},
    "D": {"amount": 5000, "term_months": 36, "apr": 0.19},
}

def monthly_payment(amount, apr, term_months):
    r = apr / 12
    return amount * r / (1 - (1 + r) ** -term_months)

def select_offer(repay_prob, annual_inc, dti):
    monthly_inc   = annual_inc / 12
    existing_debt = monthly_inc * (dti / 100)
    dti_ceiling   = monthly_inc * 0.40
    headroom      = max(0, dti_ceiling - existing_debt)

    if repay_prob >= 0.85:      max_tier = "D"
    elif repay_prob >= 0.75:    max_tier = "C"
    elif repay_prob >= 0.65:    max_tier = "B"
    else:                       max_tier = "A"

    tier_order = ["A", "B", "C", "D"]
    for tier in reversed(tier_order[:tier_order.index(max_tier) + 1]):
        offer = OFFERS[tier]
        pmt   = monthly_payment(offer["amount"], offer["apr"], offer["term_months"])
        if pmt <= headroom:
            return tier, offer, round(pmt, 2)

    offer = OFFERS["A"]
    return "A", offer, round(monthly_payment(offer["amount"], offer["apr"], offer["term_months"]), 2)

# ── 2. CONFIDENCE NARRATIVE ───────────────────────────────────────────────────
# Human readable templates describing what each feature contributed
POSITIVE_TEMPLATES = {
    "dti":                  "Low debt-to-income ratio of {val:.1f}% — manageable existing debt load",
    "revol_util":           "Low credit utilisation of {val:.1f}% — not over-relying on available credit",
    "pub_rec":              "No derogatory public records — clean financial track record",
    "annual_inc":           "Strong annual income of ${val:,.0f} — good repayment capacity",
    "emp_length":           "Stable employment of {val:.0f} year(s) — consistent income source",
    "open_acc":             "Healthy number of open accounts ({val:.0f}) — manageable credit obligations",
    "mort_acc":             "Mortgage account present — demonstrates prior lending trust",
    "credit_history_years": "Long credit history of {val:.1f} years — established track record",
    "revol_bal":            "Low revolving balance of ${val:,.0f} — limited outstanding credit card debt",
    "home_ownership":       "Home ownership status suggests financial stability",
}

NEGATIVE_TEMPLATES = {
    "dti":                  "High debt-to-income ratio of {val:.1f}% — too much income already committed",
    "revol_util":           "High credit utilisation of {val:.1f}% — over-relying on available credit",
    "pub_rec":              "{val:.0f} derogatory public record(s) — negative financial history",
    "annual_inc":           "Annual income of ${val:,.0f} may be insufficient given existing obligations",
    "emp_length":           "Short employment history of {val:.0f} year(s) — limited income stability",
    "open_acc":             "High number of open accounts ({val:.0f}) — many active credit obligations",
    "mort_acc":             "No mortgage accounts — limited evidence of prior lending trust",
    "credit_history_years": "Short credit history of {val:.1f} years — limited track record",
    "revol_bal":            "High revolving balance of ${val:,.0f} — significant outstanding credit card debt",
    "home_ownership":       "Home ownership status raises some concern",
}

def build_confidence_narrative(shap_vals, feature_names, feature_values, base_log_odds):
    """
    Builds a step-by-step confidence narrative showing how each feature
    moves the model's confidence toward or away from approval.
    Running sum is kept in log-odds space to avoid overflow.
    Only converted to probability (0-100%) at each display step.
    """
    def to_prob(log_odds):
        # Convert log-odds to probability, clamped to 0-100%
        p = 1 / (1 + np.exp(-log_odds))
        return max(0.0, min(1.0, p)) * 100

    # Sort features by absolute SHAP value — most impactful first
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap":    shap_vals,
        "value":   feature_values
    }).reindex(pd.Series(shap_vals).abs().sort_values(ascending=False).index)

    running_log_odds = base_log_odds
    steps            = []
    narrative        = []

    base_pct = to_prob(running_log_odds)
    narrative.append(f"  {'Feature':<25} {'Impact':>8}    {'Confidence':>10}  Note")
    narrative.append(f"  {'-'*25} {'-'*8}    {'-'*10}  {'-'*35}")
    narrative.append(f"  {'Base confidence':<25} {'':>8}    {base_pct:>9.1f}%  Starting point")

    for _, row in shap_df.iterrows():
        feat             = row["feature"]
        shap_val         = row["shap"]   # already in log-odds space
        val              = row["value"]

        prev_prob        = to_prob(running_log_odds)
        running_log_odds += shap_val
        new_prob         = to_prob(running_log_odds)
        impact           = new_prob - prev_prob  # actual % change after clamping

        direction = "✓" if shap_val > 0 else "✗" if shap_val < 0 else "~"

        try:
            if shap_val > 0:
                note = POSITIVE_TEMPLATES.get(feat, f"{feat} contributed positively").format(val=val)
            elif shap_val < 0:
                note = NEGATIVE_TEMPLATES.get(feat, f"{feat} raised concern").format(val=val)
            else:
                note = f"{feat} had negligible impact"
        except:
            note = f"{feat} contributed {'positively' if shap_val > 0 else 'negatively'}"

        narrative.append(
            f"  {direction} {feat:<23} {impact:>+7.1f}%  → {new_prob:>8.1f}%  {note}"
        )

        steps.append({
            "feature":      feat,
            "shap":         round(float(shap_val), 4),
            "impact_pct":   round(impact, 2),
            "running_prob": round(new_prob, 2),
            "direction":    "positive" if shap_val > 0 else "negative" if shap_val < 0 else "neutral",
            "note":         note,
            "value":        round(float(val), 4) if isinstance(val, (int, float, np.floating)) else val
        })

    final_prob = to_prob(running_log_odds)
    narrative.append(f"  {'-'*25} {'-'*8}    {'-'*10}  {'-'*35}")
    narrative.append(f"  {'Final confidence':<25} {'':>8}    {final_prob:>9.1f}%")

    return "\n".join(narrative), steps

# ── 3. MAIN DECISION FUNCTION ─────────────────────────────────────────────────
def decide(applicant: dict) -> dict:
    feature_names = metadata["features"]
    X_input       = pd.DataFrame([applicant])[feature_names]
    X_proc        = preprocessor.transform(X_input)

    repay_prob    = float(model.predict_proba(X_proc)[0][1])

    # SHAP values for this applicant
    explainer     = shap.TreeExplainer(model)
    shap_vals     = explainer.shap_values(X_proc)[0]

    # Build confidence narrative — pass raw log-odds as base, not probability
    narrative_str, narrative_steps = build_confidence_narrative(
        shap_vals, feature_names, X_proc[0], explainer.expected_value
    )

    decision = {
        "applicant":          applicant,
        "repay_probability":  round(repay_prob, 4),
        "model_version":      version,
        "decided_at":         datetime.now().isoformat(),
        "confidence_narrative": narrative_steps,  # structured for audit log
    }

    if repay_prob >= APPROVAL_THRESHOLD:
        tier, offer, pmt = select_offer(repay_prob, applicant["annual_inc"], applicant["dti"])
        decision.update({
            "decision":        "APPROVED",
            "offer_tier":      tier,
            "offer_amount":    offer["amount"],
            "offer_term":      f"{offer['term_months']} months",
            "offer_apr":       f"{offer['apr']*100:.0f}%",
            "monthly_payment": pmt,
        })
    else:
        # Decline reasons = features that pushed most toward default
        decline_steps  = [s for s in narrative_steps if s["direction"] == "negative"]
        decline_reasons = [s["note"] for s in decline_steps[:3]]
        while len(decline_reasons) < 3:
            decline_reasons.append("Overall credit profile did not meet lending criteria.")
        decision.update({
            "decision":        "DECLINED",
            "decline_reasons": decline_reasons,
        })

    # ── PRINT READABLE OUTPUT ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Decision: {decision['decision']}   "
          f"(Repay probability: {repay_prob:.4f})")
    print(f"  Model version: {version}  |  {decision['decided_at']}")
    print(f"{'='*65}")
    print(f"\n  Confidence Narrative:\n")
    print(narrative_str)

    if decision["decision"] == "APPROVED":
        print(f"\n  Offer: {decision['offer_tier']} — "
              f"${decision['offer_amount']} over {decision['offer_term']} "
              f"at {decision['offer_apr']} APR")
        print(f"  Monthly payment: ${decision['monthly_payment']}")
    else:
        print(f"\n  Decline Reasons:")
        for i, r in enumerate(decision["decline_reasons"], 1):
            print(f"  {i}. {r}")

    return decision

# ── 4. EXAMPLE USAGE ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    good_applicant = {
        "dti": 12.5, "revol_util": 18.0, "pub_rec": 0, "annual_inc": 85000,
        "emp_length": 7, "open_acc": 8, "mort_acc": 1,
        "credit_history_years": 12.0, "revol_bal": 3200, "home_ownership": "MORTGAGE"
    }

    risky_applicant = {
        "dti": 38.0, "revol_util": 82.0, "pub_rec": 2, "annual_inc": 32000,
        "emp_length": 1, "open_acc": 14, "mort_acc": 0,
        "credit_history_years": 3.0, "revol_bal": 18000, "home_ownership": "RENT"
    }

    for label, applicant in [("Strong applicant", good_applicant), ("Risky applicant", risky_applicant)]:
        print(f"\n\n{'#'*65}")
        print(f"  {label}")
        print(f"{'#'*65}")
        decide(applicant)