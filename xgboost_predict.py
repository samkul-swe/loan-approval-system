import pandas as pd
import numpy as np
import joblib
import json
import shap
import uuid
from datetime import datetime
from borderline_budget import try_approve_borderline, print_budget_status

# ── 1. LOAD MODEL ─────────────────────────────────────────────────────────────
with open("model_registry/latest.json") as f:
    latest = json.load(f)

version      = latest["version"]
preprocessor = joblib.load(latest["preprocessor_path"])

# Load calibrated model if available, fall back to raw model
if "calibrated_model_path" in latest:
    model = joblib.load(latest["calibrated_model_path"])
    print(f"Loaded calibrated model version: {version}")
else:
    model = joblib.load(latest["model_path"])
    print(f"Loaded raw model version: {version} (no calibrated model found)")

with open(f"model_registry/metadata_{version}.json") as f:
    metadata = json.load(f)

# ── 2. FEATURE SETUP ──────────────────────────────────────────────────────────
strong_num   = ["dti", "revol_util", "pub_rec", "annual_inc"]
engineered   = ["debt_burden", "credit_utilization_pressure",
                "has_public_record", "high_dti_low_income"]
support_num  = ["emp_length", "open_acc", "mort_acc", "credit_history_years", "revol_bal"]
cat_features = ["home_ownership"]
all_num      = strong_num + engineered + support_num
all_features = all_num + cat_features

# ── 3. THRESHOLDS ─────────────────────────────────────────────────────────────
APPROVAL_THRESHOLD       = 0.75
REPAID_MODEL_THRESHOLD   = 0.50
DISTANCE_TIER2_THRESHOLD = 0.75
DISTANCE_TIER3_THRESHOLD = 0.40

# ── 4. INPUT VALIDATION ───────────────────────────────────────────────────────
REQUIRED_FIELDS = [
    "dti", "revol_util", "pub_rec", "annual_inc", "debt_burden",
    "credit_utilization_pressure", "has_public_record", "high_dti_low_income",
    "emp_length", "open_acc", "mort_acc", "credit_history_years",
    "revol_bal", "home_ownership"
]

VALID_RANGES = {
    "dti":                          (0, 200),
    "revol_util":                   (0, 200),
    "pub_rec":                      (0, 50),
    "annual_inc":                   (1000, 10000000),
    "emp_length":                   (0, 10),
    "open_acc":                     (0, 100),
    "mort_acc":                     (0, 50),
    "credit_history_years":         (0, 100),
    "revol_bal":                    (0, 10000000),
    "debt_burden":                  (0, 1000000),
    "credit_utilization_pressure":  (0, 100),
    "has_public_record":            (0, 1),
    "high_dti_low_income":          (0, 1),
}

VALID_CATEGORIES = {
    "home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"]
}

def validate_applicant(applicant: dict) -> list:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in applicant or applicant[field] is None:
            errors.append(f"Missing required field: {field}")
    for field, (min_val, max_val) in VALID_RANGES.items():
        if field in applicant and applicant[field] is not None:
            try:
                val = float(applicant[field])
                if val < min_val or val > max_val:
                    errors.append(f"{field} value {val} is outside valid range [{min_val}, {max_val}]")
            except (TypeError, ValueError):
                errors.append(f"{field} must be a number, got: {applicant[field]}")
    for field, valid_values in VALID_CATEGORIES.items():
        if field in applicant and applicant[field] is not None:
            if str(applicant[field]).upper() not in valid_values:
                errors.append(f"{field} must be one of {valid_values}, got: {applicant[field]}")
    return errors

# ── 4. KNOWN DEFAULTER AVERAGES ───────────────────────────────────────────────
# From our data analysis — average feature values of confirmed defaulters
# Used in Layer 2 to measure how far a medium applicant is from the bad profile
DEFAULTER_PROFILE = {
    "dti":                  21.74,
    "revol_util":           63.30,
    "pub_rec":               0.28,
    "annual_inc":        57000.00,
    "emp_length":            5.68,
    "open_acc":             11.53,
    "mort_acc":              1.19,
    "credit_history_years": 20.50,
    "revol_bal":         14917.00,
}

# Feature weights — strong features carry more weight in the distance score
FEATURE_WEIGHTS = {
    "dti":                  0.25,   # strong
    "revol_util":           0.25,   # strong
    "annual_inc":           0.20,   # strong
    "pub_rec":              0.08,
    "emp_length":           0.05,
    "open_acc":             0.04,
    "mort_acc":             0.05,
    "credit_history_years": 0.04,
    "revol_bal":            0.04,
}

# ── 5. OFFERS ─────────────────────────────────────────────────────────────────
OFFERS = {
    "A": {"amount": 500,  "term_months": 6,  "apr": 0.09},
    "B": {"amount": 1000, "term_months": 12, "apr": 0.12},
    "C": {"amount": 2500, "term_months": 24, "apr": 0.15},
    "D": {"amount": 5000, "term_months": 36, "apr": 0.19},
}

def monthly_payment(amount, apr, term_months):
    r = apr / 12
    return round(amount * r / (1 - (1 + r) ** -term_months), 2)

# ── 6. LAYER 2 — DISTANCE FROM DEFAULTER ─────────────────────────────────────
def compute_distance_from_defaulter(applicant: dict) -> tuple:
    """
    For each numerical feature, compute how much better the applicant
    is compared to the average defaulter. Aggregate into a weighted score.

    Returns:
        distance_score: float between 0 and 1 (higher = further from defaulter)
        feature_distances: dict showing per-feature improvement
    """
    feature_distances = {}

    for feat, default_val in DEFAULTER_PROFILE.items():
        applicant_val = applicant.get(feat, default_val)
        weight        = FEATURE_WEIGHTS.get(feat, 0.04)

        # For features where lower is better (dti, revol_util, pub_rec, open_acc, revol_bal)
        # positive distance means applicant is lower than defaulter = good
        if feat in ["dti", "revol_util", "pub_rec", "open_acc", "revol_bal"]:
            if default_val > 0:
                raw_dist = (default_val - applicant_val) / default_val
            else:
                raw_dist = 0.0

        # For features where higher is better (annual_inc, emp_length, mort_acc, credit_history)
        else:
            if default_val > 0:
                raw_dist = (applicant_val - default_val) / default_val
            else:
                raw_dist = 0.0

        # Clamp to -1 to 1 range before weighting
        clamped = max(-1.0, min(1.0, raw_dist))
        feature_distances[feat] = {
            "applicant_val":  applicant_val,
            "defaulter_avg":  default_val,
            "raw_distance":   round(clamped, 4),
            "weighted":       round(clamped * weight, 4),
            "better":         clamped > 0
        }

    # Aggregate weighted distances
    raw_score = sum(v["weighted"] for v in feature_distances.values())

    # Normalize to 0-1 range (raw_score ranges from -1 to 1)
    distance_score = (raw_score + 1) / 2

    return round(distance_score, 4), feature_distances

# ── 7. LAYER 3 — OFFER SIZING ─────────────────────────────────────────────────
def select_offer(tier: int, repay_prob: float,
                 distance_score: float,
                 annual_inc: float, dti: float) -> tuple:
    """
    Select offer based on tier, confidence and affordability.
    Tier 1 (clear good)   → A, B, C or D
    Tier 2 (medium good)  → B or C
    Tier 3 (medium okay)  → A only (via budget tracker)
    """
    monthly_inc   = annual_inc / 12
    existing_debt = monthly_inc * (dti / 100)
    headroom      = max(0, monthly_inc * 0.40 - existing_debt)

    if tier == 1:
        if repay_prob >= 0.85:      candidates = ["D", "C", "B", "A"]
        elif repay_prob >= 0.75:    candidates = ["C", "B", "A"]
        elif repay_prob >= 0.65:    candidates = ["B", "A"]
        else:                       candidates = ["A"]
    elif tier == 2:
        if distance_score >= 0.85:  candidates = ["C", "B"]
        else:                       candidates = ["B", "A"]
    else:
        candidates = ["A"]

    for tier_label in candidates:
        offer = OFFERS[tier_label]
        pmt   = monthly_payment(offer["amount"], offer["apr"], offer["term_months"])
        if pmt <= headroom:
            return tier_label, offer, pmt

    # Fallback to smallest offer
    offer = OFFERS["A"]
    return "A", offer, monthly_payment(offer["amount"], offer["apr"], offer["term_months"])

# ── 8. CONFIDENCE NARRATIVE TEMPLATES ────────────────────────────────────────
POSITIVE_TEMPLATES = {
    "dti":                          "Low debt-to-income ratio of {val:.1f}% — manageable existing debt load",
    "revol_util":                   "Low credit utilisation of {val:.1f}% — not over-relying on available credit",
    "pub_rec":                      "No derogatory public records — clean financial track record",
    "annual_inc":                   "Strong annual income of ${val:,.0f} — good repayment capacity",
    "emp_length":                   "Stable employment of {val:.0f} year(s) — consistent income source",
    "open_acc":                     "Healthy number of open accounts ({val:.0f}) — manageable credit obligations",
    "mort_acc":                     "You have {val:.0f} mortgage account(s) — demonstrates prior lending trust",
    "credit_history_years":         "Long credit history of {val:.1f} years — established track record",
    "revol_bal":                    "Low revolving balance of ${val:,.0f} — limited outstanding credit card debt",
    "home_ownership":               "Home ownership status suggests financial stability",
    "debt_burden":                  "Monthly debt burden of ${val:,.0f} is manageable relative to income",
    "credit_utilization_pressure":  "Revolving debt is {val:.1%} of annual income — not over-extended",
    "has_public_record":            "No derogatory public records on file — clean financial history",
    "high_dti_low_income":          "Debt load is reasonable given income level",
}

NEGATIVE_TEMPLATES = {
    "dti":                          "High debt-to-income ratio of {val:.1f}% — too much income already committed",
    "revol_util":                   "High credit utilisation of {val:.1f}% — over-relying on available credit",
    "pub_rec":                      "{val:.0f} derogatory public record(s) — negative financial history",
    "annual_inc":                   "Annual income of ${val:,.0f} may be insufficient given existing obligations",
    "emp_length":                   "Short employment history of {val:.0f} year(s) — limited income stability" if True else "",
    "open_acc":                     "High number of open accounts ({val:.0f}) — many active credit obligations",
    "mort_acc":                     "No mortgage accounts — limited evidence of prior lending trust",
    "credit_history_years":         "Short credit history of {val:.1f} years — limited track record",
    "revol_bal":                    "High revolving balance of ${val:,.0f} — significant outstanding credit card debt",
    "home_ownership":               "Home ownership status raises some concern",
    "debt_burden":                  "Monthly debt burden of ${val:,.0f} is high relative to income",
    "credit_utilization_pressure":  "Revolving debt is {val:.1%} of annual income — over-extended on credit",
    "has_public_record":            "Has at least one derogatory public record — negative financial history",
    "high_dti_low_income":          "Combination of high debt load and low income is a risk signal",
}

def build_confidence_narrative(shap_vals, feature_names, feature_values, base_log_odds):
    def to_prob(log_odds):
        p = 1 / (1 + np.exp(-log_odds))
        return max(0.0, min(1.0, p)) * 100

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
        shap_val         = row["shap"]
        val              = row["value"]
        prev_prob        = to_prob(running_log_odds)
        running_log_odds += shap_val
        new_prob         = to_prob(running_log_odds)
        impact           = new_prob - prev_prob
        direction        = "✓" if shap_val > 0 else "✗" if shap_val < 0 else "~"

        try:
            if shap_val > 0:
                if feat == "emp_length" and float(val) == 0:
                    note = "No recorded employment history — income stability unknown"
                elif feat == "credit_utilization_pressure":
                    note = f"Revolving debt is {float(val):.1%} of annual income — not over-extended"
                else:
                    note = POSITIVE_TEMPLATES.get(feat, f"{feat} contributed positively").format(val=val)
            elif shap_val < 0:
                if feat == "emp_length" and float(val) == 0:
                    note = "No recorded employment history — income stability is a concern"
                elif feat == "credit_utilization_pressure":
                    if float(val) < 0.30:
                        note = f"Revolving debt is {float(val):.1%} of annual income — manageable but noted"
                    else:
                        note = f"Revolving debt is {float(val):.1%} of annual income — over-extended on credit"
                else:
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

# ── 9. MAIN DECISION FUNCTION ─────────────────────────────────────────────────
def decide(applicant: dict, applicant_id: str = None) -> dict:
    if applicant_id is None:
        applicant_id = str(uuid.uuid4())[:8]

    # ── FAIL SAFE ─────────────────────────────────────────────────────────────
    if model is None:
        return {
            "applicant_id": applicant_id,
            "decision":     "DECLINED",
            "tier":         0,
            "tier_label":   "System Error",
            "decline_reasons": [
                "Our loan assessment system is temporarily unavailable.",
                "Please try again shortly.",
                "If this persists, contact ClearLend support."
            ],
            "model_version": "UNKNOWN",
            "decided_at":    datetime.now().isoformat()
        }

    # ── INPUT VALIDATION ──────────────────────────────────────────────────────
    errors = validate_applicant(applicant)
    if errors:
        return {
            "applicant_id":    applicant_id,
            "decision":        "ERROR",
            "tier":            0,
            "tier_label":      "Validation Error",
            "validation_errors": errors,
            "model_version":   version,
            "decided_at":      datetime.now().isoformat()
        }

    feature_names = metadata["features"]
    X_input       = pd.DataFrame([applicant])[feature_names]
    X_proc        = preprocessor.transform(X_input)

    repay_prob    = float(model.predict_proba(X_proc)[0][1])

    # SHAP — access underlying XGBoost model from calibrated wrapper
    base_model = model.estimator if hasattr(model, "estimator") else model
    explainer    = shap.TreeExplainer(base_model)
    shap_vals    = explainer.shap_values(X_proc)[0]
    narrative_str, narrative_steps = build_confidence_narrative(
        shap_vals, feature_names, X_proc[0], explainer.expected_value
    )

    decision = {
        "applicant_id":         applicant_id,
        "applicant":            applicant,
        "repay_probability":    round(repay_prob, 4),
        "model_version":        version,
        "decided_at":           datetime.now().isoformat(),
        "confidence_narrative": narrative_steps,
    }

    # ── LAYER 1: MAIN MODEL ───────────────────────────────────────────────────
    if repay_prob >= APPROVAL_THRESHOLD:
        offer_label, offer, pmt = select_offer(
            1, repay_prob, None, applicant["annual_inc"], applicant["dti"]
        )
        decision.update({
            "decision":        "APPROVED",
            "tier":            1,
            "tier_label":      "Clear Good",
            "offer_tier":      offer_label,
            "offer_amount":    offer["amount"],
            "offer_term":      f"{offer['term_months']} months",
            "offer_apr":       f"{offer['apr']*100:.0f}%",
            "monthly_payment": pmt,
        })

    else:
        # ── LAYER 2: DISTANCE FROM DEFAULTER ─────────────────────────────────
        distance_score, feature_distances = compute_distance_from_defaulter(applicant)
        decision["distance_score"]    = distance_score
        decision["feature_distances"] = feature_distances

        if distance_score >= DISTANCE_TIER2_THRESHOLD:
            # Medium good — meaningful distance from defaulter
            offer_label, offer, pmt = select_offer(
                2, repay_prob, distance_score, applicant["annual_inc"], applicant["dti"]
            )
            decision.update({
                "decision":        "APPROVED",
                "tier":            2,
                "tier_label":      "Medium Good",
                "offer_tier":      offer_label,
                "offer_amount":    offer["amount"],
                "offer_term":      f"{offer['term_months']} months",
                "offer_apr":       f"{offer['apr']*100:.0f}%",
                "monthly_payment": pmt,
            })

        elif distance_score >= DISTANCE_TIER3_THRESHOLD:
            # Medium okay — some distance from defaulter, trial loan via budget
            budget_result = try_approve_borderline(
                applicant_id, applicant, distance_score, repay_prob
            )
            offer  = OFFERS["A"]
            pmt    = monthly_payment(offer["amount"], offer["apr"], offer["term_months"])
            decision.update({
                "decision":        budget_result["decision"],  # APPROVED or WAITLISTED
                "tier":            3,
                "tier_label":      "Medium Okay",
                "offer_tier":      "A",
                "offer_amount":    offer["amount"],
                "offer_term":      f"{offer['term_months']} months",
                "offer_apr":       f"{offer['apr']*100:.0f}%",
                "monthly_payment": pmt,
                "budget_remaining": budget_result["budget_remaining"],
                "message":         budget_result["message"],
            })

        else:
            # Clear decline — too close to defaulter profile
            decline_steps   = [s for s in narrative_steps if s["direction"] == "negative"]
            decline_reasons = [s["note"] for s in decline_steps[:3]]
            while len(decline_reasons) < 3:
                decline_reasons.append("Overall credit profile did not meet lending criteria.")
            decision.update({
                "decision":        "DECLINED",
                "tier":            4,
                "tier_label":      "Clear Bad",
                "decline_reasons": decline_reasons,
            })

    # ── PRINT ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ID: {applicant_id}  |  Tier {decision['tier']}: {decision['tier_label']}")
    print(f"  Decision: {decision['decision']}   "
          f"(Repay prob: {repay_prob:.4f})", end="")
    if "distance_score" in decision:
        print(f"  |  Distance score: {distance_score:.4f}", end="")
    print(f"\n  Model version: {version}  |  {decision['decided_at']}")
    print(f"{'='*65}")
    print(f"\n  Confidence Narrative:\n")
    print(narrative_str)

    if decision["decision"] in ("APPROVED", "WAITLISTED"):
        if decision["tier"] == 3:
            print(f"\n  {decision['message']}")
        else:
            print(f"\n  Offer: {decision['offer_tier']} — "
                  f"${decision['offer_amount']} over {decision['offer_term']} "
                  f"at {decision['offer_apr']} APR")
            print(f"  Monthly payment: ${decision['monthly_payment']}")

        if "feature_distances" in decision:
            print(f"\n  Distance from defaulter breakdown:")
            for feat, vals in decision["feature_distances"].items():
                symbol = "✓" if vals["better"] else "✗"
                print(f"  {symbol} {feat:<25} "
                      f"yours: {vals['applicant_val']:>10.2f}  "
                      f"defaulter avg: {vals['defaulter_avg']:>10.2f}")
    else:
        print(f"\n  Decline Reasons:")
        for i, r in enumerate(decision["decline_reasons"], 1):
            print(f"  {i}. {r}")

    return decision

# ── 10. EXAMPLE USAGE ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from borderline_budget import reset_budget
    reset_budget(10000.00)

    tier1 = {
        "dti": 12.5, "revol_util": 18.0, "pub_rec": 0, "annual_inc": 85000,
        "debt_burden": round((12.5/100) * 85000 / 12, 2),
        "credit_utilization_pressure": round(3200 / 85000, 3),
        "has_public_record": 0, "high_dti_low_income": 0,
        "emp_length": 7, "open_acc": 8, "mort_acc": 1,
        "credit_history_years": 12.0, "revol_bal": 3200, "home_ownership": "MORTGAGE"
    }
    tier2 = {
        "dti": 13.0, "revol_util": 38.0, "pub_rec": 0, "annual_inc": 78000,
        "debt_burden": round((13.0/100) * 78000 / 12, 2),
        "credit_utilization_pressure": round(9000 / 78000, 3),
        "has_public_record": 0, "high_dti_low_income": 0,
        "emp_length": 7, "open_acc": 9, "mort_acc": 2,
        "credit_history_years": 16.0, "revol_bal": 9000, "home_ownership": "MORTGAGE"
    }
    tier3 = {
        "dti": 19.0, "revol_util": 58.0, "pub_rec": 0, "annual_inc": 55000,
        "debt_burden": round((19.0/100) * 55000 / 12, 2),
        "credit_utilization_pressure": round(13000 / 55000, 3),
        "has_public_record": 0, "high_dti_low_income": 0,
        "emp_length": 4, "open_acc": 11, "mort_acc": 0,
        "credit_history_years": 10.0, "revol_bal": 13000, "home_ownership": "RENT"
    }
    tier4 = {
        "dti": 38.0, "revol_util": 82.0, "pub_rec": 2, "annual_inc": 32000,
        "debt_burden": round((38.0/100) * 32000 / 12, 2),
        "credit_utilization_pressure": round(18000 / 32000, 3),
        "has_public_record": 1, "high_dti_low_income": 1,
        "emp_length": 1, "open_acc": 14, "mort_acc": 0,
        "credit_history_years": 3.0, "revol_bal": 18000, "home_ownership": "RENT"
    }

    import sys

    with open("decision_output.txt", "w") as f:
        # Redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout      = f

        for label, applicant in [
            ("Tier 1 — Clear Good",  tier1),
            ("Tier 2 — Medium Good", tier2),
            ("Tier 3 — Medium Okay", tier3),
            ("Tier 4 — Clear Bad",   tier4),
        ]:
            print(f"\n\n{'#'*65}")
            print(f"  {label}")
            print(f"{'#'*65}")
            decide(applicant)

        print(f"\n\n{'#'*65}")
        print(f"  Budget Status")
        print(f"{'#'*65}")
        print_budget_status()

        # Restore stdout
        sys.stdout = original_stdout

    print("Output written to decision_output.txt")