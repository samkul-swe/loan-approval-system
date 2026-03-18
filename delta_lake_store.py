import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
from datetime import datetime

# ── WHAT IS DELTA LAKE ────────────────────────────────────────────────────────
# Delta Lake stores data as Parquet files + a transaction log (JSON files)
# The transaction log is what gives you:
#   - ACID transactions — writes are atomic, no partial data
#   - Time travel — query data as it looked at any past version
#   - Audit history — every change is recorded
#   - Schema enforcement — rejects data that doesn't match expected structure
#
# Structure on disk:
#   delta_tables/
#   └── loan_decisions/
#       ├── _delta_log/          ← transaction log (JSON files)
#       │   ├── 00000.json       ← version 0 — initial write
#       │   ├── 00001.json       ← version 1 — next append
#       │   └── ...
#       └── part-00000-*.parquet ← actual data in Parquet format

DECISIONS_TABLE  = "delta_tables/loan_decisions"
OUTCOMES_TABLE   = "delta_tables/loan_outcomes"
MODEL_LOG_TABLE  = "delta_tables/model_versions"

# ── 1. WRITING YOUR FIRST DELTA TABLE ─────────────────────────────────────────
# Scenario: store loan decisions as they come in

# Schema defined explicitly so Delta Lake knows the type of every column
# even when values are None — without this it sees Null type and errors
DECISIONS_SCHEMA = pa.schema([
    # ── WHO AND WHEN ──────────────────────────────────────────────────────────
    pa.field("applicant_id",          pa.string()),
    pa.field("decided_at",            pa.string()),

    # ── THE DECISION ──────────────────────────────────────────────────────────
    pa.field("decision",              pa.string()),
    pa.field("tier",                  pa.int32()),
    pa.field("tier_label",            pa.string()),

    # ── WHY THE DECISION WAS MADE ─────────────────────────────────────────────
    pa.field("repay_probability",     pa.float64()),
    pa.field("distance_score",        pa.float64()),
    pa.field("model_version",         pa.string()),

    # ── SELECTED OFFER (if approved) ──────────────────────────────────────────
    pa.field("offer_tier",            pa.string()),
    pa.field("offer_amount",          pa.float64()),
    pa.field("offer_term",            pa.string()),
    pa.field("offer_apr",             pa.string()),
    pa.field("monthly_payment",       pa.float64()),

    # ── ALL PLANS CONSIDERED ──────────────────────────────────────────────────
    # Every offer evaluated — amount, payment, whether it fit affordability
    pa.field("plan_a_amount",         pa.float64()),
    pa.field("plan_a_monthly",        pa.float64()),
    pa.field("plan_a_fits",           pa.bool_()),
    pa.field("plan_b_amount",         pa.float64()),
    pa.field("plan_b_monthly",        pa.float64()),
    pa.field("plan_b_fits",           pa.bool_()),
    pa.field("plan_c_amount",         pa.float64()),
    pa.field("plan_c_monthly",        pa.float64()),
    pa.field("plan_c_fits",           pa.bool_()),
    pa.field("plan_d_amount",         pa.float64()),
    pa.field("plan_d_monthly",        pa.float64()),
    pa.field("plan_d_fits",           pa.bool_()),
    pa.field("dti_headroom",          pa.float64()),  # how much monthly headroom they had
    pa.field("selected_reason",       pa.string()),   # why this offer was selected

    # ── MINIMUM PLAN FOR DECLINED (future path) ───────────────────────────────
    # What would need to change for this person to qualify for the smallest offer
    pa.field("min_plan_possible",     pa.bool_()),    # could they qualify for anything?
    pa.field("min_plan_amount",       pa.float64()),  # smallest amount we'd consider
    pa.field("min_plan_condition",    pa.string()),   # what needs to improve

    # ── DECLINE REASONS ───────────────────────────────────────────────────────
    pa.field("decline_reason_1",      pa.string()),
    pa.field("decline_reason_2",      pa.string()),
    pa.field("decline_reason_3",      pa.string()),

    # ── APPLICANT PROFILE AT DECISION TIME ────────────────────────────────────
    pa.field("dti",                           pa.float64()),
    pa.field("revol_util",                    pa.float64()),
    pa.field("pub_rec",                       pa.float64()),
    pa.field("annual_inc",                    pa.float64()),
    pa.field("emp_length",                    pa.float64()),
    pa.field("open_acc",                      pa.float64()),
    pa.field("mort_acc",                      pa.float64()),
    pa.field("credit_history_years",          pa.float64()),
    pa.field("revol_bal",                     pa.float64()),
    pa.field("home_ownership",                pa.string()),
    pa.field("debt_burden",                   pa.float64()),
    pa.field("credit_utilization_pressure",   pa.float64()),
    pa.field("has_public_record",             pa.int32()),
    pa.field("high_dti_low_income",           pa.int32()),
])


def evaluate_all_plans(annual_inc: float, dti: float) -> dict:
    """
    Evaluate all four offers against the applicant's affordability.
    Returns a dict with payment, fits status and headroom for every plan.
    This runs regardless of approval/decline so we always have the full picture.
    """
    OFFERS = {
        "A": {"amount": 500,  "term_months": 6,  "apr": 0.09},
        "B": {"amount": 1000, "term_months": 12, "apr": 0.12},
        "C": {"amount": 2500, "term_months": 24, "apr": 0.15},
        "D": {"amount": 5000, "term_months": 36, "apr": 0.19},
    }

    monthly_inc   = annual_inc / 12
    existing_debt = monthly_inc * (dti / 100)
    headroom      = max(0, monthly_inc * 0.40 - existing_debt)

    plans = {}
    for label, offer in OFFERS.items():
        r       = offer["apr"] / 12
        payment = round(offer["amount"] * r / (1 - (1 + r) ** -offer["term_months"]), 2)
        plans[label] = {
            "amount":  offer["amount"],
            "monthly": payment,
            "fits":    payment <= headroom
        }

    return plans, round(headroom, 2)


def compute_minimum_plan(applicant: dict, repay_prob: float) -> dict:
    """
    For declined applicants — figure out if there's any path to approval.
    Checks if Offer A ($500) would be affordable and what would need to improve.

    Returns:
        possible: bool — could we ever approve this person for anything?
        amount: float — minimum amount we'd consider
        condition: str — what needs to change
    """
    MIN_REPAY_PROB = 0.75   # break-even threshold
    MIN_AMOUNT     = 300.0  # absolute floor

    dti        = applicant.get("dti", 100)
    revol_util = applicant.get("revol_util", 100)
    pub_rec    = applicant.get("pub_rec", 0)
    annual_inc = applicant.get("annual_inc", 0)

    # Hard blocks — some profiles can't be approved at any amount
    if pub_rec >= 3:
        return {
            "possible":  False,
            "amount":    None,
            "condition": "Multiple derogatory public records prevent approval at any amount"
        }

    if annual_inc < 15000:
        return {
            "possible":  False,
            "amount":    None,
            "condition": "Income is too low to support any loan repayment"
        }

    # Identify the biggest blocker and suggest what needs to improve
    conditions = []
    if dti > 35:
        conditions.append(f"reduce debt-to-income ratio below 35% (currently {dti:.1f}%)")
    if revol_util > 80:
        conditions.append(f"reduce credit utilisation below 80% (currently {revol_util:.1f}%)")
    if repay_prob < 0.60:
        conditions.append("improve overall credit profile")

    if not conditions:
        # Borderline — close to approval
        return {
            "possible":  True,
            "amount":    MIN_AMOUNT,
            "condition": f"A minimum trial loan of ${MIN_AMOUNT:.0f} may be possible "
                         f"once repayment probability improves above 75%"
        }

    return {
        "possible":  True,
        "amount":    MIN_AMOUNT,
        "condition": f"A minimum loan of ${MIN_AMOUNT:.0f} may be possible if you: "
                     + " and ".join(conditions)
    }


def write_decision(decision: dict):
    """
    Append a full loan decision record to the Delta table.
    Includes all plans considered, selected offer reasoning,
    and minimum plan path for declined applicants.
    """
    applicant = decision.get("applicant", {})
    reasons   = decision.get("decline_reasons", [])

    # Evaluate all plans for every applicant regardless of outcome
    plans, headroom = evaluate_all_plans(
        applicant.get("annual_inc", 0),
        applicant.get("dti", 0)
    )

    # Determine why the selected offer was chosen
    if decision.get("decision") == "APPROVED":
        selected = decision.get("offer_tier")
        fits     = [k for k, v in plans.items() if v["fits"]]
        not_fits = [k for k, v in plans.items() if not v["fits"]]
        if not_fits:
            selected_reason = (f"Offer {selected} selected — "
                               f"Offer(s) {', '.join(not_fits)} exceeded monthly affordability")
        else:
            selected_reason = f"Offer {selected} selected based on repay probability tier"
    else:
        selected_reason = "No offer — application declined"

    # Compute minimum plan path for declined applicants
    min_plan = {"possible": None, "amount": None, "condition": None}
    if decision.get("decision") in ("DECLINED",):
        min_plan = compute_minimum_plan(applicant, decision.get("repay_probability", 0))

    row = {
        # Who and when
        "applicant_id":         str(decision.get("applicant_id") or ""),
        "decided_at":           str(decision.get("decided_at") or ""),

        # Decision
        "decision":             str(decision.get("decision") or ""),
        "tier":                 int(decision.get("tier") or 0),
        "tier_label":           str(decision.get("tier_label") or ""),

        # Why
        "repay_probability":    float(decision.get("repay_probability") or 0.0),
        "distance_score":       float(decision["distance_score"]) if decision.get("distance_score") is not None else None,
        "model_version":        str(decision.get("model_version") or ""),

        # Selected offer
        "offer_tier":           str(decision["offer_tier"]) if decision.get("offer_tier") else None,
        "offer_amount":         float(decision["offer_amount"]) if decision.get("offer_amount") is not None else None,
        "offer_term":           str(decision["offer_term"]) if decision.get("offer_term") else None,
        "offer_apr":            str(decision["offer_apr"]) if decision.get("offer_apr") else None,
        "monthly_payment":      float(decision["monthly_payment"]) if decision.get("monthly_payment") is not None else None,

        # All plans considered
        "plan_a_amount":        plans["A"]["amount"],
        "plan_a_monthly":       plans["A"]["monthly"],
        "plan_a_fits":          plans["A"]["fits"],
        "plan_b_amount":        plans["B"]["amount"],
        "plan_b_monthly":       plans["B"]["monthly"],
        "plan_b_fits":          plans["B"]["fits"],
        "plan_c_amount":        plans["C"]["amount"],
        "plan_c_monthly":       plans["C"]["monthly"],
        "plan_c_fits":          plans["C"]["fits"],
        "plan_d_amount":        plans["D"]["amount"],
        "plan_d_monthly":       plans["D"]["monthly"],
        "plan_d_fits":          plans["D"]["fits"],
        "dti_headroom":         headroom,
        "selected_reason":      selected_reason,

        # Minimum plan for declined
        "min_plan_possible":    min_plan["possible"],
        "min_plan_amount":      float(min_plan["amount"]) if min_plan["amount"] else None,
        "min_plan_condition":   min_plan["condition"],

        # Decline reasons
        "decline_reason_1":     reasons[0] if len(reasons) > 0 else None,
        "decline_reason_2":     reasons[1] if len(reasons) > 1 else None,
        "decline_reason_3":     reasons[2] if len(reasons) > 2 else None,

        # Applicant profile
        "dti":                          float(applicant.get("dti") or 0.0),
        "revol_util":                   float(applicant.get("revol_util") or 0.0),
        "pub_rec":                      float(applicant.get("pub_rec") or 0.0),
        "annual_inc":                   float(applicant.get("annual_inc") or 0.0),
        "emp_length":                   float(applicant.get("emp_length") or 0.0),
        "open_acc":                     float(applicant.get("open_acc") or 0.0),
        "mort_acc":                     float(applicant.get("mort_acc") or 0.0),
        "credit_history_years":         float(applicant.get("credit_history_years") or 0.0),
        "revol_bal":                    float(applicant.get("revol_bal") or 0.0),
        "home_ownership":               str(applicant.get("home_ownership") or ""),
        "debt_burden":                  float(applicant.get("debt_burden") or 0.0),
        "credit_utilization_pressure":  float(applicant.get("credit_utilization_pressure") or 0.0),
        "has_public_record":            int(applicant.get("has_public_record") or 0),
        "high_dti_low_income":          int(applicant.get("high_dti_low_income") or 0),
    }

    table = pa.Table.from_pydict(
        {k: [v] for k, v in row.items()},
        schema=DECISIONS_SCHEMA
    )

    write_deltalake(
        table_or_uri = DECISIONS_TABLE,
        data         = table,
        mode         = "append",
        schema_mode  = "merge"
    )

    # Print summary
    print(f"\nDecision logged: {row['applicant_id']} → {row['decision']}")
    print(f"  Repay prob: {row['repay_probability']:.2f}  |  "
          f"Headroom: ${row['dti_headroom']:.0f}/mo")
    print(f"  Plans evaluated: "
          + ", ".join([f"{k}(${plans[k]['monthly']:.0f})"
                       + ("✓" if plans[k]["fits"] else "✗")
                       for k in ["A","B","C","D"]]))
    if row["decision"] == "APPROVED":
        print(f"  Selected: {row['selected_reason']}")
    elif row["min_plan_condition"]:
        print(f"  Min plan path: {row['min_plan_condition']}")


# ── 2. READING FROM A DELTA TABLE ─────────────────────────────────────────────
def read_all_decisions() -> pd.DataFrame:
    """Read all decisions from the Delta table."""
    dt = DeltaTable(DECISIONS_TABLE)
    return dt.to_pandas()

def read_approved_only() -> pd.DataFrame:
    """Read only approved decisions using partition filter."""
    dt = DeltaTable(DECISIONS_TABLE)
    # Filter pushdown — only reads relevant Parquet files
    return dt.to_pandas(
        filters=[("decision", "=", "APPROVED")]
    )


# ── 3. TIME TRAVEL ────────────────────────────────────────────────────────────
# This is one of Delta Lake's most powerful features
# Every write creates a new version — you can query any past version

def read_decisions_at_version(version: int) -> pd.DataFrame:
    """Read decisions table as it looked at a specific version."""
    dt = DeltaTable(DECISIONS_TABLE, version=version)
    return dt.to_pandas()

def read_decisions_at_timestamp(timestamp: str) -> pd.DataFrame:
    """
    Read decisions table as it looked at a specific point in time.
    timestamp format: "2026-03-01T00:00:00"
    Useful for: "what did our approval rate look like last month?"
    """
    dt = DeltaTable(DECISIONS_TABLE)
    return dt.load_as_version(timestamp).to_pandas()

def show_table_history():
    """Show the full history of changes to the decisions table."""
    dt = DeltaTable(DECISIONS_TABLE)
    history = dt.history()
    print("\n── Table History ──")
    for entry in history:
        print(f"  Version {entry['version']}  |  "
              f"{entry['timestamp']}  |  "
              f"{entry['operation']}")


# ── 4. RECORDING LOAN OUTCOMES (REPAID / DEFAULTED) ──────────────────────────
# When a loan resolves, record the outcome
# This is the ground truth feedback loop for model retraining

def record_outcome(applicant_id: str, outcome: str, amount: float):
    """
    Record the final outcome of a loan.
    outcome: 'REPAID' or 'DEFAULTED'
    """
    df = pd.DataFrame([{
        "applicant_id":  applicant_id,
        "outcome":       outcome,
        "amount":        amount,
        "resolved_at":   datetime.now().isoformat(),
    }])

    write_deltalake(
        table_or_uri = OUTCOMES_TABLE,
        data         = df,
        mode         = "append"
    )
    print(f"Outcome recorded: {applicant_id} → {outcome} (${amount:.2f})")


# ── 5. MODEL VERSION LOG ──────────────────────────────────────────────────────
# Track every model version deployed, its metrics and when it was active

def log_model_version(version: str, roc_auc: float,
                       pr_auc: float, ks_stat: float):
    df = pd.DataFrame([{
        "version":      version,
        "roc_auc":      roc_auc,
        "pr_auc":       pr_auc,
        "ks_stat":      ks_stat,
        "deployed_at":  datetime.now().isoformat(),
        "status":       "ACTIVE"
    }])
    write_deltalake(
        table_or_uri = MODEL_LOG_TABLE,
        data         = df,
        mode         = "append"
    )
    print(f"Model version logged: {version}")


# ── 6. DRIFT DETECTION HELPER ─────────────────────────────────────────────────
# Compare current approval rate vs historical baseline
# This is the foundation of drift detection

def check_approval_rate_drift(window_days: int = 7) -> dict:
    """
    Compare approval rate over the last N days vs all-time baseline.
    A significant shift signals potential drift.
    """
    df = read_all_decisions()
    if len(df) == 0:
        return {"error": "No decisions recorded yet"}

    df["decided_at"] = pd.to_datetime(df["decided_at"])
    cutoff           = pd.Timestamp.now() - pd.Timedelta(days=window_days)

    recent   = df[df["decided_at"] >= cutoff]
    baseline = df

    baseline_rate = (baseline["decision"] == "APPROVED").mean()
    recent_rate   = (recent["decision"] == "APPROVED").mean() if len(recent) > 0 else None

    drift = abs(recent_rate - baseline_rate) if recent_rate is not None else None

    result = {
        "baseline_approval_rate": round(baseline_rate, 3),
        "recent_approval_rate":   round(recent_rate, 3) if recent_rate else None,
        "drift":                  round(drift, 3) if drift else None,
        "alert":                  drift > 0.10 if drift else False  # alert if >10% shift
    }

    print(f"\n── Approval Rate Drift Check ──")
    print(f"Baseline rate (all time): {result['baseline_approval_rate']:.1%}")
    print(f"Recent rate (last {window_days}d): {result['recent_approval_rate']:.1%}"
          if result['recent_approval_rate'] else "  Recent: no data")
    if result["alert"]:
        print(f"⚠️  ALERT: Approval rate shifted by {result['drift']:.1%} — check for drift")
    else:
        print(f"✓  No significant drift detected")

    return result


# ── 7. EXAMPLE USAGE ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import shutil, os

    # Clean up previous run
    for path in [DECISIONS_TABLE, OUTCOMES_TABLE, MODEL_LOG_TABLE]:
        if os.path.exists(path):
            shutil.rmtree(path)

    sample_decisions = [
        {
            "applicant_id":      "APP001",
            "decided_at":        datetime.now().isoformat(),
            "decision":          "APPROVED",
            "tier":              1,
            "tier_label":        "Clear Good",
            "repay_probability": 0.89,
            "distance_score":    None,
            "model_version":     "20260305_235624",
            "offer_tier":        "D",
            "offer_amount":      5000,
            "offer_term":        "36 months",
            "offer_apr":         "19%",
            "monthly_payment":   183.28,
            "applicant": {
                "dti": 12.5, "revol_util": 18.0, "pub_rec": 0,
                "annual_inc": 85000, "emp_length": 7, "open_acc": 8,
                "mort_acc": 1, "credit_history_years": 12.0,
                "revol_bal": 3200, "home_ownership": "MORTGAGE",
                "debt_burden": 885.0, "credit_utilization_pressure": 0.038,
                "has_public_record": 0, "high_dti_low_income": 0
            }
        },
        {
            "applicant_id":      "APP002",
            "decided_at":        datetime.now().isoformat(),
            "decision":          "DECLINED",
            "tier":              4,
            "tier_label":        "Clear Bad",
            "repay_probability": 0.52,
            "distance_score":    0.21,
            "model_version":     "20260305_235624",
            "offer_tier":        None,
            "offer_amount":      None,
            "offer_term":        None,
            "offer_apr":         None,
            "monthly_payment":   None,
            "decline_reasons": [
                "High debt-to-income ratio of 38.0% — too much income already committed",
                "2 derogatory public record(s) — negative financial history",
                "Short credit history of 3.0 years — limited track record"
            ],
            "applicant": {
                "dti": 38.0, "revol_util": 82.0, "pub_rec": 2,
                "annual_inc": 32000, "emp_length": 1, "open_acc": 14,
                "mort_acc": 0, "credit_history_years": 3.0,
                "revol_bal": 18000, "home_ownership": "RENT",
                "debt_burden": 1013.0, "credit_utilization_pressure": 0.562,
                "has_public_record": 1, "high_dti_low_income": 1
            }
        },
        {
            "applicant_id":      "APP003",
            "decided_at":        datetime.now().isoformat(),
            "decision":          "APPROVED",
            "tier":              3,
            "tier_label":        "Medium Okay",
            "repay_probability": 0.76,
            "distance_score":    0.57,
            "model_version":     "20260305_235624",
            "offer_tier":        "A",
            "offer_amount":      500,
            "offer_term":        "6 months",
            "offer_apr":         "9%",
            "monthly_payment":   85.53,
            "applicant": {
                "dti": 19.0, "revol_util": 58.0, "pub_rec": 0,
                "annual_inc": 55000, "emp_length": 4, "open_acc": 11,
                "mort_acc": 0, "credit_history_years": 10.0,
                "revol_bal": 13000, "home_ownership": "RENT",
                "debt_burden": 871.0, "credit_utilization_pressure": 0.236,
                "has_public_record": 0, "high_dti_low_income": 0
            }
        },
    ]

    print("── Writing Decisions to Delta Table ──")
    for d in sample_decisions:
        write_decision(d)

    print("\n── Reading All Decisions ──")
    df = read_all_decisions()
    print(df[[
        "applicant_id", "decision", "tier_label",
        "repay_probability", "offer_amount",
        "decline_reason_1", "dti", "annual_inc"
    ]].to_string(index=False))

    print("\n── Table History ──")
    show_table_history()

    print("\n── Recording Outcomes ──")
    record_outcome("APP001", "REPAID", 5000)
    record_outcome("APP003", "DEFAULTED", 500)

    print("\n── Logging Model Version ──")
    log_model_version("20260305_235624", 0.9741, 0.9825, 0.8994)

    print("\n── Time Travel — Version 0 ──")
    df_v0 = read_decisions_at_version(0)
    print(f"Version 0 had {len(df_v0)} decision(s)")
    print(df_v0[["applicant_id", "decision", "offer_amount"]].to_string(index=False))

    check_approval_rate_drift(window_days=7)