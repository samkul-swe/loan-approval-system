import json
import os
from datetime import datetime

BUDGET_FILE = "borderline_budget.json"

DEFAULT_BUDGET = {
    "total_budget":     10000.00,
    "amount_deployed":  0.00,
    "amount_repaid":    0.00,
    "amount_defaulted": 0.00,
    "waitlist":         [],
    "approved":         [],
    "last_updated":     datetime.now().isoformat()
}

# ── LOAD / SAVE ───────────────────────────────────────────────────────────────
def load_budget() -> dict:
    if os.path.exists(BUDGET_FILE):
        with open(BUDGET_FILE) as f:
            return json.load(f)
    save_budget(DEFAULT_BUDGET)
    return DEFAULT_BUDGET.copy()

def save_budget(state: dict):
    state["last_updated"] = datetime.now().isoformat()
    with open(BUDGET_FILE, "w") as f:
        json.dump(state, f, indent=2)

def remaining_budget(state: dict) -> float:
    return state["total_budget"] - state["amount_deployed"]

# ── AMOUNT CALCULATION ────────────────────────────────────────────────────────
def calculate_borderline_amount(distance_score: float) -> float:
    """
    Amount is based on distance score from defaulter profile.
    Higher distance = more confident = slightly higher amount.
    All Tier 3 amounts are small — this is a trial loan.
    """
    if distance_score >= 0.70:   return 500.0
    elif distance_score >= 0.60: return 400.0
    elif distance_score >= 0.50: return 350.0
    else:                        return 300.0

# ── TRY APPROVE ───────────────────────────────────────────────────────────────
def try_approve_borderline(applicant_id: str,
                            applicant: dict,
                            distance_score: float,
                            main_prob: float) -> dict:
    state       = load_budget()
    amount      = calculate_borderline_amount(distance_score)
    budget_left = remaining_budget(state)

    record = {
        "applicant_id":   applicant_id,
        "applicant":      applicant,
        "distance_score": round(distance_score, 4),
        "main_prob":      round(main_prob, 4),
        "amount":         amount,
        "timestamp":      datetime.now().isoformat(),
        "status":         None
    }

    if budget_left >= amount:
        record["status"] = "APPROVED"
        state["amount_deployed"] += amount
        state["approved"].append(record)
        save_budget(state)
        return {
            "decision":         "APPROVED",
            "amount":           amount,
            "budget_remaining": round(remaining_budget(state), 2),
            "message":          f"Approved as Tier 3 trial loan — "
                                f"${amount:.0f} at 9% APR over 6 months. "
                                f"Budget remaining: ${remaining_budget(state):,.2f}"
        }
    else:
        record["status"] = "WAITLISTED"
        state["waitlist"].append(record)
        save_budget(state)
        return {
            "decision":         "WAITLISTED",
            "amount":           amount,
            "budget_remaining": round(budget_left, 2),
            "message":          f"Tier 3 budget currently exhausted "
                                f"(${budget_left:.2f} remaining, need ${amount:.0f}). "
                                f"You have been added to the waitlist and will be "
                                f"reconsidered as repayments come in."
        }

# ── RECORD REPAYMENT ──────────────────────────────────────────────────────────
def record_repayment(applicant_id: str, amount: float):
    state = load_budget()
    state["amount_repaid"]  += amount
    state["total_budget"]   += amount

    approved_from_waitlist = []
    remaining_waitlist     = []

    for applicant in state["waitlist"]:
        if remaining_budget(state) >= applicant["amount"]:
            applicant["status"] = "APPROVED"
            state["amount_deployed"] += applicant["amount"]
            approved_from_waitlist.append(applicant)
            state["approved"].append(applicant)
        else:
            remaining_waitlist.append(applicant)

    state["waitlist"] = remaining_waitlist
    save_budget(state)

    print(f"Repayment of ${amount:.2f} recorded for {applicant_id}")
    print(f"Budget replenished to: ${remaining_budget(state):,.2f}")
    if approved_from_waitlist:
        print(f"Approved {len(approved_from_waitlist)} from waitlist:")
        for a in approved_from_waitlist:
            print(f"  - {a['applicant_id']} → ${a['amount']:.0f}")

# ── RECORD DEFAULT ────────────────────────────────────────────────────────────
def record_default(applicant_id: str, amount: float):
    state = load_budget()
    state["amount_defaulted"] += amount
    save_budget(state)
    print(f"Default of ${amount:.2f} recorded for {applicant_id}")
    print(f"Total defaulted so far: ${state['amount_defaulted']:,.2f}")

# ── STATUS ────────────────────────────────────────────────────────────────────
def print_budget_status():
    state       = load_budget()
    budget_left = remaining_budget(state)
    total_resolved = state["amount_repaid"] + state["amount_defaulted"]
    repay_rate  = (state["amount_repaid"] / total_resolved) if total_resolved > 0 else None

    print(f"\n── Tier 3 Budget Status ──")
    print(f"Total budget:       ${state['total_budget']:>10,.2f}")
    print(f"Deployed:           ${state['amount_deployed']:>10,.2f}")
    print(f"Remaining:          ${budget_left:>10,.2f}")
    print(f"Repaid so far:      ${state['amount_repaid']:>10,.2f}")
    print(f"Defaulted so far:   ${state['amount_defaulted']:>10,.2f}")
    if repay_rate:
        print(f"Actual repay rate:  {repay_rate:>10.1%}")
    print(f"Approved:           {len(state['approved']):>10,}")
    print(f"On waitlist:        {len(state['waitlist']):>10,}")
    print(f"Last updated:       {state['last_updated']}")

# ── RESET (for testing) ───────────────────────────────────────────────────────
def reset_budget(total: float = 10000.00):
    state = DEFAULT_BUDGET.copy()
    state["total_budget"] = total
    save_budget(state)
    print(f"Budget reset to ${total:,.2f}")