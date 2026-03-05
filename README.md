# ClearLend Loan Approval System

A machine learning system that makes real-time loan approval decisions for point-of-sale lending. Given a new applicant's financial profile, the system decides whether to approve or decline, selects an appropriate financing offer, and produces a full explanation of every decision.

Built by Sampada Kulkarni as a step-by-step learning project. See `ABOUT.md` for author profile and `DECISIONS.md` for the reasoning behind every design choice.

---

## The Problem

> *ClearLend is a point-of-sale lender. Merchants integrate ClearLend into their checkout flow so customers can split large purchases into installment payments. When a customer applies, ClearLend has roughly 2 seconds to make a decision and respond.*

**What the system must do:**
- Decide whether to approve or decline a loan application in under 2 seconds
- If approved, select the most appropriate financing offer
- Every declined application must produce at least 3 human-readable reasons
- Decisions must be reproducible — same input, same model version, same output
- Must be able to roll back to any previous model version within minutes
- Must detect when the applicant population starts looking different from training data
- New models must prove themselves before replacing existing ones (canary deployment)

**Available offers:**

| Offer | Amount | Term | APR |
|---|---|---|---|
| A | $500 | 6 months | 9% |
| B | $1,000 | 12 months | 12% |
| C | $2,500 | 24 months | 15% |
| D | $5,000 | 36 months | 19% |

**Business cost asymmetry:** Approving a defaulter costs ClearLend approximately 3x more than the revenue lost from declining a good borrower.

---

## Current Status — March 2026

### ✅ Complete

- Data cleaning and feature engineering pipeline
- XGBoost classifier with Optuna hyperparameter tuning
- Isotonic regression probability calibration
- Four-tier decision system (Clear Good / Medium Good / Medium Okay / Clear Bad)
- SHAP-based confidence narrative for every decision
- 3 human-readable decline reasons per declined application
- Offer selection by tier and affordability
- Tier 3 trial loan budget tracker with waitlist
- Model versioning — every model saved with timestamp, metadata and audit trail
- Grade validation — Grade A 97.7%, Grade B 92.8%, Grade C 87.5%
- Stress test — 8/8 correct on clear approve and clear decline cases
- Probability calibration — mean absolute gap reduced from 0.18 to 0.003

### ⏳ In Progress

- Data drift detection
- Canary deployment

### 📋 Planned (see Refinement Plan below)

- Mahalanobis distance for tier scoring
- PR AUC and KS statistic as primary evaluation metrics
- Loan outcome tracking and model feedback loop
- Disparate impact analysis
- Input validation and system hardening
- Offer selection by income percentage

---

## Quick Start

```bash
git clone https://github.com/samkul-swe/loan-approval-system
cd loan-approval-system
pip install -r requirements.txt
```

Download the Lending Club dataset from Kaggle (`lending_club_loan_two.csv`) and place it in the project root.

Run in order:

```bash
python eda.py                  # clean data, engineer features → lending_club_clean.csv
python xgboost_train.py        # train model → model_registry/
python optuna_tune.py          # tune hyperparameters → model_registry/best_params.json
python xgboost_train.py        # retrain with tuned params
python calibration.py          # calibrate probabilities
python xgboost_predict.py      # run example decisions → decision_output.txt
python stress_test.py          # validate on 12 test cases
python grade_validation.py     # validate against Lending Club grades
```

---

## How the Model Works

### Input Features

| Feature | Type | Description |
|---|---|---|
| dti | Numerical | Debt-to-income ratio (%) |
| revol_util | Numerical | Revolving credit utilisation (%) |
| pub_rec | Numerical | Number of derogatory public records |
| annual_inc | Numerical | Annual income ($) |
| debt_burden | Engineered | Monthly debt payment = (dti/100 × annual_inc) / 12 |
| credit_utilization_pressure | Engineered | Revolving balance as fraction of annual income |
| has_public_record | Engineered | Binary — 1 if pub_rec > 0 |
| high_dti_low_income | Engineered | Binary — 1 if dti > 30 and annual_inc < 50,000 |
| emp_length | Numerical | Years of employment (0-10) |
| open_acc | Numerical | Number of open credit accounts |
| mort_acc | Numerical | Number of mortgage accounts |
| credit_history_years | Engineered | Years since earliest credit line (per loan date) |
| revol_bal | Numerical | Total revolving credit balance ($) |
| home_ownership | Categorical | RENT / OWN / MORTGAGE / OTHER |

### Decision Pipeline

```
Applicant
    ↓
Layer 1 — Calibrated XGBoost model
    ├── repay_prob >= 0.75 → TIER 1: Clear Good → full offer selection (A-D)
    └── repay_prob < 0.75
            ↓
        Layer 2 — Distance from defaulter score
            ├── score >= 0.75 → TIER 2: Medium Good → Offer B or C
            ├── score 0.40–0.75 → TIER 3: Medium Okay → Offer A (budget tracked)
            └── score < 0.40 → TIER 4: Clear Bad
                                            ↓
                                        DECLINED + 3 SHAP-based reasons
```

### Model Architecture

- **Algorithm**: XGBoost classifier with isotonic regression calibration
- **Training**: Confident defaulters (prob < 0.35) + random repaid sample, balanced 50/50
- **Hyperparameters**: Optuna-tuned (100 trials), stored in `model_registry/best_params.json`
- **Calibration**: Isotonic regression — mean calibration gap 0.003
- **Explainability**: SHAP values per decision — full confidence narrative

### Model Versioning

```
model_registry/
├── latest.json                               # points to current model
├── xgb_model_YYYYMMDD_HHMMSS.joblib         # raw XGBoost model
├── preprocessor_YYYYMMDD_HHMMSS.joblib      # fitted preprocessor
├── calibrated_model_YYYYMMDD_HHMMSS.joblib  # calibrated wrapper
├── metadata_YYYYMMDD_HHMMSS.json            # ROC-AUC, features, config
└── best_params.json                          # Optuna best parameters
```

To roll back: update `latest.json` to point to any prior version.

---

## Project Structure

```
loan-approval-system/
├── eda.py                          # data cleaning, feature engineering
├── xgboost_train.py                # model training
├── calibration.py                  # probability calibration
├── xgboost_predict.py              # decision engine
├── optuna_tune.py                  # hyperparameter tuning
├── stress_test.py                  # model stress testing
├── grade_validation.py             # external grade validation
├── generate_declined_profiles.py   # synthetic declined profiles
├── verify_declined_profiles.py     # verify profiles against model
├── borderline_budget.py            # Tier 3 budget tracker
├── model_registry/                 # saved models and metadata
├── lending_club_loan_two.csv       # raw dataset (not committed)
├── lending_club_clean.csv          # cleaned dataset (generated)
├── ABOUT.md                        # author profile
├── MODEL_CARD.md                   # formal model documentation
├── DECISIONS.md                    # design decision log
└── requirements.txt
```

---

## Key Decisions

| Decision | Rationale |
|---|---|
| XGBoost over logistic regression | No single feature correlated with target above 0.08 — signal is in interactions, not individual features |
| Confident defaulters + random repaid training | Sharp defaulter view + full diversity of good borrowers — best grade validation balance |
| Isotonic calibration | Mean gap reduced from 0.18 to 0.003 — probabilities now represent real repay rates |
| Approval threshold 0.75 | Break-even given 3x cost asymmetry: repay_rate - (1-repay_rate)×3 = 0 → 0.75 |
| Four-tier system | Uncertainty is informative — borderline applicants deserve proportional treatment not binary decisions |
| Dropped grade/sub_grade | Assigned during Lending Club underwriting — using them copies their model, not building our own |
| Dropped zip_code/state | Fair lending risk — geographic signals can replicate historical redlining |
| credit_history_years per loan date | Fixed 2020 reference date introduced subtle leakage for older loans |

Full reasoning in `DECISIONS.md`.

---

## Refinement Plan

### Priority 1 — Before Any Real Deployment
- Input validation — reject missing or out-of-range values
- Budget file locking — prevent race conditions
- Fail-safe fallback — decline all if model files missing or corrupted

### Priority 2 — Model Quality
- Mahalanobis distance for statistically rigorous tier scoring
- PR AUC and KS statistic as primary evaluation metrics
- Probability calibration tail fix for Grade E-G
- Systematic leakage detection pass
- Offer selection by income percentage

### Priority 3 — Production Readiness
- Drift detection — alert when applicant distributions diverge from training
- Canary deployment — 5% traffic to new models before full rollout
- Loan outcome tracking and model feedback loop
- Cold start strategy — conservative thresholds + circuit breaker for first 6 months

### Priority 4 — Fairness and Compliance
- Disparate impact analysis once demographic data available
- Adverse action notice generation
- Model governance documentation

---

## Known Limitations

**Grade E-F-G over-approval** — model approves ~80% of Grade E-G borrowers despite 37-48% default rates. Data limitation, not model limitation. See `MODEL_CARD.md`.

**Proxy dataset** — trained on Lending Club data (2007-2018), not ClearLend's actual customers.

**No drift detection yet** — planned.

**No canary deployment yet** — planned.

**Test set leakage** — ROC-AUC of 0.9734 may be optimistic. See `MODEL_CARD.md`.

**No fairness testing** — disparate impact analysis not performed.

---

## Validation Results

| Grade | Approval Rate | Actual Repay Rate |
|---|---|---|
| A | 97.7% | 93.7% |
| B | 92.8% | 87.4% |
| C | 87.5% | 78.8% |
| D | 82.9% | 71.1% |
| E | 80.9% | 62.6% |
| F | 80.4% | 57.2% |
| G | 82.2% | 52.2% |

Stress test: 8/8 correct on clear approve and clear decline cases.