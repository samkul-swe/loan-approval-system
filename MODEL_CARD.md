# Model Card — ClearLend Loan Approval Model

## Model Details

| | |
|---|---|
| Model type | XGBoost classifier with isotonic regression calibration |
| Version | See `model_registry/latest.json` |
| Trained by | Sampada Kulkarni |
| Date | March 2026 |
| Repository | https://github.com/samkul-swe/loan-approval-system |

---

## Intended Use

**Primary use case**: Real-time loan approval decisions for ClearLend point-of-sale lending. Given a customer's financial profile at checkout, the model decides whether to approve or decline a loan application and selects an appropriate financing offer.

**Decision latency**: Designed to serve decisions in under 2 seconds end to end.

**Intended users**: ClearLend's loan origination system. Not intended for direct use by applicants or non-technical staff without appropriate interfaces.

**Out-of-scope uses**:
- Mortgage lending decisions
- Business loan decisions
- Any context where demographic data is available and used as input
- Batch scoring of large applicant pools without human oversight

---

## Training Data

**Source**: Lending Club loan dataset (Kaggle proxy — `lending_club_loan_two.csv`)

**Time period**: Loans issued approximately 2007-2018

**Size**: 396,030 loans with known outcomes (Fully Paid or Charged Off)

**Class distribution**: ~80% repaid, ~20% defaulted

**Training strategy**: Balanced 50/50 sample — confident defaulters (initial model probability < 0.35) paired with a random sample of repaid loans of equal size. This approach ensures the model learns a sharp view of bad borrowers while seeing the full diversity of good borrowers.

**Features deliberately excluded**:
- `grade`, `sub_grade` — Lending Club's own underwriting assessment (data leakage)
- `int_rate`, `installment` — derived from underwriting decision
- `loan_amnt` — decision output, not applicant input
- `issue_d` — post-decision date
- `zip_code`, `addr_state` — geographic features with fair lending risk
- `emp_title` — high cardinality free text

---

## Evaluation

| Metric | Value |
|---|---|
| ROC-AUC (test set) | 0.9734 |
| Accuracy | 95% |
| Precision (defaulted) | 0.92 |
| Recall (defaulted) | 0.98 |
| Brier score (before calibration) | 0.2013 |
| Brier score (after calibration) | 0.1519 |
| Mean calibration gap (before) | 0.18 |
| Mean calibration gap (after) | 0.003 |

**Note on ROC-AUC**: The reported 0.9734 was produced after multiple training iterations that used test set performance to guide decisions. The true out-of-sample performance may be lower. This is a known limitation documented in the refinement plan.

**External validation against Lending Club grades** (grades were never used as features):

| Grade | Approval Rate | Actual Repay Rate |
|---|---|---|
| A | 97.7% | 93.7% |
| B | 92.8% | 87.4% |
| C | 87.5% | 78.8% |
| D | 82.9% | 71.1% |
| E | 80.9% | 62.6% |
| F | 80.4% | 57.2% |
| G | 82.2% | 52.2% |

---

## Ethical Considerations

**Fair lending**: The model does not use race, gender, age, national origin, marital status or religion as features. However the following features may act as proxies for protected characteristics:

- `home_ownership` — renters are disproportionately younger, lower income, and from minority communities
- `annual_inc` — income correlates with race and gender due to systemic inequalities
- `credit_history_years` — longer histories naturally favour older applicants

No disparate impact analysis has been performed due to absence of demographic data in the training set. This must be conducted before regulated deployment using actual applicant demographic data.

**Explainability**: Every decision — approved or declined — produces a full SHAP-based confidence narrative showing which features drove the decision and by how much. Every declined application produces at least 3 specific human-readable reasons. This satisfies the adverse action notice requirement under ECOA.

**Auditability**: Every decision is logged with model version, timestamp, repay probability, tier assignment and full feature contributions. Any historical decision can be reconstructed from the audit log.

---

## Known Limitations

**Grade E-F-G over-approval**: The model approves approximately 80% of Grade E, F, G borrowers despite actual default rates of 37-48%. The available financial features do not cleanly separate these borrowers. This is a data limitation — the Lending Club dataset only contains issued loans, so the model has never seen truly high-risk applicants who were declined before receiving a loan (survivorship bias).

**Proxy dataset mismatch**: The model was trained on Lending Club personal loan data. ClearLend is a point-of-sale lender with different loan sizes ($500-$5,000 vs Lending Club's $1,000-$40,000), different customer intent, and different application context. Probability thresholds and calibration are optimised for the Lending Club population.

**Calibration tail risk**: Isotonic regression calibration performs well across the bulk of the distribution but overestimates repay probability for Grade E-G borrowers. Grade G average predicted probability is 0.81 while actual repay rate is 52%.

**No demographic fairness testing**: Disparate impact analysis has not been performed.

**No drift detection**: The system does not currently alert when incoming applicant distributions diverge from training data.

**No canary deployment**: New model versions replace the current model without a proving period.

**Cold start risk**: The model has not been validated on real ClearLend applicants. Performance on ClearLend's actual customer base is unknown.

---

## Recommendations Before Production Deployment

1. Conduct disparate impact analysis using real applicant demographic data
2. Implement drift detection before going live
3. Implement canary deployment — route 5% of traffic to new models before full rollout
4. Run in shadow mode alongside human underwriters for the first 30-60 days
5. Set a cold start budget and circuit breaker — if actual defaults exceed predicted by more than X%, pause approvals
6. Begin collecting ClearLend-specific repayment outcomes immediately for retraining
7. Retrain on ClearLend data once 1,000+ resolved loans are available
8. Commission an independent fairness audit before scaling

---

## Refinement Roadmap

See `DECISIONS.md` for the full list of planned improvements including Mahalanobis distance scoring, PR AUC evaluation, loan outcome tracking and feedback loop, and offer selection by income percentage.