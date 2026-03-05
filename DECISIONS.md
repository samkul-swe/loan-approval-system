# Design Decisions

This document records the reasoning behind every major design decision in the ClearLend loan approval system. It exists so that future developers can understand not just what was built but why, and can make informed decisions about what to change.

---

## Data Decisions

### Why we dropped grade, sub_grade, int_rate and installment

These columns were assigned by Lending Club during their own underwriting process. Using them would mean the model is learning to replicate Lending Club's decisions rather than making its own from raw applicant data. This is data leakage — information that would not be available at the time ClearLend makes its decision.

### Why we dropped zip_code and addr_state

Geographic features in credit decisions carry significant fair lending risk. Using zip code or state as a credit signal can replicate historical redlining patterns even without intent. Dropped to avoid disparate impact on protected groups.

### Why we dropped emp_title

High cardinality free text field with inconsistent values. "Software Engineer", "SWE", "Sr. SWE" are the same job — the raw text adds noise not signal.

### Why we kept home_ownership despite fair lending concerns

home_ownership is a genuine financial stability signal — having a mortgage demonstrates prior lending trust and long-term financial commitment. We kept it but flagged it in the model card as a potential proxy variable requiring fairness monitoring.

### Why credit_history_years uses the loan issue date as reference

An earlier version used a fixed reference date of January 2020 for all loans. This introduced subtle leakage — loans issued in 2010 appeared to have 10 fewer years of credit history than they actually had at application time. The fix computes credit_history_years as (issue_d - earliest_cr_line) so each loan's history is calculated relative to when it was actually issued.

### Why we removed pub_rec_bankruptcies and total_acc

The correlation matrix showed pub_rec_bankruptcies correlates at 0.70 with pub_rec — they measure the same thing. total_acc correlates at 0.68 with open_acc. Keeping both in each pair adds noise without adding signal.

---

## Feature Engineering Decisions

### Why we added debt_burden

Raw dti is a percentage — it doesn't capture the absolute dollar magnitude of someone's debt obligations. A 20% dti on $30,000 income is very different from 20% on $200,000 income. debt_burden = (dti/100 × annual_inc) / 12 captures the actual monthly dollar commitment.

### Why we added credit_utilization_pressure

revol_util already captures utilisation as a percentage of credit limit. credit_utilization_pressure = revol_bal / annual_inc captures how much of annual income is tied up in revolving debt — a different dimension of financial stress. In the EDA this showed a 72% mean difference between defaulters (0.38) and repaid (0.22) — the strongest separation of any feature.

### Why we added high_dti_low_income

Neither high dti nor low income alone is a decisive signal. The dangerous combination is both together. This engineered feature captures that interaction explicitly so the model doesn't have to discover it through multiple tree splits. It became the second most important feature after Optuna tuning.

### Why we did not add SMOTE for class imbalance

SMOTE generates synthetic borrower profiles by interpolating between existing minority class examples. In credit risk this creates artificial borrowers that never existed — with feature combinations that no real defaulter ever had. This is particularly problematic for a regulated lending product where model decisions must be grounded in real historical data. We used undersampling instead.

### Why we removed thin_file flag

The EDA showed thin_file (credit_history_years < 5 AND open_acc < 4) occurred in only 0.1% of applicants with no meaningful difference between defaulters and repaid. Zero signal, dropped.

---

## Model Decisions

### Why XGBoost over logistic regression

The correlation matrix showed no single feature correlated with the target above 0.08. The signal is in feature interactions, not individual features. Logistic regression assigns fixed weights to features independently — it cannot capture "dti matters more when revol_util is also high." XGBoost builds sequential trees where each split is conditioned on prior splits, naturally capturing these interactions.

### Why the two-pass confident defaulter training strategy

The model is trained on confident defaulters (initial model probability < 0.35) paired with a random sample of repaid borrowers. The asymmetry is intentional:

- Defaulter side is strict: the model learns a sharp, unambiguous view of what bad looks like
- Repaid side is random: the model sees the full diversity of good borrowers, not just the most obvious ones

Using all defaulters made the model too strict on good borrowers (Grade A dropped to 34%). Using only the most confident defaulters with random repaid gave the best balance — Grade A at 97.7% with meaningful defaulter filtering.

### Why we did not use class weights instead

Class weights tell the model to penalise defaulter mistakes more heavily. This is a valid approach but it doesn't solve the uncertain middle — a borderline applicant still gets a single probability output with no mechanism for a second opinion. The two-pass approach combined with the four-tier decision system gives more nuanced handling of uncertainty.

### Why Optuna for hyperparameter tuning

Manual parameter selection is efficient but explores a tiny fraction of the search space and misses interaction effects between parameters. Optuna uses Bayesian optimisation — it learns from each trial which parameter combinations work better and focuses the search there. 100 trials found that gamma=0.05 (not 1.0 as manually set) and significant L1/L2 regularisation improved the cost analysis metrics meaningfully even when ROC-AUC barely moved.

### Why isotonic regression over Platt scaling for calibration

Platt scaling fits a logistic curve to map raw probabilities to calibrated ones — it assumes a specific functional form. Isotonic regression is non-parametric and finds the best monotonic mapping without assumptions about shape. XGBoost probabilities are often already roughly monotonic but not linearly calibrated, making isotonic regression the better fit.

---

## Decision System Decisions

### Why four tiers instead of binary approve/decline

A binary system forces every uncertain applicant into an approve or decline bucket. The four-tier system acknowledges that uncertainty is itself informative:

- Tier 1 (clear good): high confidence approval, full offer
- Tier 2 (medium good): meaningful distance from defaulter profile, moderate offer
- Tier 3 (medium okay): some distance from defaulter, trial loan tracked via budget
- Tier 4 (clear bad): close to defaulter profile, decline with reasons

This gives borderline applicants a fair chance proportional to the confidence level rather than treating all uncertainty the same way.

### Why the distance-from-defaulter score uses weighted features

Strong features (dti, revol_util, annual_inc, debt_burden, credit_utilization_pressure) carry more weight than supporting features in the distance calculation. This reflects their demonstrated importance in separating the two classes and ensures the distance score is driven by the most meaningful financial signals.

### Why the approval threshold is 0.75 not 0.50

After calibration, a 0.50 probability means exactly 50% of those applicants repay. Given the 3x cost asymmetry (defaults cost 3x more than missed revenue), the break-even repay rate is: repay_rate - (1-repay_rate) × 3 = 0 → repay_rate = 0.75. Any approval below 75% repay probability is expected to be unprofitable.

### Why we did not raise the threshold to 0.87 to filter Grade E-G

Raising the threshold to 0.87 reduced Grade A approvals from 97.7% to 34.3% because calibrated probabilities are compressed into a narrow 0.75-0.97 range. A threshold-based approach is too blunt for this problem. The Grade E-G issue requires a targeted rule or better features, not a global threshold change. Documented as a known limitation for future work.

### Why the Tier 3 budget tracker exists

The borderline analysis showed that 82.5% of Tier 3 applicants actually repaid historically. But these applicants are indistinguishable from the 17.5% that defaulted using available features. Rather than declining all of them or approving all of them, the budget tracker:

- Ring-fences a fixed capital allocation for this risky tier
- Approves applicants until the budget is exhausted
- Automatically approves waitlisted applicants as repayments come in
- Tracks actual repay/default rates to validate the 82.5% assumption over time

---

## Evaluation Decisions

### Why we used grade validation as external validation

Lending Club grades were deliberately excluded from training to avoid leakage. After training, running the model against those grades and checking if approval rates align with grade quality (Grade A highest, Grade G lowest) provides an independent sanity check that doesn't touch the test set.

### Why ROC-AUC is not the primary business metric

ROC-AUC measures separation across all thresholds equally. For this problem, a false positive (approving a defaulter) costs 3x more than a false negative (declining a good borrower). ROC-AUC doesn't know this. The cost analysis (false positives × 3 + false negatives × 1) is a more honest measure of business performance. PR AUC and KS statistic are planned additions.

---

## Planned Improvements

- Mahalanobis distance for more statistically rigorous tier scoring
- PR AUC and KS statistic as primary evaluation metrics
- Systematic leakage detection pass
- Loan outcome tracking and model feedback loop
- Disparate impact analysis once demographic data is available
- Drift detection — alert when incoming applicant distributions diverge from training
- Canary deployment — route small traffic percentage to new models before full rollout
- Offer selection by income percentage rather than fixed dti ceiling
- Input validation and system hardening before production deployment