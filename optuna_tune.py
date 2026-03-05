import pandas as pd
import numpy as np
import json
import joblib
import optuna
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv("lending_club_clean.csv")

strong_num   = ["dti", "revol_util", "pub_rec", "annual_inc"]
engineered   = ["debt_burden", "credit_utilization_pressure",
                "has_public_record", "high_dti_low_income"]
support_num  = ["emp_length", "open_acc", "mort_acc", "credit_history_years", "revol_bal"]
cat_features = ["home_ownership"]
all_num      = strong_num + engineered + support_num
all_features = all_num + cat_features

X = df[all_features]
y = df["target"]

# ── 2. PREPROCESSOR ───────────────────────────────────────────────────────────
def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), all_num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_features)
    ], remainder="drop")

# ── 3. BUILD TRAINING DATASET ─────────────────────────────────────────────────
# Same strategy as xgboost_train.py — confident defaulters + random repaid
print("Building training dataset...")

prep_temp    = build_preprocessor()
n_def        = (y == 0).sum()
def_idx      = y[y == 0].index
rep_idx      = y[y == 1].index.to_series().sample(n=n_def, random_state=42)
bal_idx      = pd.concat([def_idx.to_series(), rep_idx])
X_bal_proc   = prep_temp.fit_transform(X.loc[bal_idx])
y_bal        = y.loc[bal_idx]

initial_model = xgb.XGBClassifier(
    max_depth=4, n_estimators=100, learning_rate=0.1,
    random_state=42, n_jobs=-1, eval_metric="auc"
)
initial_model.fit(X_bal_proc, y_bal, verbose=False)

X_all_proc        = prep_temp.transform(X)
all_probs         = initial_model.predict_proba(X_all_proc)[:, 1]
df_scored         = df[all_features + ["target"]].copy()
df_scored["prob"] = all_probs

confident_defaulters = df_scored[
    (df_scored["target"] == 0) & (df_scored["prob"] < 0.35)
]
n_defaulters   = len(confident_defaulters)
repaid_sampled = df[df["target"] == 1].sample(n=n_defaulters, random_state=42)
training_df    = pd.concat([repaid_sampled, confident_defaulters])

print(f"Confident defaulters: {n_defaulters:,}")
print(f"Repaid sample:        {n_defaulters:,}")
print(f"Training dataset:     {len(training_df):,}")

X_train_data = training_df[all_features]
y_train_data = training_df["target"]

# ── 4. THREE WAY SPLIT: TRAIN / VALIDATION / TEST ────────────────────────────
# Train 70% — model learns from this
# Validation 15% — Optuna evaluates each trial on this
# Test 15% — final honest evaluation after best params found
X_temp, X_test, y_temp, y_test = train_test_split(
    X_train_data, y_train_data,
    test_size=0.15, random_state=42, stratify=y_train_data
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 85% ≈ 15% of total
)
print(f"\nTrain: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# Fit preprocessor on train only
preprocessor = build_preprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

# ── 5. OPTUNA OBJECTIVE ───────────────────────────────────────────────────────
def objective(trial):
    params = {
        # How deep each tree can go
        # Too shallow: misses interactions. Too deep: overfits.
        "max_depth": trial.suggest_int("max_depth", 3, 8),

        # How much each new tree corrects the previous one
        # Lower = more stable but needs more trees
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # Number of trees — more trees with lower learning rate is generally better
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),

        # Minimum improvement required to create a new branch
        # Higher = more conservative splits, less overfitting
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),

        # Fraction of training rows each tree sees
        # Introduces variety between trees, reduces overfitting
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),

        # Fraction of features each tree sees
        # Forces trees to find different combinations of features
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # Minimum sum of instance weights in a leaf
        # Higher = more conservative, prevents overfitting on small groups
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),

        # L2 regularization — penalises large weights
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),

        # L1 regularization — encourages sparse weights
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
    }

    model = xgb.XGBClassifier(
        **params,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(
        X_train_proc, y_train,
        eval_set=[(X_val_proc, y_val)],
        verbose=False
    )

    y_prob  = model.predict_proba(X_val_proc)[:, 1]
    auc     = roc_auc_score(y_val, y_prob)
    return auc

# ── 6. RUN OPTUNA ─────────────────────────────────────────────────────────────
print("\nRunning Optuna — 100 trials...")
print("This will take a few minutes.\n")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest ROC-AUC on validation set: {study.best_value:.4f}")
print(f"Best parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ── 7. RETRAIN WITH BEST PARAMS ON FULL TRAIN+VAL ────────────────────────────
# Now that we have the best params, retrain on train+val combined
# Test set remains untouched for final evaluation
print("\nRetraining with best parameters on train+val...")

X_trainval_proc = np.vstack([X_train_proc, X_val_proc])
y_trainval      = pd.concat([y_train, y_val])

best_model = xgb.XGBClassifier(
    **study.best_params,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
best_model.fit(X_trainval_proc, y_trainval, verbose=False)

# ── 8. FINAL EVALUATION ON TEST SET ──────────────────────────────────────────
y_prob_test = best_model.predict_proba(X_test_proc)[:, 1]
y_pred_test = best_model.predict(X_test_proc)
test_auc    = roc_auc_score(y_test, y_prob_test)

print(f"\n── Final Test Set Evaluation ──")
print(f"ROC-AUC: {test_auc:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_test, target_names=["Defaulted", "Repaid"]))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
print(f"Correctly declined defaulters:  {tn:,}")
print(f"Defaulters approved by mistake: {fp:,}")
print(f"Good borrowers declined:        {fn:,}")
print(f"Good borrowers approved:        {tp:,}")

# ── 9. COMPARE WITH CURRENT MODEL ────────────────────────────────────────────
print(f"\n── Comparison ──")
with open("model_registry/latest.json") as f:
    latest = json.load(f)
with open(f"model_registry/metadata_{latest['version']}.json") as f:
    current_meta = json.load(f)

print(f"Current model ROC-AUC:  {current_meta['roc_auc']:.4f}")
print(f"Optuna model ROC-AUC:   {test_auc:.4f}")
improvement = test_auc - current_meta["roc_auc"]
if improvement > 0:
    print(f"Improvement: +{improvement:.4f} ✓ — worth retraining with these params")
else:
    print(f"Difference: {improvement:.4f} — current params are already well tuned")

# ── 10. SAVE BEST PARAMS ──────────────────────────────────────────────────────
best_params_path = "model_registry/best_params.json"
with open(best_params_path, "w") as f:
    json.dump({
        "best_params":   study.best_params,
        "val_auc":       round(study.best_value, 4),
        "test_auc":      round(test_auc, 4),
        "n_trials":      100,
        "tuned_at":      pd.Timestamp.now().isoformat()
    }, f, indent=2)

print(f"\nBest params saved to {best_params_path}")
print("Update xgboost_train.py with these params and retrain to deploy.")