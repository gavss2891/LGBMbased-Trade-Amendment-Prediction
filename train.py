# Model configuration, data splitting, hyperparameter tuning, and final training.
#
# Data split — all splits are strictly temporal, no shuffling:
#   Train / Val  Jan 2025 – Dec 2025   Optuna tuning (expanding CV) + final model training
#   Test (OOS)   Jan 2026              Held-out evaluation; never seen during tuning
#
# Two expanding folds are used for tuning:
#   Fold Nov 2025 — train on data before Nov, validate on Nov
#   Fold Dec 2025 — train on data before Dec, validate on Dec
#
# Expanding folds are preferred over rolling windows because the historical features
# only become meaningful with more training data.
#
# Hyperparameter tuning — Optuna TPE sampler, 200 trials (first 80 random).
# The study is persisted to an SQLite file so interrupted runs resume automatically.
# Optimisation target: Average Precision at the top 0.1% of trades by score (AP@0.001).
# This targets the operational regime where reviewers can act on only a small fraction
# of daily trade flow.
#
# Final model — trained on the full train/val set using parameters loaded from
# best_lgbm_params_currentbest.json. The last 10% of the train/val set (by time) acts
# as an early-stopping validation set only; it is not a held-out evaluation split.
# The saved model is written to model_oos.txt.
#
# Skip the Optuna section if best_lgbm_params_currentbest.json is already present.

import optuna
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

# Experiment configuration
RANDOM_STATE          = 42
BURN_IN_FRAC          = 0.05
N_TRIALS              = 200
N_STARTUP_TRIALS      = 80
MAX_BOOST_ROUNDS      = 5000
EARLY_STOPPING_ROUNDS = 200
AP_RATE               = 0.001  # top 0.1% of rows

STUDY_NAME  = "lgbm_optuna_full_data_updated_features"
STUDY_DB    = "sqlite:///optuna_full_data_updated_features.db"
HISTORY_CSV = "tuning_history_full_data_updated_features.csv"
BEST_PARAMS = "best_lgbm_params_full_data_updated_features.json"

VAL_MONTHS = [
    (pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-30")),
    (pd.Timestamp("2026-12-01"), pd.Timestamp("2026-12-31")),
]
OOS_START = pd.Timestamp("2026-01-01")
OOS_END   = pd.Timestamp("2026-01-31")

# Metric helpers

def ap_at_rate(y_true, y_score, rate):
    y_true, y_score = np.asarray(y_true, int), np.asarray(y_score, float)
    k = max(1, min(int(round(rate * y_true.size)), y_true.size))
    y = y_true[np.argsort(-y_score)[:k]]
    hits, s = 0, 0.0
    for i, yi in enumerate(y, 1):
        if yi:
            hits += 1
            s += hits / i
    return float(s / hits) if hits else 0.0


def precision_recall_at_rate(y_true, y_score, rate):
    y_true, y_score = np.asarray(y_true, int), np.asarray(y_score, float)
    k   = max(1, min(int(round(rate * y_true.size)), y_true.size))
    tp  = int(y_true[np.argsort(-y_score)[:k]].sum())
    tot = int(y_true.sum())
    return tp / k, (tp / tot if tot else 0.0), tp, k - tp, k


def make_categorical_consistent(X_tr, X_te):
    X_tr, X_te = X_tr.copy(), X_te.copy()
    cat_cols = [c for c in X_tr if not pd.api.types.is_numeric_dtype(X_tr[c])]
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)
    return X_tr, X_te, cat_cols


base_params = dict(
    objective="binary", boosting_type="gbdt",
    seed=RANDOM_STATE, verbosity=-1,
    metric="binary_logloss", num_threads=-1,
)


def fit_fold_predict(hp, X_tr, y_tr, X_va, y_va, cat_cols):
    p      = {**base_params, **hp}
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols or None, free_raw_data=False)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=cat_cols or None, free_raw_data=False)
    model  = lgb.train(
        p, dtrain, MAX_BOOST_ROUNDS, [dval], ["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    return model.predict(X_va, num_iteration=model.best_iteration)


# Data preparation — burn in the earliest 5% of sorted data where features are near-empty
burn_in_idx = int(len(df_ml) * BURN_IN_FRAC)
df_sorted   = df_ml.iloc[burn_in_idx:].copy().drop(columns=["_combo_key", "_combo_key2"], errors="ignore")

non_feature_cols = [c for c in ["NBInternal", "NBExternal", "AmendedOnDay", "ChangedToCounterparty", "Business"] if c in df_sorted.columns]
X_all  = df_sorted.drop(columns=[target_col, date_col] + non_feature_cols)
y_all  = df_sorted[target_col].astype(int).to_numpy()
dates  = df_sorted[date_col].values

tv_mask   = dates < OOS_START
test_mask = (dates >= OOS_START) & (dates < OOS_END)

X_trainval = X_all.loc[tv_mask]
y_trainval = y_all[tv_mask]
tv_dates   = dates[tv_mask]

X_test = X_all.loc[test_mask]
y_test = y_all[test_mask]

X_trainval, X_test, cat_cols = make_categorical_consistent(X_trainval, X_test)

tuning_splits = []
for val_start, val_end in VAL_MONTHS:
    tr_idx = np.where(tv_dates < val_start)[0]
    va_idx = np.where((tv_dates >= val_start) & (tv_dates < val_end))[0]
    if len(tr_idx) and len(va_idx):
        tuning_splits.append((tr_idx, va_idx))

print(f"Tuning folds: {len(tuning_splits)}  |  trainval rows: {len(X_trainval):,}  |  test rows: {len(X_test):,}")
for (tr, va), (vs, ve) in zip(tuning_splits, VAL_MONTHS):
    print(f"  Fold {vs.strftime('%b %Y')}: train={len(tr):,}  val={len(va):,}")

# Monthly amendment count across the full dataset — used to sense-check label distribution
amended        = df_ml[df_ml["IsCounterpartyChanged"] == 1].copy()
amended["Month"] = pd.to_datetime(amended["TradeEntryDate"]).dt.to_period("M")
monthly_counts = amended.groupby("Month").size().sort_index()

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(monthly_counts.index.astype(str), monthly_counts.values, color="steelblue", edgecolor="white")

for bar, val in zip(bars, monthly_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + monthly_counts.max() * 0.01,
            str(val), ha="center", va="bottom", fontsize=8)

ax.set_title("Counterparty Amendments per Month")
ax.set_xlabel("Month")
ax.set_ylabel("Count")
ax.set_xticklabels(monthly_counts.index.astype(str), rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Hyperparameter tuning with Optuna.
# Skip this block if best_lgbm_params_currentbest.json already exists and jump to
# final model training below.

best_params_, best_oof_ap = None, -np.inf


def objective(trial):
    global best_params_, best_oof_ap
    hp = {
        "num_leaves":              trial.suggest_int  ("num_leaves",              64, 1000),
        "max_depth":               trial.suggest_int  ("max_depth",               5, 15),
        "min_gain_to_split":       trial.suggest_float("min_gain_to_split",       0.0, 5.0),
        "min_data_in_leaf":        trial.suggest_int  ("min_data_in_leaf",        1, 1000),
        "learning_rate":           trial.suggest_float("learning_rate",           0.005, 0.08, log=True),
        "feature_fraction":        trial.suggest_float("feature_fraction",        0.6, 1.0),
        "bagging_fraction":        trial.suggest_float("bagging_fraction",        0.6, 1.0),
        "bagging_freq":            trial.suggest_int  ("bagging_freq",            0, 20),
        "lambda_l2":               trial.suggest_float("lambda_l2",               1e-4, 10.0, log=True),
        "lambda_l1":               trial.suggest_float("lambda_l1",               1e-4, 10.0, log=True),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
    }
    oof = np.full(len(y_trainval), np.nan)
    for tr_idx, va_idx in tuning_splits:
        oof[va_idx] = fit_fold_predict(
            hp,
            X_trainval.iloc[tr_idx], y_trainval[tr_idx],
            X_trainval.iloc[va_idx], y_trainval[va_idx],
            cat_cols,
        )
    mask = ~np.isnan(oof)
    ap   = ap_at_rate(y_trainval[mask], oof[mask], AP_RATE)
    p, r, tp, fp, k = precision_recall_at_rate(y_trainval[mask], oof[mask], AP_RATE)
    if ap > best_oof_ap:
        best_oof_ap, best_params_ = ap, hp.copy()
    print(f"[{trial.number + 1:>3}] AP@{AP_RATE:.4g}={ap:.6f}  P={p:.4f}  R={r:.4f}  TP={tp} FP={fp}", flush=True)
    return -ap


def _save_csv(study, trial):
    (study.trials_dataframe(attrs=("number", "value", "params", "state"))
          .rename(columns={"number": "iter", "value": "loss"})
          .assign(ap_rate=lambda d: -d["loss"])
          .to_csv(HISTORY_CSV, index=False))


_resuming = False
try:
    _tmp      = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)
    _resuming = len(_tmp.trials) > 0
except Exception:
    pass

_sampler = optuna.samplers.TPESampler(
    seed=None if _resuming else RANDOM_STATE,
    n_startup_trials=N_STARTUP_TRIALS,
)
study = optuna.create_study(
    direction="minimize",
    sampler=_sampler,
    study_name=STUDY_NAME,
    storage=STUDY_DB,
    load_if_exists=True,
)

if study.trials:
    best_oof_ap  = -study.best_value
    best_params_ = dict(study.best_params)
    print(f"Resuming from trial {len(study.trials)}  (best AP so far: {best_oof_ap:.6f})")

remaining = N_TRIALS - len(study.trials)
if remaining > 0:
    study.optimize(objective, n_trials=remaining, callbacks=[_save_csv], show_progress_bar=True)
else:
    print(f"Already completed {N_TRIALS} trials — nothing to run.")

best_params_ = dict(study.best_params)
best_oof_ap  = -study.best_value

json.dump({"params": best_params_, "ap_rate": best_oof_ap}, open(BEST_PARAMS, "w"))
print(f"\nBest OOF AP@{AP_RATE:.4g}: {best_oof_ap:.6f}")
print("Best params:", best_params_)

# Final model training on the full train/val set.
# Parameters are loaded from best_lgbm_params_currentbest.json.
# To use freshly tuned parameters instead, replace the json.load call with best_params_
# from the study object above.

best_params_ = json.load(open("best_lgbm_params_currentbest.json"))["params"]
oos_params   = {**base_params, **best_params_}

tail_n     = max(1, int(0.1 * len(X_trainval)))
dtrain_oos = lgb.Dataset(X_trainval.iloc[:-tail_n], label=y_trainval[:-tail_n],
                          categorical_feature=cat_cols or None, free_raw_data=False)
dval_oos   = lgb.Dataset(X_trainval.iloc[-tail_n:],  label=y_trainval[-tail_n:],
                          reference=dtrain_oos, categorical_feature=cat_cols or None, free_raw_data=False)

model_oos = lgb.train(
    params=oos_params,
    train_set=dtrain_oos,
    num_boost_round=MAX_BOOST_ROUNDS,
    valid_sets=[dval_oos],
    valid_names=["val"],
    callbacks=[
        lgb.early_stopping(500, first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

proba_oos = model_oos.predict(X_test, num_iteration=model_oos.best_iteration)

oos_ap                   = ap_at_rate(y_test, proba_oos, rate=AP_RATE)
oos_p, oos_r, oos_tp, oos_fp, oos_k = precision_recall_at_rate(y_test, proba_oos, AP_RATE)

print(f"Final OOS trades:    {len(y_test):,}")
print(f"Final OOS positives: {int(y_test.sum()):,}")
print(f"Date range:          {dates[test_mask].min()} \u2192 {dates[test_mask].max()}")
print(f"\nAP@{AP_RATE:.4g}:  {oos_ap:.6f}")
print(f"P@{AP_RATE:.4g}:   {oos_p:.4f}")
print(f"R@{AP_RATE:.4g}:   {oos_r:.4f}")
print(f"TP={oos_tp}  FP={oos_fp}  K={oos_k}")

model_oos.save_model("model_oos.txt")
model_oos = lgb.Booster(model_file="model_oos.txt")
