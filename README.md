# Counterparty Amendment Prediction

> **Demonstration only.** The original trade data and pipeline is not available in this repository. Scripts are provided to illustrate the methodology.

## What This Does

This model scores each trade at booking time with a probability that its counterparty will subsequently change, so the daily review list is ranked and the real amendments surface at the top.

When a counterparty amendment is not caught early, the trade gets sent to the wrong party and only gets corrected the night of or days later. By that point it has already bounced around between counterparties, generated manual work, and introduced the risk of a settlement failure. Catching it at booking time reduces financial friction, prevents wrong trades from going out, cuts manual errors, and improves first-time-right rates.

The class imbalance is severe: roughly 1 amended trade for every 1,000 clean ones. Despite that, the model has empirically achieved 90% precision at 60% recall on out-of-sample data.

## Approach

The model is a LightGBM binary classifier trained on historical trade data. Because raw trade attributes alone carry little signal, the core of the pipeline is feature engineering based on historical group behaviour.

Each trade is assigned to an identity group using eight attributes: Counterparty, Fund, ExecutionVenue, TradeCategory, ClearingHouse, CounterpartyRisk, InstrumentCcy, and TradeType. A second group excludes Counterparty, capturing the structural context shared across counterparties for the same trade profile.

For each group, the model sees:

- Historical amendment count and rate
- Number of distinct days with at least one amendment
- Number of distinct counterparties seen on clean trades
- Concentration of the dominant counterparty on clean trades
- Full historical counterparty distribution

All features are computed with a strict one-day lag. The current day's trades never contribute to their own features.

## Model and Evaluation

Training and evaluation are strictly temporal with no shuffling:

| Split | Period | Role |
|---|---|---|
| Train + Validation | Jan 2025 -- Dec 2025 | Training and tuning |
| Test (OOS) | Jan 2026 | Final evaluation only |

Hyperparameters are tuned with Optuna (200 trials, TPE sampler) across two expanding folds. The optimisation target is Average Precision at the top 0.1% of trades by score, which is the operationally relevant regime.

The evaluation output is a threshold table: for every decision threshold from the maximum score downward, it reports the number of flagged trades, true positives, false positives, precision, recall, and F1. This lets operations choose the review list size based on capacity.

SHAP values are computed in probability-point space. Each flagged trade exports its top five features and their individual contribution to the predicted probability, relative to the population baseline.

## Repository Contents

| File | Description |
|---|---|
| `data_ingestion.py` | Celonis connection and PQL data pull |
| `preprocessing.py` | Date filtering, recency cutoff, internal trade removal |
| `feature_engineering.py` | Combo key construction and historical feature computation |
| `train.py` | Optuna tuning and final model training |
| `evaluate.py` | Threshold-based evaluation and counterparty distribution |
| `shap_export.py` | SHAP explainability and flagged trade export |
| `best_lgbm_params_currentbest.json` | Best hyperparameters from Optuna |
