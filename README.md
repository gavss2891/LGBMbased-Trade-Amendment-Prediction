# Counterparty Amendment Prediction

## Overview

This model predicts, at the time of booking, whether a trade's counterparty will subsequently be amended. The output is a probability score per trade. Scores are used to produce a ranked daily review list for operations, concentrating genuine amendments at the top.

> **Demonstration only.** The original trade data is not available in this repository and all model outputs have been omitted to maintain confidentiality. The scripts are provided to illustrate the methodology and pipeline structure.

---

## Repository Contents

| File | Description |
|---|---|
| `data_ingestion.py` | Celonis connection, attribute discovery, and PQL data pull |
| `preprocessing.py` | Date filtering, recency cutoff, internal trade removal, attribute selection |
| `feature_engineering.py` | Combo-key construction and time-aware historical feature computation |
| `train.py` | Experiment configuration, data split, Optuna tuning, and final model training |
| `evaluate.py` | Threshold-based precision/recall/F1 evaluation and counterparty distribution building |
| `shap_export.py` | SHAP explainability in probability space and flagged trade export |
| `best_lgbm_params_currentbest.json` | Best hyperparameters from Optuna tuning |

---

## Environment Setup

The pipeline connects to Celonis via `pycelonis`. Credentials are loaded from a `.env` file in the working directory:

```
CELONIS_URL=
CELONIS_API_TOKEN=
CELONIS_POOL_ID=
CELONIS_SPACE_ID=
CELONIS_PACKAGE_ID=
CELONIS_KNOWLEDGE_MODEL_ID=
```

Install dependencies:

```bash
pip install --extra-index-url=https://pypi.celonis.cloud/ pycelonis
pip install lightgbm optuna shap pandas numpy matplotlib python-dotenv
```

---

## Running the Pipeline

The scripts are designed to run sequentially in a single Python session. Execute them in order:

```
data_ingestion.py → preprocessing.py → feature_engineering.py → train.py → evaluate.py → shap_export.py
```

The Optuna tuning block in `train.py` can be skipped if `best_lgbm_params_currentbest.json` is already present — the final training block loads parameters from that file directly.

---

## Data

### Source

Trade data is queried from the `BASE_TRADE_FOR_ML` record in a Celonis Knowledge Model using PQL. The query pulls attributes covering trade structure, booking metadata, counterparty, and the target flag `IsCounterpartyChanged`.

### Filtering

- Trades from **1 January 2025** onwards only.
- The most recent **14 days** are excluded. Most amendments arrive within two weeks of booking, so trades inside that window do not yet have settled labels. Including them would cause genuine positives to appear as negatives.
- All **internal trades** are excluded (`IsInternal == 1`). The concept of a counterparty amendment does not apply to them.

---

## Feature Engineering

Feature engineering is the core of the pipeline. Raw trade attributes are not fed directly to the model. Instead, each trade is represented by its group's historical behaviour.

### Trade Identity Groups (Combo Keys)

Eight attributes define a trade's identity:

```
Counterparty | Fund | ExecutionVenue | TradeCategory |
ClearingHouse | CounterpartyRisk | InstrumentCcy | TradeType
```

These are concatenated with a `|||` separator to form `_combo_key`. A second key, `_combo_key2`, is constructed identically but with `Counterparty` excluded — this captures the structural context shared across counterparties for a given trade profile.

### Combo Key Features (full identity, including counterparty)

| Feature | Description |
|---|---|
| `_combo_key_hist_sum` | Total amendments on this exact booking combination, historically |
| `_combo_key_hist_cnt` | Total trades seen on this combination, historically |
| `_combo_key_hist_daycounts` | Number of distinct calendar days on which at least one amendment occurred |
| `_combo_key_hist_pct` | Amendment rate: `hist_sum / hist_cnt × 100` |

### Combo Key 2 Features (identity excluding counterparty)

| Feature | Description |
|---|---|
| `_combo_key2_hist_n_unique_cp` | Number of distinct counterparties on clean trades for this structure |
| `_combo_key2_hist_dom_cp_pct` | Percentage share held by the single most common counterparty on clean trades |
| `_combo_key2_hist_cp_dist` | Full string distribution: `CP: count (%) | CP: count (%)...` sorted by frequency |

### Leakage Prevention

All features use a strict lag of 1 day. The aggregation is computed at the daily level, cumulated through day `d-1`. The current day's trades never contribute to their own features.

---

## Model

### Algorithm

LightGBM gradient-boosted decision trees with binary cross-entropy loss. Categorical string columns are cast to `category` dtype before training and LightGBM's native categorical handling is used. Category sets are aligned between train and test splits to avoid unseen-category issues.

### Train / Validation / Test Split

All splits are strictly temporal — no shuffling.

| Split | Date Range | Purpose |
|---|---|---|
| Burn-in (excluded) | First 5% of sorted data | Removed — features are near-empty for the earliest trades |
| Train + Validation | Jan 2025 – Dec 2025 | Model training and hyperparameter tuning |
| Test (OOS) | Jan 2026 | Final evaluation only — never seen during any training or tuning step |

Two expanding folds are used during hyperparameter tuning:

- **Fold Nov 2025**: Train on data before Nov 2025, validate on Nov 2025
- **Fold Dec 2025**: Train on data before Dec 2025, validate on Dec 2025

---

## Hyperparameter Tuning

Tuning is performed with **Optuna** using the TPE sampler over 200 trials (first 80 random). The study is persisted to an SQLite file so interrupted runs resume automatically.

### Optimisation Target

**Average Precision at a rate of 0.1% (AP@0.001)** — evaluates the quality of the model's ranking within the top 0.1% of trades by score, the regime that matters operationally.

### Search Space

| Parameter | Range |
|---|---|
| `num_leaves` | 64 – 1,000 |
| `max_depth` | 5 – 15 |
| `min_gain_to_split` | 0.0 – 5.0 |
| `min_data_in_leaf` | 1 – 1,000 |
| `learning_rate` | 0.005 – 0.08 (log scale) |
| `feature_fraction` | 0.6 – 1.0 |
| `bagging_fraction` | 0.6 – 1.0 |
| `bagging_freq` | 0 – 20 |
| `lambda_l1` | 1e-4 – 10.0 (log scale) |
| `lambda_l2` | 1e-4 – 10.0 (log scale) |
| `min_sum_hessian_in_leaf` | 1e-3 – 10.0 (log scale) |

### Best Parameters

```json
{
  "num_leaves": 414,
  "max_depth": 15,
  "min_gain_to_split": 3.660,
  "min_data_in_leaf": 599,
  "learning_rate": 0.01061,
  "feature_fraction": 0.6624,
  "bagging_fraction": 0.6232,
  "bagging_freq": 18,
  "lambda_l1": 0.3470,
  "lambda_l2": 0.1013,
  "min_sum_hessian_in_leaf": 0.00121
}
```

---

## Final Model Training

The final model is trained on the full train+val set (Jan 2025 – Dec 2025). The last 10% of that set by time governs early stopping only and is not a holdout for evaluation.

---

## Output Files

> Output files are not included in this repository to maintain confidentiality.

### Threshold Tables

`threshold_metrics_full.csv` and `threshold_metrics_has_history_only.csv` contain, for every threshold from the maximum predicted score down to near zero:

| Column | Description |
|---|---|
| `Threshold` | Decision threshold applied to predicted probability |
| `N Flagged` | Number of trades flagged at or above this threshold |
| `TP` | True positives |
| `FP` | False positives |
| `Precision` | TP / (TP + FP) |
| `Recall` | TP / total amendments in test set |
| `F1 Score` | Harmonic mean of precision and recall |

### SHAP Output Files

Each row corresponds to one flagged trade. Key columns:

| Column | Description |
|---|---|
| `NBInternal` | Trade identifier |
| `TradeEntryDate` | Booking date |
| `predicted_proba` | Model's predicted amendment probability |
| `actual_label` | Ground truth (1 = was amended) |
| `_combo_key` | Full identity group string |
| `_combo_key_hist_pct` | Historical amendment rate for this group |
| `_combo_key2_hist_cp_dist` | Historical counterparty distribution for this structure |
| `ChangedToCounterparty_dist` | Which counterparty it was actually changed to (if amended) |
| `top1` … `top5` | Top 5 SHAP-ranked features: `feature = value (±contribution in pp)` |

SHAP contributions are expressed in **probability-point space**. Each contribution is the number of percentage points that feature adds to or subtracts from the predicted probability, relative to the population baseline.
