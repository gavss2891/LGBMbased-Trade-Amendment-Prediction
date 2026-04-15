# SHAP explainability and flagged trade export.
#
# SHAP values are computed in log-odds space (TreeExplainer default) then rescaled
# into probability space, so each feature contribution represents percentage points
# added to or subtracted from the predicted probability relative to the population
# baseline. This makes the decomposition directly readable without any transformation.
#
# Three output files are produced:
#
#   top500_overall_full_data_updated_features.csv
#       Top 500 trades by predicted probability across the full test set, with top-5
#       SHAP feature contributions and historical counterparty distribution.
#
#   per_day_shap_with_history_only.csv
#       Per day, top 50 highest-probability trades filtered to predicted_proba > 0.01
#       AND where the counterparty had prior amendment history. Highest-confidence
#       actionable flags.
#
#   per_day_shap_all_trades.csv
#       Same per-day top 50, but filtered only on predicted_proba > 0.01 with no
#       history restriction.
#
# Output columns — Identity:
#   NBInternal               Internal trade identifier
#   TradeEntryDate           Trade entry timestamp
#   _combo_key               Full combination of trade identity attributes
#
# Output columns — Prediction:
#   predicted_proba          Model's predicted amendment probability
#   actual_label             Ground truth (1 = amended, 0 = not amended)
#   ChangedToCounterparty_dist  Historical distribution of counterparties this group
#                               was amended to (from training data, up to trade date)
#
# Output columns — Engineered features:
#   _combo_key_hist_sum        Past amendment count for this combo
#   _combo_key_hist_cnt        Past total trade count for this combo
#   _combo_key_hist_daycounts  Distinct days with at least one amendment
#   _combo_key_hist_pct        Historical amendment rate (%)
#   _combo_key2_hist_n_unique_cp  Distinct counterparties seen historically
#   _combo_key2_hist_dom_cp_pct   Share of the most frequent counterparty
#   _combo_key2_hist_cp_dist      Full counterparty distribution string
#
# Output columns — SHAP:
#   top1 … top5   Top features driving the prediction in format:
#                 feature = value (±contribution in probability points)

import shap
import numpy as np
import pandas as pd
from scipy.special import expit

proba_series = pd.Series(proba_oos, index=X_test.index)

# Top 500 trades overall — SHAP explanations for the highest-probability flags

top500_idx   = proba_series.nlargest(500).index
X_test_small = X_test.loc[top500_idx]
proba_small  = proba_series.loc[top500_idx].to_numpy()
y_small      = pd.Series(y_test, index=X_test.index).loc[top500_idx].to_numpy()

explainer    = shap.TreeExplainer(model_oos)
shap_values  = explainer.shap_values(X_test_small)
base_logodds = explainer.expected_value
base_prob    = float(expit(base_logodds))

# Rescale SHAP from log-odds to probability space
logodds_sum  = shap_values.sum(axis=1, keepdims=True)
scale        = np.where(
    np.abs(logodds_sum) > 1e-10,
    (proba_small[:, None] - base_prob) / logodds_sum,
    0.0,
)
prob_shap_mat = shap_values * scale

meta = df_ml.loc[top500_idx, [
    "NBInternal", "NBExternal", "TradeEntryDate",
    "_combo_key", "_combo_key_hist_cnt", "_combo_key_hist_pct",
    "_combo_key2_hist_n_unique_cp", "_combo_key2_hist_dom_cp_pct",
]].copy()
meta["ChangedToCounterparty_dist"]  = cp_dist_series.loc[top500_idx].values
meta["ChangedToCounterparty_dist2"] = cp_dist2_series.loc[top500_idx].values

feat_names = np.array(X_test_small.columns.tolist())
feat_vals  = X_test_small.to_numpy(dtype=object)

TOP_N      = 5
rows       = []
top_n_idx  = np.argpartition(-np.abs(prob_shap_mat), TOP_N, axis=1)[:, :TOP_N]

for i, idx in enumerate(top500_idx):
    record = {
        "NBInternal":                   meta.at[idx, "NBInternal"],
        "NBExternal":                   meta.at[idx, "NBExternal"],
        "TradeEntryDate":               meta.at[idx, "TradeEntryDate"],
        "_combo_key":                   meta.at[idx, "_combo_key"],
        "ChangedToCounterparty_dist":   meta.at[idx, "ChangedToCounterparty_dist"],
        "ChangedToCounterparty_dist2":  meta.at[idx, "ChangedToCounterparty_dist2"],
        "_combo_key_hist_cnt":          meta.at[idx, "_combo_key_hist_cnt"],
        "_combo_key_hist_pct":          meta.at[idx, "_combo_key_hist_pct"],
        "_combo_key2_hist_n_unique_cp": meta.at[idx, "_combo_key2_hist_n_unique_cp"],
        "_combo_key2_hist_dom_cp_pct":  meta.at[idx, "_combo_key2_hist_dom_cp_pct"],
        "predicted_proba":              round(proba_small[i], 6),
        "actual_label":                 int(y_small[i]),
    }
    cands = top_n_idx[i]
    cands = cands[np.argsort(-np.abs(prob_shap_mat[i, cands]))]
    for rank, fi in enumerate(cands, start=1):
        contrib = prob_shap_mat[i, fi]
        record[f"top{rank}"] = f"{feat_names[fi]} = {feat_vals[i, fi]} ({contrib:+.4f})"
    rows.append(record)

top500_overall = (
    pd.DataFrame(rows)
    .assign(TradeDate=lambda x: pd.to_datetime(x["TradeEntryDate"], unit="ms").dt.date)
    .sort_values("predicted_proba", ascending=False)
    .reset_index(drop=True)
)

top500_overall.to_csv("top500_overall_full_data_updated_features.csv", index=False)
top500_overall.head()

# Engineered features included in the per-day filtered exports
engineered_cols = [
    "_combo_key_hist_sum",
    "_combo_key_hist_cnt",
    "_combo_key_hist_daycounts",
    "_combo_key_hist_pct",
    "_combo_key2_hist_n_unique_cp",
    "_combo_key2_hist_dom_cp_pct",
]

# Per-day top 50 — filtered to predicted_proba > 0.01 AND counterparty had history.
# This is the highest-confidence actionable subset for daily operations review.

mask         = (proba_series > 0.01) & (cp_dist_series.loc[X_test.index] != "No history")
filtered_idx = X_test.index[mask]

X_filt     = X_test.loc[filtered_idx]
proba_filt = proba_series.loc[filtered_idx].to_numpy()
y_filt     = pd.Series(y_test, index=X_test.index).loc[filtered_idx].to_numpy()

shap_vals_f   = explainer.shap_values(X_filt)
logodds_sum_f = shap_vals_f.sum(axis=1, keepdims=True)
scale_f       = np.where(
    np.abs(logodds_sum_f) > 1e-10,
    (proba_filt[:, None] - base_prob) / logodds_sum_f,
    0.0,
)
prob_shap_f = shap_vals_f * scale_f

meta_f = df_ml.loc[filtered_idx, ["NBInternal", "TradeEntryDate", "_combo_key", "Counterparty"] + engineered_cols].copy()
meta_f["ChangedToCounterparty_dist"] = cp_dist_series.loc[filtered_idx].values
meta_f["_combo_key2_hist_cp_dist"]   = df.loc[filtered_idx, "_combo_key2_hist_cp_dist"].values

feat_vals_f = X_filt.to_numpy(dtype=object)
top_n_idx_f = np.argpartition(-np.abs(prob_shap_f), TOP_N, axis=1)[:, :TOP_N]

rows_f = []
for i, idx in enumerate(filtered_idx):
    record = {
        "NBInternal":                 meta_f.at[idx, "NBInternal"],
        "TradeEntryDate":             meta_f.at[idx, "TradeEntryDate"],
        "_combo_key":                 meta_f.at[idx, "_combo_key"],
        "ChangedToCounterparty_dist": meta_f.at[idx, "ChangedToCounterparty_dist"],
        "ChangedToCounterparty":      df_ml.at[idx, "ChangedToCounterparty"],
        "predicted_proba":            round(proba_filt[i], 6),
        "actual_label":               int(y_filt[i]),
    }
    for col in engineered_cols:
        record[col] = meta_f.at[idx, col]
    record["_combo_key2_hist_cp_dist"] = meta_f.at[idx, "_combo_key2_hist_cp_dist"]
    cands = top_n_idx_f[i]
    cands = cands[np.argsort(-np.abs(prob_shap_f[i, cands]))]
    for rank, fi in enumerate(cands, start=1):
        record[f"top{rank}"] = f"{feat_names[fi]} = {feat_vals_f[i, fi]} ({prob_shap_f[i, fi]:+.4f})"
    rows_f.append(record)

filtered_top50_history = (
    pd.DataFrame(rows_f)
    .assign(TradeDate=lambda x: pd.to_datetime(x["TradeEntryDate"], unit="ms").dt.date)
    .sort_values(["TradeDate", "predicted_proba"], ascending=[True, False])
    .groupby("TradeDate", sort=False).head(50)
    .reset_index(drop=True)
)

filtered_top50_history = pd.concat([
    pd.concat([grp, pd.DataFrame([{}])], ignore_index=True)
    for _, grp in filtered_top50_history.groupby("TradeDate", sort=False)
], ignore_index=True)

filtered_top50_history.to_csv("per_day_shap_with_history_only.csv", index=False)

# Per-day top 50 — filtered to predicted_proba > 0.01 only, no history restriction.

mask         = proba_series > 0.01
filtered_idx = X_test.index[mask]

X_filt     = X_test.loc[filtered_idx]
proba_filt = proba_series.loc[filtered_idx].to_numpy()
y_filt     = pd.Series(y_test, index=X_test.index).loc[filtered_idx].to_numpy()

shap_vals_f   = explainer.shap_values(X_filt)
logodds_sum_f = shap_vals_f.sum(axis=1, keepdims=True)
scale_f       = np.where(
    np.abs(logodds_sum_f) > 1e-10,
    (proba_filt[:, None] - base_prob) / logodds_sum_f,
    0.0,
)
prob_shap_f = shap_vals_f * scale_f

meta_f = df_ml.loc[filtered_idx, ["NBInternal", "TradeEntryDate", "_combo_key", "Counterparty"] + engineered_cols].copy()
meta_f["ChangedToCounterparty_dist"] = cp_dist_series.loc[filtered_idx].values
meta_f["_combo_key2_hist_cp_dist"]   = df.loc[filtered_idx, "_combo_key2_hist_cp_dist"].values

feat_vals_f = X_filt.to_numpy(dtype=object)
top_n_idx_f = np.argpartition(-np.abs(prob_shap_f), TOP_N, axis=1)[:, :TOP_N]

rows_f = []
for i, idx in enumerate(filtered_idx):
    record = {
        "NBInternal":                 meta_f.at[idx, "NBInternal"],
        "TradeEntryDate":             meta_f.at[idx, "TradeEntryDate"],
        "_combo_key":                 meta_f.at[idx, "_combo_key"],
        "ChangedToCounterparty_dist": meta_f.at[idx, "ChangedToCounterparty_dist"],
        "ChangedToCounterparty":      df_ml.at[idx, "ChangedToCounterparty"],
        "predicted_proba":            round(proba_filt[i], 6),
        "actual_label":               int(y_filt[i]),
    }
    for col in engineered_cols:
        record[col] = meta_f.at[idx, col]
    record["_combo_key2_hist_cp_dist"] = meta_f.at[idx, "_combo_key2_hist_cp_dist"]
    cands = top_n_idx_f[i]
    cands = cands[np.argsort(-np.abs(prob_shap_f[i, cands]))]
    for rank, fi in enumerate(cands, start=1):
        record[f"top{rank}"] = f"{feat_names[fi]} = {feat_vals_f[i, fi]} ({prob_shap_f[i, fi]:+.4f})"
    rows_f.append(record)

filtered_top50_all = (
    pd.DataFrame(rows_f)
    .assign(TradeDate=lambda x: pd.to_datetime(x["TradeEntryDate"], unit="ms").dt.date)
    .sort_values(["TradeDate", "predicted_proba"], ascending=[True, False])
    .groupby("TradeDate", sort=False).head(50)
    .reset_index(drop=True)
)

filtered_top50_all = pd.concat([
    pd.concat([grp, pd.DataFrame([{}])], ignore_index=True)
    for _, grp in filtered_top50_all.groupby("TradeDate", sort=False)
], ignore_index=True)

filtered_top50_all.to_csv("per_day_shap_all_trades.csv", index=False)
