# Model evaluation on the held-out test set (Jan 2026).
#
# Two evaluation passes are run:
#
#   Full test set — all Jan 2026 trades; threshold-based precision/recall/F1 curves
#                   saved to threshold_metrics_full.csv
#
#   History-only  — restricted to trades whose combo key has at least one prior
#                   amendment in the historical data; typically smaller but with
#                   significantly higher precision because the model can leverage
#                   _combo_key_hist_pct as a strong signal.
#                   Results saved to threshold_metrics_has_history_only.csv.
#
# Historical counterparty distribution is also built here and attached to downstream
# SHAP exports to help analysts understand which counterparty a flagged trade is likely
# to be amended to.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Full test set — precision, recall, F1 across all probability thresholds

sorted_indices  = np.argsort(-proba_oos)
total_positives = int(y_test.sum())
n_test          = len(y_test)

thresholds = np.linspace(proba_oos.max(), proba_oos.min(), 300)

precisions, recalls, f1_scores, n_flagged_list = [], [], [], []
for t in thresholds:
    mask      = proba_oos >= t
    n_pred    = mask.sum()
    tp        = int(y_test[mask].sum())
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall    = tp / total_positives if total_positives > 0 else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    n_flagged_list.append(n_pred)

precisions, recalls, f1_scores = map(np.array, [precisions, recalls, f1_scores])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.plot(thresholds, precisions, '-', linewidth=2, label='Precision', color='blue')
ax1.plot(thresholds, recalls,    '-', linewidth=2, label='Recall',    color='green')
ax1.set_xlabel('Probability Threshold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Precision and Recall vs Threshold (Test Set)', fontsize=14, fontweight='bold')
ax1.invert_xaxis()
ax1.set_xlim([1, 0])
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_ylim([0, 1])

ax2 = axes[1]
ax2.plot(thresholds, f1_scores, '-', linewidth=2, label='F1 Score', color='purple')
ax2.set_xlabel('Probability Threshold', fontsize=12)
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_title('F1 Score vs Threshold (Test Set)', fontsize=14, fontweight='bold')
ax2.invert_xaxis()
ax2.set_xlim([1, 0])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.show()

print(f"\nTest Set Precision/Recall Summary:")
print(f"Total test samples: {n_test:,}")
print(f"Total positives:    {total_positives:,}")
print(f"Score range:        [{proba_oos.min():.4f}, {proba_oos.max():.4f}]")

best_f1_idx = np.argmax(f1_scores)
print(f"\nOptimal threshold (max F1): {thresholds[best_f1_idx]:.4f}")
print(f"  N Flagged: {n_flagged_list[best_f1_idx]:,}")
print(f"  Precision: {precisions[best_f1_idx]:.4f}")
print(f"  Recall:    {recalls[best_f1_idx]:.4f}")
print(f"  F1 Score:  {f1_scores[best_f1_idx]:.4f}")

step_thresholds = np.arange(round(proba_oos.max(), 2), proba_oos.min() - 0.01, -0.01)

print(f"\nPrecision & Recall at Every 0.01 Threshold Increment:")
print(f"{'Threshold':<12} {'N Flagged':<12} {'TP':<8} {'FP':<8} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-" * 76)

for t in step_thresholds:
    mask      = proba_oos >= t
    n_pred    = mask.sum()
    tp        = int(y_test[mask].sum())
    fp        = n_pred - tp
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall    = tp / total_positives if total_positives > 0 else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"{t:<12.4f} {n_pred:<12,} {tp:<8,} {fp:<8,} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

rows = []
for t in step_thresholds:
    mask      = proba_oos >= t
    n_pred    = mask.sum()
    tp        = int(y_test[mask].sum())
    fp        = n_pred - tp
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall    = tp / total_positives if total_positives > 0 else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    rows.append({
        "Threshold": round(t, 4),
        "N Flagged": n_pred,
        "TP":        tp,
        "FP":        fp,
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1 Score":  round(f1, 4),
    })

pd.DataFrame(rows).to_csv("threshold_metrics_full.csv", index=False)

# Historical counterparty distribution.
# Built from the full df_ml (not just training data) and attached to SHAP outputs so
# analysts can see which counterparty a flagged trade has historically been amended to.

changed_only = df_ml[df_ml["IsCounterpartyChanged"] == 1][
    ["_combo_key", "_combo_key2", "TradeEntryDate", "ChangedToCounterparty"]
].copy()
changed_only["_d"] = pd.to_datetime(changed_only["TradeEntryDate"]).dt.floor("D")
changed_only        = changed_only.dropna(subset=["ChangedToCounterparty"])
changed_sorted      = changed_only.sort_values("_d")

test_combo_keys  = df_ml.loc[X_test.index, "_combo_key"].values
test_combo_keys2 = df_ml.loc[X_test.index, "_combo_key2"].values
test_dates_floor = pd.to_datetime(df_sorted.loc[X_test.index, date_col]).dt.floor("D").values
unique_query_dates = sorted(set(test_dates_floor))


def build_dist_str(cp_counts: dict) -> str:
    if not cp_counts:
        return "No history"
    total = sum(cp_counts.values())
    parts = [
        f"{cp}: {cnt / total * 100:.0f}%"
        for cp, cnt in sorted(cp_counts.items(), key=lambda x: -x[1])
    ]
    return "{" + "; ".join(parts) + "}"


running_counts  = defaultdict(lambda: defaultdict(int))
running_counts2 = defaultdict(lambda: defaultdict(int))
cp_dist_lookup  = {}
cp_dist_lookup2 = {}

ck_arr  = changed_sorted["_combo_key"].values
ck_arr2 = changed_sorted["_combo_key2"].values
cp_arr  = changed_sorted["ChangedToCounterparty"].values
d_arr   = changed_sorted["_d"].values
i, n_changed = 0, len(ck_arr)

for qd in unique_query_dates:
    while i < n_changed and d_arr[i] < qd:
        running_counts[ck_arr[i]][cp_arr[i]]   += 1
        running_counts2[ck_arr2[i]][cp_arr[i]] += 1
        i += 1
    mask = test_dates_floor == qd
    for k in set(test_combo_keys[mask]):
        cp_dist_lookup[(k, qd)]  = build_dist_str(dict(running_counts.get(k, {})))
    for k in set(test_combo_keys2[mask]):
        cp_dist_lookup2[(k, qd)] = build_dist_str(dict(running_counts2.get(k, {})))

cp_dist_series = pd.Series(
    [cp_dist_lookup.get((ck, d), "No history")  for ck, d in zip(test_combo_keys,  test_dates_floor)],
    index=X_test.index,
)
cp_dist2_series = pd.Series(
    [cp_dist_lookup2.get((ck, d), "No history") for ck, d in zip(test_combo_keys2, test_dates_floor)],
    index=X_test.index,
)

# Evaluation restricted to trades with prior amendment history in the combo key.
# The heuristic baseline (raw historical amendment rate) is plotted alongside
# the ML model to quantify the uplift from the learned model.

has_history_mask = cp_dist_series.loc[X_test.index] != "No history"
hist_indices     = X_test.index[has_history_mask]
pos_mask         = has_history_mask.values

proba_oos_hist      = proba_oos[pos_mask]
y_test_hist         = y_test[pos_mask]
heuristic_proba_hist = X_test.loc[hist_indices, "_combo_key_hist_pct"].values
heuristic_proba_prob = heuristic_proba_hist / 100.0


def threshold_metrics(proba, y, thresholds):
    total_pos = int(y.sum())
    n = len(y)
    precisions, recalls, f1_scores = [], [], []
    for t in thresholds:
        mask      = proba >= t
        n_pred    = mask.sum()
        tp        = int(y[mask].sum())
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall    = tp / total_pos if total_pos > 0 else 0.0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return np.array(precisions), np.array(recalls), np.array(f1_scores), total_pos, n


ml_thresholds   = np.linspace(proba_oos_hist.max(),       proba_oos_hist.min(),       300)
heur_thresholds = np.linspace(heuristic_proba_prob.max(), heuristic_proba_prob.min(), 300)

precisions_hist, recalls_hist, f1_scores_hist, total_positives_hist, n_hist = \
    threshold_metrics(proba_oos_hist, y_test_hist, ml_thresholds)

precisions_heur, recalls_heur, f1_scores_heur, _, _ = \
    threshold_metrics(heuristic_proba_prob, y_test_hist, heur_thresholds)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

titles      = ['Precision', 'Recall', 'F1 Score']
ml_series   = [precisions_hist, recalls_hist, f1_scores_hist]
heur_series = [precisions_heur, recalls_heur, f1_scores_heur]

for i, ax in enumerate(axes):
    ax.plot(ml_thresholds,   ml_series[i],   '-',  linewidth=2, label='ML Model',  color=['blue',          'green',          'purple'][i])
    ax.plot(heur_thresholds, heur_series[i], '--', linewidth=2, label='Heuristic', color=['cornflowerblue', 'mediumseagreen', 'mediumpurple'][i])
    ax.invert_xaxis()
    ax.set_xlabel('Probability Threshold', fontsize=10)
    ax.set_ylabel(titles[i], fontsize=11)
    ax.set_title(f'{titles[i]}\n(Has History)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xlim([1, 0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()


def print_summary(label, proba, y, total_pos, n, thresholds, precisions, recalls, f1_scores):
    print(f"\n{label} Summary:")
    print(f"  Score range: [{proba.min():.4f}, {proba.max():.4f}]")
    print(f"  Total test samples (has history): {n:,}  ({n / n_test * 100:.1f}% of full set)")
    print(f"  Total positives: {total_pos:,}")
    best_f1_idx = np.argmax(f1_scores)
    print(f"\n  Optimal threshold (max F1): {thresholds[best_f1_idx]:.4f}")
    print(f"    Precision: {precisions[best_f1_idx]:.4f}")
    print(f"    Recall:    {recalls[best_f1_idx]:.4f}")
    print(f"    F1 Score:  {f1_scores[best_f1_idx]:.4f}")
    step_thresholds = np.arange(round(proba.max(), 2), proba.min() - 0.01, -0.01)
    print(f"\n  {'Threshold':<12} {'N Flagged':<12} {'TP':<8} {'FP':<8} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("  " + "-" * 74)
    for t in step_thresholds:
        mask      = proba >= t
        n_pred    = mask.sum()
        tp        = int(y[mask].sum())
        fp        = n_pred - tp
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall    = tp / total_pos if total_pos > 0 else 0.0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {t:<12.4f} {n_pred:<12,} {tp:<8,} {fp:<8,} {precision:<12.4f} {recall:<12.4f} {f1:<10.4f}")


print_summary(
    "ML Model (Has History)",
    proba_oos_hist, y_test_hist,
    total_positives_hist, n_hist, ml_thresholds,
    precisions_hist, recalls_hist, f1_scores_hist,
)

step_thresholds = np.arange(round(proba_oos_hist.max(), 2), proba_oos_hist.min() - 0.01, -0.01)
rows = []
for t in step_thresholds:
    mask      = proba_oos_hist >= t
    n_pred    = mask.sum()
    tp        = int(y_test_hist[mask].sum())
    fp        = n_pred - tp
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall    = tp / total_positives_hist if total_positives_hist > 0 else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    rows.append({
        "Threshold": round(t, 4),
        "N Flagged": n_pred,
        "TP":        tp,
        "FP":        fp,
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1 Score":  round(f1, 4),
    })

pd.DataFrame(rows).to_csv("threshold_metrics_has_history_only.csv", index=False)
