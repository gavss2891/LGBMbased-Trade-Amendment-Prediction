# Historical combo-key feature engineering.
#
# Each trade is represented by its group's historical behaviour rather than its raw
# attributes. Two group keys are built:
#
#   _combo_key  — full identity (Counterparty | Fund | ExecutionVenue | TradeCategory |
#                 ClearingHouse | CounterpartyRisk | InstrumentCcy | TradeType)
#   _combo_key2 — same dimensions with Counterparty excluded; captures the structural
#                 context shared across counterparties for a given trade profile
#
# All features are computed from data strictly before the trade's booking date
# (lag = 1 calendar day), so no future information leaks into training.
#
# Features produced:
#   _combo_key_hist_sum        — amendment count for this exact group, historically
#   _combo_key_hist_cnt        — total trades seen for this group, historically
#   _combo_key_hist_daycounts  — distinct calendar days on which an amendment occurred
#   _combo_key_hist_pct        — amendment rate: hist_sum / hist_cnt * 100
#   _combo_key2_hist_n_unique_cp  — distinct counterparties seen on clean (unamended)
#                                   trades for this structural group
#   _combo_key2_hist_dom_cp_pct   — share held by the single most common counterparty
#                                   on clean trades; high values indicate a predictable
#                                   counterparty expectation
#
# Trades with no prior history for their combo key receive NaN, which LightGBM
# handles natively via its missing-value split mechanism.

import pandas as pd
import numpy as np

target_col = 'IsCounterpartyChanged'
date_col   = 'TradeEntryDate'
lag_days   = 1

SEP       = "|||"
combo_col = "_combo_key"

df = df.copy()
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

for c in history_attrs:
    df[c] = df[c].fillna("MISSING").astype(str)

df[combo_col] = df[history_attrs].agg(SEP.join, axis=1)
y = df[target_col].astype("int8")

d       = df[date_col].dt.floor("D")
df["_d"] = d

daily = (
    pd.DataFrame({combo_col: df[combo_col].values, "_d": d.values, "_y": y.values})
      .groupby([combo_col, "_d"], sort=False)["_y"]
      .agg(sum="sum", cnt="count")
      .reset_index()
      .sort_values([combo_col, "_d"])
)

daily["cum_sum"] = daily.groupby(combo_col, sort=False)["sum"].cumsum()
daily["cum_cnt"] = daily.groupby(combo_col, sort=False)["cnt"].cumsum()

daily["hist_sum"] = daily.groupby(combo_col, sort=False)["cum_sum"].shift(lag_days)
daily["hist_cnt"] = daily.groupby(combo_col, sort=False)["cum_cnt"].shift(lag_days)

daily["has_amendment_day"] = (daily["sum"] > 0).astype("int8")
daily["cum_day_cnt"]       = daily.groupby(combo_col, sort=False)["has_amendment_day"].cumsum()
daily["hist_day_cnt"]      = daily.groupby(combo_col, sort=False)["cum_day_cnt"].shift(lag_days)

daily = daily[[combo_col, "_d", "hist_sum", "hist_cnt", "hist_day_cnt"]]

df = df.merge(daily, on=[combo_col, "_d"], how="left")

df[f"{combo_col}_hist_sum"]      = df["hist_sum"].astype("float32")
df[f"{combo_col}_hist_cnt"]      = df["hist_cnt"].astype("float32")
df[f"{combo_col}_hist_daycounts"] = df["hist_day_cnt"].astype("float32")

cnt = df[f"{combo_col}_hist_cnt"].to_numpy(dtype="float32")
s   = df[f"{combo_col}_hist_sum"].to_numpy(dtype="float32")

pct    = np.full(len(df), np.nan, dtype="float32")
m      = ~np.isnan(cnt)
pct[m] = np.where(cnt[m] == 0, 0.0, (s[m] / cnt[m]) * 100.0).astype("float32")

df[f"{combo_col}_hist_pct"] = pct

# Counterparty consistency features derived from the structure-only key (combo_key2).
# Only unamended trades inform what the "correct" counterparty looks like.
cp_col       = "Counterparty"
cp_exclude   = {"Counterparty"}
history_attrs2 = [c for c in history_attrs if c not in cp_exclude]
combo_col2   = "_combo_key2"
df[combo_col2] = df[history_attrs2].agg(SEP.join, axis=1)

df_valid = df.loc[df[target_col] == 0, [combo_col2, "_d", cp_col]].copy()

daily_cp = (
    df_valid.groupby([combo_col2, "_d", cp_col], sort=False)
    .size()
    .reset_index(name="_cnt")
    .sort_values([combo_col2, cp_col, "_d"])
)
daily_cp["_cum_cnt"] = daily_cp.groupby([combo_col2, cp_col])["_cnt"].cumsum()

all_days = df[[combo_col2, "_d"]].drop_duplicates().sort_values([combo_col2, "_d"])

_ref    = daily_cp[[combo_col2, "_d", cp_col, "_cum_cnt"]].rename(columns={"_d": "_hist_d"})
crossed = all_days.merge(_ref, on=combo_col2)
crossed = crossed[crossed["_hist_d"] <= crossed["_d"]]

latest_cp = (
    crossed.sort_values("_hist_d")
    .groupby([combo_col2, "_d", cp_col], sort=False)["_cum_cnt"]
    .last()
    .reset_index()
)

# Full counterparty distribution string per (combo_key2, day), shifted by lag.
latest_cp["_total"] = latest_cp.groupby([combo_col2, "_d"])["_cum_cnt"].transform("sum")
latest_cp["_pct"]   = (latest_cp["_cum_cnt"] / latest_cp["_total"] * 100).round(1)

cp_dist_str = (
    latest_cp.sort_values([combo_col2, "_d", "_cum_cnt"], ascending=[True, True, False])
    .groupby([combo_col2, "_d"], sort=False)
    .apply(lambda g: " | ".join(
        f"{row[cp_col]}: {int(row['_cum_cnt'])} ({row['_pct']}%)"
        for _, row in g.iterrows()
    ))
    .reset_index(name="_cp_dist_str")
    .sort_values([combo_col2, "_d"])
)
cp_dist_str["hist_cp_dist"] = cp_dist_str.groupby(combo_col2, sort=False)["_cp_dist_str"].shift(lag_days)

df = df.merge(cp_dist_str[[combo_col2, "_d", "hist_cp_dist"]], on=[combo_col2, "_d"], how="left")
df[f"{combo_col2}_hist_cp_dist"] = df["hist_cp_dist"]
df = df.drop(columns=["hist_cp_dist"])

cp_day = (
    latest_cp.groupby([combo_col2, "_d"], sort=False)
    .agg(
        _n_unique_cp=(cp_col, "nunique"),
        _total=("_cum_cnt", "sum"),
        _max=("_cum_cnt", "max"),
    )
    .reset_index()
    .sort_values([combo_col2, "_d"])
)
cp_day["_dom_cp_pct"] = (cp_day["_max"] / cp_day["_total"] * 100).astype("float32")

cp_day["hist_n_unique_cp"] = cp_day.groupby(combo_col2, sort=False)["_n_unique_cp"].shift(lag_days)
cp_day["hist_dom_cp_pct"]  = cp_day.groupby(combo_col2, sort=False)["_dom_cp_pct"].shift(lag_days)

df = df.merge(
    cp_day[[combo_col2, "_d", "hist_n_unique_cp", "hist_dom_cp_pct"]],
    on=[combo_col2, "_d"], how="left"
)
df[f"{combo_col2}_hist_n_unique_cp"] = df["hist_n_unique_cp"].astype("float32")
df[f"{combo_col2}_hist_dom_cp_pct"]  = df["hist_dom_cp_pct"].astype("float32")
df = df.drop(columns=["hist_n_unique_cp", "hist_dom_cp_pct"])

df_ml = df.drop(columns=["_d", "hist_sum", "hist_cnt", "hist_day_cnt"]).copy()
df_ml.convert_dtypes()
df_ml.info()
df_ml.tail()
