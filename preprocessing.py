# Data preprocessing and filtering applied to the raw ingested dataset.
#
# Three filters are applied before modelling:
#   - Start date: only trades on or after 2025-01-01
#   - Recency cutoff: the most recent 14 days are dropped because most counterparty
#     amendments occur within two weeks of booking; trades inside that window do not
#     yet have settled labels and would appear as false negatives
#   - Internal trades: rows where IsInternal == 1 are removed, as internal trades
#     have no external counterparty to amend
#
# Attribute selection narrows the dataset to the eight dimensions used to define a
# trade's identity group (combo key), plus metadata and identifier columns.
# Edit history_attrs here to add or remove features — the combo key in
# feature_engineering.py is built directly from this list.

import pandas as pd

df = df_global.copy()

df["TradeEntryDate"] = pd.to_datetime(df["TradeEntryDate"], errors="coerce")

start_date = pd.Timestamp("2025-01-01")

cutoff = df["TradeEntryDate"].max() - pd.Timedelta(days=14)

mask = (df["TradeEntryDate"] >= start_date) & (df["TradeEntryDate"] < cutoff)
df   = df.loc[mask]
df   = df[df["IsInternal"] != 1].copy()

# Eight categorical dimensions that define a unique trade profile.
# Any column outside this list and the reserved metadata columns is dropped.
history_attrs = [
    'Counterparty', 'Fund', 'ExecutionVenue', 'TradeCategory',
    'ClearingHouse', 'CounterpartyRisk', 'InstrumentCcy', 'TradeType',
]

drop_cols = [c for c in df.columns if c not in history_attrs and c not in [
    'TradeEntryDate', 'IsCounterpartyChanged', 'AmendedOnDay',
    'NBInternal', 'NBExternal', 'ChangedToCounterparty', 'Version', 'Business',
]]
df = df.drop(columns=drop_cols)
df.info()
