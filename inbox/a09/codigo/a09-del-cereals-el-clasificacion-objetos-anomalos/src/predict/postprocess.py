from __future__ import annotations

import pandas as pd


def format_output(df_pred: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["sample_id", "timestamp", "timestamp_end", "window_index"] if c in df_pred.columns]
    other = [c for c in df_pred.columns if c not in cols]
    out = df_pred[cols + other].copy()
    sort_cols = [c for c in ["sample_id", "timestamp", "timestamp_end", "window_index"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out
