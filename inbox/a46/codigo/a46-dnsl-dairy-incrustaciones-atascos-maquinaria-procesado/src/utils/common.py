from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.utils.paths import relativize_payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_columns(df: pd.DataFrame, defaults: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


def derive_cycle_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "asset_id" not in out.columns:
        out["asset_id"] = "asset_000"
    out["asset_id"] = out["asset_id"].astype(str)

    if "cycle_id" not in out.columns or out["cycle_id"].isna().all():
        candidate = None
        if "batch_id" in out.columns and (~out["batch_id"].astype(str).isin(["", "none", "nan", "None"])).any():
            candidate = out["batch_id"].astype(str)
        elif "episode_id" in out.columns and (~out["episode_id"].astype(str).isin(["", "none", "nan", "None"])).any():
            candidate = out["episode_id"].astype(str)
        if candidate is not None:
            out["cycle_id"] = candidate
        else:
            out["cycle_id"] = ""
            for asset_id, idx in out.groupby("asset_id", sort=False).groups.items():
                g = out.loc[idx].sort_values("timestamp")
                ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
                gap = ts.diff().dt.total_seconds().fillna(0.0)
                cycle_ord = (gap > 6 * 3600).cumsum() + 1
                out.loc[g.index, "cycle_id"] = [f"{asset_id}_C{int(v):05d}" for v in cycle_ord]
    out["cycle_id"] = out["cycle_id"].astype(str).replace({"nan": "none", "None": "none"})

    if "sequence_id" not in out.columns or out["sequence_id"].isna().all():
        out["sequence_id"] = out["asset_id"].astype(str) + "::" + out["cycle_id"].astype(str)
    else:
        out["sequence_id"] = out["sequence_id"].astype(str)
        pair_count = out[["asset_id", "cycle_id"]].drop_duplicates().shape[0]
        if out["sequence_id"].nunique() < pair_count:
            out["sequence_id"] = out["asset_id"].astype(str) + "::" + out["cycle_id"].astype(str)

    if "cycle_index" not in out.columns or pd.to_numeric(out["cycle_index"], errors="coerce").isna().all():
        out["cycle_index"] = 0
        for asset_id, idx in out.groupby("asset_id", sort=False).groups.items():
            g = out.loc[idx].sort_values("timestamp")
            seen: Dict[str, int] = {}
            nxt = 1
            vals = []
            for cycle_id in g["cycle_id"].astype(str).tolist():
                if cycle_id not in seen:
                    seen[cycle_id] = nxt
                    nxt += 1
                vals.append(seen[cycle_id])
            out.loc[g.index, "cycle_index"] = vals
    out["cycle_index"] = pd.to_numeric(out["cycle_index"], errors="coerce").fillna(0).astype(int)
    return out


def sequence_group_columns(df: pd.DataFrame) -> list[str]:
    if "sequence_id" in df.columns:
        return ["asset_id", "sequence_id"]
    if "cycle_id" in df.columns:
        return ["asset_id", "cycle_id"]
    return ["asset_id"]


def minutes_to_next_event(timestamps_ns: np.ndarray, event_times_ns: np.ndarray) -> np.ndarray:
    out = np.full(len(timestamps_ns), np.nan, dtype=np.float32)
    if len(event_times_ns) == 0:
        return out
    idx = np.searchsorted(event_times_ns, timestamps_ns, side="left")
    valid = idx < len(event_times_ns)
    out[valid] = (event_times_ns[idx[valid]] - timestamps_ns[valid]).astype(np.float64) / 60e9
    return out.astype(np.float32)


def minutes_since_last_event(timestamps_ns: np.ndarray, event_times_ns: np.ndarray) -> np.ndarray:
    out = np.full(len(timestamps_ns), np.nan, dtype=np.float32)
    if len(event_times_ns) == 0:
        return out
    idx = np.searchsorted(event_times_ns, timestamps_ns, side="right") - 1
    valid = idx >= 0
    out[valid] = (timestamps_ns[valid] - event_times_ns[idx[valid]]).astype(np.float64) / 60e9
    return out.astype(np.float32)


def previous_event_value(timestamps_ns: np.ndarray, event_times_ns: np.ndarray, values: Sequence[str], default: str = "none") -> np.ndarray:
    out = np.array([default] * len(timestamps_ns), dtype=object)
    if len(event_times_ns) == 0:
        return out
    idx = np.searchsorted(event_times_ns, timestamps_ns, side="right") - 1
    valid = idx >= 0
    if np.any(valid):
        vals = np.asarray(list(values), dtype=object)
        out[valid] = vals[idx[valid]]
    return out


def next_event_value(timestamps_ns: np.ndarray, event_times_ns: np.ndarray, values: Sequence[str], default: str = "none") -> np.ndarray:
    out = np.array([default] * len(timestamps_ns), dtype=object)
    if len(event_times_ns) == 0:
        return out
    idx = np.searchsorted(event_times_ns, timestamps_ns, side="left")
    valid = idx < len(event_times_ns)
    if np.any(valid):
        vals = np.asarray(list(values), dtype=object)
        out[valid] = vals[idx[valid]]
    return out


def normalize_fault_type(x: str) -> str:
    s = str(x).strip().lower()
    if "clog" in s:
        return "clogging"
    if "foul" in s or "cip_extra" in s or "mechanical" in s:
        return "fouling"
    if "prevent" in s:
        return "preventive"
    return "other"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return default
        return float(x)
    except Exception:
        return default


def nan_to_zero(x: float) -> float:
    return 0.0 if x is None or not math.isfinite(float(x)) else float(x)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _conv(x: Any) -> Any:
        if isinstance(x, (np.integer, np.int64, np.int32)):
            return int(x)
        if isinstance(x, (np.floating, np.float32, np.float64)):
            if math.isnan(float(x)) or math.isinf(float(x)):
                return None
            return float(x)
        if isinstance(x, (pd.Timestamp,)):
            return x.isoformat()
        return x

    normalized_payload = relativize_payload(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(normalized_payload, f, indent=2, default=_conv)
