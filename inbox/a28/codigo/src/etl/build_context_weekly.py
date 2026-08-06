from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.reproducibility.hashes import describe_existing_files
from src.reproducibility.runtime import official_paths
from src.utils import ensure_directory, write_json


def _to_base100(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    base = float(valid.iloc[0])
    if base == 0:
        return numeric
    return numeric / base * 100.0


def build_context_weekly(config: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    paths = official_paths(config)
    interim_dir = ensure_directory(repo_root / "data" / "interim" / "external")

    demand_monthly = pd.read_csv(interim_dir / "demand_index_monthly__mixed_context.csv")
    supply_monthly = pd.read_csv(interim_dir / "supply_index_monthly__mixed_context.csv")
    purchase_price_weekly = pd.read_csv(interim_dir / "purchase_price_weekly__mixed_context.csv")

    demand_monthly["date"] = pd.to_datetime(demand_monthly["date"], errors="coerce")
    supply_monthly["date"] = pd.to_datetime(supply_monthly["date"], errors="coerce")
    purchase_price_weekly["date"] = pd.to_datetime(purchase_price_weekly["date"], errors="coerce")

    start_month = min(demand_monthly["date"].min(), supply_monthly["date"].min())
    end_month = max(demand_monthly["date"].max(), supply_monthly["date"].max())
    weekly_start = start_month - pd.to_timedelta(start_month.weekday(), unit="D")
    weekly_end = purchase_price_weekly["date"].max()
    weekly_index = pd.date_range(start=weekly_start, end=weekly_end, freq="W-MON")

    demand_series = (
        demand_monthly.groupby("date", as_index=True)["value"].mean().sort_index().reindex(
            pd.date_range(start=start_month, end=end_month, freq="MS")
        ).ffill()
    )
    supply_series = (
        supply_monthly.groupby("date", as_index=True)["value"].mean().sort_index().reindex(
            pd.date_range(start=start_month, end=end_month, freq="MS")
        ).ffill()
    )
    demand_weekly = demand_series.reindex(demand_series.index.union(weekly_index)).sort_index().ffill().reindex(weekly_index)
    supply_weekly = supply_series.reindex(supply_series.index.union(weekly_index)).sort_index().ffill().reindex(weekly_index)

    context_weekly = pd.DataFrame(
        {
            "date": weekly_index,
            "supply_index": _to_base100(supply_weekly),
            "purchase_price_index": _to_base100(purchase_price_weekly.set_index("date")["value"].sort_index().reindex(weekly_index).ffill()),
            "demand_index": _to_base100(demand_weekly),
        }
    )
    context_weekly.to_csv(paths["context_weekly"], index=False)

    supply_valid_dates = supply_monthly.loc[
        pd.to_numeric(supply_monthly["value"], errors="coerce").notna(),
        "date",
    ]
    first_supply_week = context_weekly.loc[context_weekly["supply_index"].notna(), "date"].min()
    unique_purchase_price_values = (
        pd.to_numeric(context_weekly["purchase_price_index"], errors="coerce").dropna().round(6).unique().tolist()
    )
    limitations = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context_weekly_path": "data/processed/external/context/context_weekly_for_simulation.csv",
        "external_long_path": "data/processed/external/context/external_long.csv",
        "active_sources": [
            {"source": "INE", "dataset": "CPI"},
            {"source": "MAPA", "dataset": "SLAUGHTER_MAPA"},
        ],
        "source_coverage": {
            "MAPA_SLAUGHTER_MAPA": {
                "date_min": str(supply_valid_dates.min().date()) if not supply_valid_dates.empty else None,
                "date_max": str(supply_valid_dates.max().date()) if not supply_valid_dates.empty else None,
            }
        },
        "traced_sources": [
            {"source": "MAPA", "dataset": "PRICES_OM", "status": "fallback_constant"}
        ],
        "candidate_sources": [
            {"source": "DATACOMEX", "dataset": "TRADE_PRESSURE"},
            {"source": "EUROSTAT", "dataset": "SLAUGHTER"},
            {"source": "MAPA", "dataset": "CONSUMPTION_PANEL"},
        ],
        "limitations": [
            {
                "issue": "purchase_price_index_is_placeholder_constant",
                "status": "traced_not_active",
                "affected_artifact": "data/processed/external/context/context_weekly_for_simulation.csv",
                "evidence": {
                    "unique_values": len(unique_purchase_price_values),
                    "prices_om_subseries": ["fallback_constant"] if len(unique_purchase_price_values) == 1 else ["weekly_series"],
                },
                "technical_comment": "MAPA_PRICES_OM remains traced only. The defended route does not claim an active weekly price feed unless a reproducible weekly series is available.",
            },
            {
                "issue": "external_sources_are_contextual_proxies",
                "status": "documented_scope_boundary",
                "affected_artifact": "data/processed/external/context/context_weekly_for_simulation.csv",
                "technical_comment": "External context variables are real/proxy signals and do not represent internal plant inventory, orders or receipts.",
            },
            {
                "issue": "supply_index_absent_before_mapa_slaughter_coverage",
                "status": "no_backward_fill_applied",
                "affected_artifact": "data/processed/external/context/context_weekly_for_simulation.csv",
                "evidence": {
                    "first_valid_supply_week": str(first_supply_week.date()) if pd.notna(first_supply_week) else None,
                    "pre_2021_missing_rows": int(
                        context_weekly.loc[context_weekly["date"] < pd.Timestamp("2021-01-01"), "supply_index"].isna().sum()
                    ),
                },
                "technical_comment": "MAPA_SLAUGHTER_MAPA starts in 2021 in the local snapshot. supply_index remains NaN before the first observed value; modeling and inference fill NaNs later with train-fitted statistics and an explicit neutral fallback for all-missing train columns.",
            },
        ],
    }
    write_json(paths["context_limitations"], limitations)

    metadata_path = interim_dir / "build_context_weekly__mixed_context.json"
    metadata = {
        "scope": "mixed_context",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": describe_existing_files(
            [
                paths["external_long"],
                interim_dir / "demand_index_monthly__mixed_context.csv",
                interim_dir / "supply_index_monthly__mixed_context.csv",
                interim_dir / "purchase_price_weekly__mixed_context.csv",
            ],
            repo_root=repo_root,
        ),
        "outputs": describe_existing_files([paths["context_weekly"], paths["context_limitations"]], repo_root=repo_root),
        "row_count": int(len(context_weekly)),
        "columns": context_weekly.columns.tolist(),
        "date_min": str(context_weekly["date"].min().date()) if not context_weekly.empty else None,
        "date_max": str(context_weekly["date"].max().date()) if not context_weekly.empty else None,
        "first_valid_supply_week": str(first_supply_week.date()) if pd.notna(first_supply_week) else None,
    }
    write_json(metadata_path, metadata)
    return {
        "context_weekly_path": str(paths["context_weekly"]),
        "limitations_path": str(paths["context_limitations"]),
        "metadata_path": str(metadata_path),
    }
