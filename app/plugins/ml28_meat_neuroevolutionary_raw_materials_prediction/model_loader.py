"""'Artifact' loading for ml28.

This plugin has no trained model artifact (.pkl/.pt) — the decision engine it serves
(src/cli/platform_run.py in the original delivery) is a deterministic rules engine over
config/platform_config.yaml constants, not a fitted model. See inbox/a28/manifest.yaml
known_issues for why (the delivery's own trained artifacts — upstream_predictor/purchase_trigger/
quantity_optimizer .pkl — belong to a separate, disconnected `mixed_context` ML pipeline that
platform_run never loads).

load_platform_config() exists only so this plugin follows the same load()/is_loaded() shape as
every other plugin in this repo — there is nothing to download from S3 or MLflow.
"""
from __future__ import annotations

from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction.constants import (
    BASELINE_SAFETY_STOCK_FACTOR,
    MAX_STOCKOUT_INCREASE_PCT,
    PURCHASE_TRIGGER_GAP_SIGMOID_SCALE,
    ROUNDING_DECIMALS,
)


def load_platform_config() -> dict:
    """Return the platform config dict shape expected by the vendored _vendor/ functions."""
    return {
        "platform": {
            "rounding_decimals": ROUNDING_DECIMALS,
            "heuristics": {"purchase_trigger_gap_sigmoid_scale": PURCHASE_TRIGGER_GAP_SIGMOID_SCALE},
            "simulation": {"baseline_safety_stock_factor": BASELINE_SAFETY_STOCK_FACTOR},
            "guardrails": {"max_stockout_increase_pct": MAX_STOCKOUT_INCREASE_PCT},
        },
    }
