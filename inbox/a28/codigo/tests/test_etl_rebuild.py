from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_etl_rebuild_outputs_exist() -> None:
    ensure_repro_smoke_pipeline()

    external_long = REPO_ROOT / "data/processed/external/context/external_long.csv"
    context_weekly = REPO_ROOT / "data/processed/external/context/context_weekly_for_simulation.csv"
    synthetic_layer = REPO_ROOT / "data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv"
    modeling = REPO_ROOT / "data/processed/baseline/feature_engineering_modeling__mixed_context.csv"

    for path in [external_long, context_weekly, synthetic_layer, modeling]:
        assert path.exists(), path
        assert path.stat().st_size > 0

    assert not pd.read_csv(external_long).empty
    assert not pd.read_csv(context_weekly).empty
    assert not pd.read_csv(synthetic_layer).empty
    assert not pd.read_csv(modeling).empty
