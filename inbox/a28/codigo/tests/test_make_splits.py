from __future__ import annotations

import pandas as pd

from src.utils import read_json
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_make_splits_outputs_are_chronological() -> None:
    ensure_repro_smoke_pipeline()
    split_dir = REPO_ROOT / "data/splits/baseline/default__mixed_context"
    train = pd.read_csv(split_dir / "train.csv")
    validation = pd.read_csv(split_dir / "validation.csv")
    test = pd.read_csv(split_dir / "test.csv")
    metadata = read_json(split_dir / "split_metadata.json")

    train_dates = pd.to_datetime(train["date"])
    validation_dates = pd.to_datetime(validation["date"])
    test_dates = pd.to_datetime(test["date"])

    assert metadata["criterion"] == "chronological"
    assert train_dates.max() < validation_dates.min()
    assert validation_dates.max() < test_dates.min()
    assert metadata["splits"]["train"]["rows"] == len(train)
    assert metadata["splits"]["validation"]["rows"] == len(validation)
    assert metadata["splits"]["test"]["rows"] == len(test)
