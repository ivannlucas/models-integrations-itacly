"""Regression tests for the /train review fixes of ml10, ml5 and ml23.

Real plugin code, no mocks.
"""
# pylint: disable=protected-access

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


# ── ml10: reproducible split with the original tick cap, and test-split metrics ─────────────

def _flat_dataset(root, n_ticks=5, n_other=4):
    for cls, n in (("fly", n_other), ("mos", n_other), ("tick", n_ticks)):
        (root / cls).mkdir(parents=True)
        for i in range(n):
            (root / cls / f"{cls}_{i:04d}.jpg").write_bytes(b"x")


def _split_listing(splits_root):
    return sorted(str(p.relative_to(splits_root)) for p in splits_root.rglob("*.jpg"))


def test_ml10_split_is_reproducible(tmp_path):
    """ml10's split is the same on every run and keeps 70% of each class for train."""
    # Lazy: heavy plugin deps, loaded only when this test runs.
    from app.plugins.ml10_dairy_disease_vector_detection.plugin import (  # pylint: disable=import-outside-toplevel
        _create_splits,
    )
    _flat_dataset(tmp_path / "data", n_other=20)
    first = _split_listing(
        _create_splits(tmp_path / "data", ["fly", "mos", "tick"], str(tmp_path / "a"))
    )
    second = _split_listing(
        _create_splits(tmp_path / "data", ["fly", "mos", "tick"], str(tmp_path / "b"))
    )
    assert first == second
    assert sum(p.startswith("train/fly/") for p in first) == 14  # 70/15/15 of 20


def test_ml10_split_caps_ticks_like_the_original(tmp_path, monkeypatch):
    """ml10's split caps tick images at MAX_TICKS like the original."""
    # Lazy: heavy plugin deps, loaded only when this test runs.
    from app.plugins.ml10_dairy_disease_vector_detection import plugin  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(plugin, "MAX_TICKS", 3)
    _flat_dataset(tmp_path / "data", n_ticks=10)
    listing = _split_listing(
        plugin._create_splits(tmp_path / "data", ["fly", "mos", "tick"], str(tmp_path / "s"))
    )
    assert sum("/tick/" in p for p in listing) == 3
    assert sum("/fly/" in p for p in listing) == 4


def test_ml10_test_metrics_follow_the_original_predictor_formulas():
    """ml10's test metrics match the original accuracy/precision/recall/f1 formulas."""
    # Lazy: heavy plugin deps, loaded only when this test runs.
    from app.plugins.ml10_dairy_disease_vector_detection.plugin import (  # pylint: disable=import-outside-toplevel
        _cls_test_metrics,
    )

    class Fixed(torch.nn.Module):
        """Predicts the class encoded in the first feature."""

        def forward(self, x):
            """Return the one-hot of the first feature as logits."""
            return torch.nn.functional.one_hot(  # pylint: disable=not-callable  # torch C function
                x[:, 0].long(), 3
            ).float()

    x = torch.tensor([[0.0], [0.0], [1.0], [2.0]])
    y = torch.tensor([0, 1, 1, 2])  # one "mos" predicted as "fly"
    m = _cls_test_metrics(
        Fixed(), DataLoader(TensorDataset(x, y), batch_size=2), ["fly", "mos", "tick"], "cpu"
    )
    assert m["test_accuracy"] == 0.75
    assert m["test_per_class"]["fly"] == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": 0.6667,
        "support": 1,
    }
    assert m["test_per_class"]["mos"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
        "support": 2,
    }


# ── ml5: clear errors instead of sklearn's or a silent val_acc=0 ────────────────────────────

def _images(n_clips):
    return {"images": [{"id": i, "clip_name": f"clip{i}"} for i in range(n_clips)]}


def test_ml5_too_few_clips_is_a_clear_error():
    """ml5 rejects fewer than 4 clips with a clear error and splits 4 as 2/1/1."""
    # Lazy: heavy plugin deps, loaded only when this test runs.
    from app.plugins.ml5_meat_cow_behaviour.training import split_clips  # pylint: disable=import-outside-toplevel
    with pytest.raises(ValueError, match="al menos 4"):
        split_clips(_images(3))
    assert {k: len(v) for k, v in split_clips(_images(4)).items()} == {
        "train": 2,
        "val": 1,
        "test": 1,
    }


def test_ml5_training_without_validation_clips_is_rejected():
    """ml5 training without validation clips is rejected."""
    # Lazy: heavy plugin deps, loaded only when this test runs.
    from app.plugins.ml5_meat_cow_behaviour.training import train_classifier  # pylint: disable=import-outside-toplevel
    with pytest.raises(ValueError, match="val=0"):
        train_classifier(
            [{"behavior": "feeding"}], [], "/nonexistent", clip_length=32, device="cpu"
        )


# ── ml23: empty or too-short dataset ────────────────────────────────────────────────────────

def test_ml23_dataset_without_rows_beyond_the_horizon_is_a_clear_error():
    """ml23 rejects empty or too-short series with a clear 6-month error."""
    # Lazy: heavy plugin deps, loaded only when this test runs.
    from app.plugins.ml23_lactic_market_price_forecast.training import (  # pylint: disable=import-outside-toplevel
        prepare_horizon_dataset,
    )
    empty = pd.DataFrame(columns=["producto", "canal", "fecha", "target_precio_medio"])
    with pytest.raises(ValueError, match="6 meses"):
        prepare_horizon_dataset(empty, horizon=6)
    short = pd.DataFrame(
        {
            "producto": "leche",
            "canal": "x",
            "fecha": pd.date_range("2020-01", periods=4, freq="MS"),
            "target_precio_medio": np.arange(4.0),
        }
    )
    with pytest.raises(ValueError, match="6 meses"):
        prepare_horizon_dataset(short, horizon=6)
