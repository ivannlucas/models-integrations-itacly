"""ml41 training loads the clips without extra copies, with bit-identical results.

The previous code kept the list of spectrograms, the stacked array, the 80/20 split and the
normalized copies alive at once (about 4 copies of the dataset, 512 KiB per clip); with a full
MIMII combination (~1,000 normal clips) that alone outgrew a 2 GiB pod.
"""
# pylint: disable=protected-access  # unit tests of the private loading helpers
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly import plugin as ml41
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.constants import (
    TRAINING_HYPERPARAMS_BY_MACHINE,
    TRAINING_SPLIT_RATIO,
    TRAINING_SPLIT_SEED,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.preprocessing import (
    normalize_logmel,
)

_SHAPE = (1, 8, 16)


def _fake_logmel(path: str) -> np.ndarray:
    """Deterministic float32 'spectrogram' per file, in the log-mel dB range."""
    seed = int(Path(path).stem)
    return (np.random.default_rng(seed).normal(-40, 15, _SHAPE)).astype(np.float32)


def _reference_split(wavs: list[Path]) -> tuple:
    """The previous implementation of _split_normal_clips, verbatim."""
    specs = np.stack([_fake_logmel(str(p)) for p in wavs], axis=0)
    # pylint: disable-next=no-member  # legacy RandomState, as the original
    idx = np.random.RandomState(TRAINING_SPLIT_SEED).permutation(len(specs))
    cut = int(TRAINING_SPLIT_RATIO * len(specs))
    train_raw, val_raw = specs[idx[:cut]], specs[idx[cut:]]
    norm_mean, norm_std = float(train_raw.mean()), float(train_raw.std())
    return (
        normalize_logmel(train_raw, norm_mean, norm_std),
        normalize_logmel(val_raw, norm_mean, norm_std),
        (norm_mean, norm_std),
    )


@pytest.fixture(name="combo")
def _combo(tmp_path, monkeypatch):
    monkeypatch.setattr(ml41, "wav_to_logmel", _fake_logmel)
    normal = tmp_path / "-6_dB_fan" / "id_00" / "normal"
    normal.mkdir(parents=True)
    for i in range(53):
        (normal / f"{i:08d}.wav").write_bytes(b"")
    return ("fan", "id_00", "-6_dB", normal.parent)


def test_split_is_bit_identical_to_the_previous_implementation(combo):
    """Same clips, same order, same normalization stats and values, all float32."""
    train, val, norm = ml41._split_normal_clips(combo, TRAINING_HYPERPARAMS_BY_MACHINE["fan"])
    ref_train, ref_val, ref_norm = _reference_split(sorted((combo[3] / "normal").glob("*.wav")))
    assert norm == ref_norm
    assert train.dtype == ref_train.dtype == np.float32
    assert val.dtype == ref_val.dtype == np.float32
    np.testing.assert_array_equal(train, ref_train)
    np.testing.assert_array_equal(val, ref_val)


def test_load_logmels_matches_np_stack(combo):
    """The preallocated loader returns what np.stack of each clip returned."""
    wavs = sorted((combo[3] / "normal").glob("*.wav"))
    order = [5, 0, 52, 7]
    got = ml41._load_logmels([wavs[i] for i in order])
    np.testing.assert_array_equal(got, np.stack([_fake_logmel(str(wavs[i])) for i in order]))


def test_normalize_in_place_matches_normalize_logmel():
    """In-place normalization gives the same float32 values as normalize_logmel."""
    data = np.random.default_rng(0).normal(-40, 15, (20, *_SHAPE)).astype(np.float32)
    expected = normalize_logmel(data, -38.123456789, 14.987654321)
    got = ml41._normalize_in_place(data.copy(), -38.123456789, 14.987654321)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)
