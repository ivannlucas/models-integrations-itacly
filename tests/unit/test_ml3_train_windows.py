"""ml3 training cuts its 168 h windows batch by batch instead of stacking them all.

Stacking every stride-1 window (the original crear_secuencias) multiplies the scaled rows by
168: ~11 GiB for the delivered 1.6M-row dataset, which OOM-killed the pod. _Windows keeps the
rows once; these tests pin that it yields exactly the same windows and targets.
"""
# pylint: disable=protected-access  # unit tests of the private windowing helpers
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.plugins.ml3_wine_disease_pest_forecast import training

_FEATURES = ["a", "b", "c"]
_COLUMNS = ("y_c", "y_r", "serie")
_WINDOW = 5


def _reference_crear_secuencias(df, window_size, features, columns):
    """The previous implementation (port of train.crear_secuencias), verbatim."""
    t_class, t_reg, series_col = columns
    x, y_c, y_r = [], [], []
    for _, grupo in df.groupby(series_col, sort=False):
        data_features = grupo[features].values
        data_class = grupo[t_class].values
        data_reg = grupo[t_reg].values
        for i in range(len(data_features) - window_size):
            x.append(data_features[i:i + window_size])
            y_c.append(data_class[i + window_size])
            y_r.append(data_reg[i + window_size])
    return (
        np.array(x, dtype=np.float32),
        np.array(y_c, dtype=np.int8),
        np.array(y_r, dtype=np.float32),
    )


def _frame(lengths=(12, 3, 9, 6, 5)) -> pd.DataFrame:
    """Series of several lengths (some shorter than the window), mixed float32/float64."""
    rng = np.random.default_rng(7)
    parts = []
    for sid, n in zip(["s3", "s1", "s9", "s2", "s4"], lengths):
        parts.append(pd.DataFrame({
            "serie": sid,
            "a": rng.normal(size=n),
            "b": rng.normal(size=n).astype(np.float32),
            "c": rng.normal(size=n) * 1e3,
            "y_c": rng.integers(0, 4, n),
            "y_r": rng.random(n),
        }))
    return pd.concat(parts, ignore_index=True)


def test_windows_and_targets_are_bit_identical_to_crear_secuencias():
    """Every window, label and severity matches the stacked original, in the same order."""
    df = _frame()
    windows = training._crear_ventanas(df, _WINDOW, _FEATURES, _COLUMNS)
    x_ref, yc_ref, yr_ref = _reference_crear_secuencias(df, _WINDOW, _FEATURES, _COLUMNS)
    assert len(windows) == len(x_ref) == (12 - 5) + (9 - 5) + (6 - 5)
    x = windows.take(np.arange(len(windows)))
    assert x.dtype == np.float32 and x.shape == x_ref.shape
    np.testing.assert_array_equal(x, x_ref)
    assert windows.y_class.dtype == np.int8 and windows.y_reg.dtype == np.float32
    np.testing.assert_array_equal(windows.y_class, yc_ref)
    np.testing.assert_array_equal(windows.y_reg, yr_ref)


def test_frame_without_any_window_is_empty():
    """Series no longer than the window give no window (and _require_windows reports it)."""
    windows = training._crear_ventanas(_frame((3, 5, 2, 4, 1)), _WINDOW, _FEATURES, _COLUMNS)
    assert len(windows) == 0
    assert windows.take(np.arange(0)).shape == (0, _WINDOW, len(_FEATURES))


def test_memory_holds_each_row_once():
    """The split keeps rows x features floats, not windows x window x features."""
    df = _frame((400, 300, 250, 200, 150))
    windows = training._crear_ventanas(df, 168, _FEATURES, _COLUMNS)
    assert windows.features.nbytes == len(df) * len(_FEATURES) * 4
    assert windows.features.nbytes * 50 < len(windows) * 168 * len(_FEATURES) * 4


@pytest.fixture(name="batches")
def _batches():
    """The batching logic in batches of 4, without Keras (see _WindowBatchesBase)."""
    def make(windows, **kwargs):
        return training._WindowBatchesBase(windows, 4, **kwargs)
    return make


def test_shuffled_epochs_cover_every_window_once_in_a_new_order(batches):
    """Like fit(shuffle=True): each epoch is a permutation of all windows, reshuffled."""
    windows = training._crear_ventanas(_frame(), _WINDOW, _FEATURES, _COLUMNS)
    data = batches(windows, shuffle_seed=42)
    assert len(data) == 3  # 12 windows / 4, last batch full here

    def epoch():
        xs, ycs = [], []
        # Indexed, as Keras does: a PyDataset is not an iterator (no IndexError past the end).
        for i in range(len(data)):  # pylint: disable=consider-using-enumerate
            x, y = data[i]
            assert set(y) == {"out_class", "out_reg"}
            xs.append(x)
            ycs.append(y["out_class"])
        return np.concatenate(xs), np.concatenate(ycs)

    first_x, _ = epoch()
    data.on_epoch_end()
    second_x, _ = epoch()
    every = windows.take(np.arange(len(windows)))

    def as_rows(x):
        return sorted(map(bytes, x.reshape(len(x), -1)))

    assert as_rows(first_x) == as_rows(every) == as_rows(second_x)
    assert not np.array_equal(first_x, second_x)


def test_unshuffled_batches_keep_order_and_last_partial_batch(batches):
    """Validation/test batches follow the window order; the last batch may be shorter."""
    windows = training._crear_ventanas(_frame((12, 3, 9, 6, 7)), _WINDOW, _FEATURES, _COLUMNS)
    data = batches(windows)
    assert len(windows) == 14 and len(data) == 4
    xs = [data[i][0] for i in range(len(data))]
    assert [len(x) for x in xs] == [4, 4, 4, 2]
    np.testing.assert_array_equal(np.concatenate(xs), windows.take(np.arange(14)))


def test_predict_batches_yield_only_the_windows(batches):
    """targets=False is the input of model.predict: windows only."""
    windows = training._crear_ventanas(_frame(), _WINDOW, _FEATURES, _COLUMNS)
    x = batches(windows, targets=False)[0]
    assert isinstance(x, np.ndarray)
    np.testing.assert_array_equal(x, windows.take(np.arange(4)))
