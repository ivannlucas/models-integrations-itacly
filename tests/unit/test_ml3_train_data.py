"""ml3 /train input loading: CSV or parquet by content, whatever the file name."""
import pandas as pd
import pytest

from app.plugins.ml3_wine_disease_pest_forecast.constants import TRAIN_HARD_REQUIRED_COLUMNS
from app.plugins.ml3_wine_disease_pest_forecast.training import load_retraining_input


def _frame() -> pd.DataFrame:
    row = {c: 0.0 for c in TRAIN_HARD_REQUIRED_COLUMNS}
    row.update({"Fecha": "2022-05-31 10:00:00", "ID_Serie": 1, "Clase_Entrenamiento": "HEALTHY"})
    return pd.DataFrame([row, row])


@pytest.mark.parametrize("name", ["train.csv", "upload-0f3a"])
def test_csv_with_or_without_extension(tmp_path, name):
    """An s3:// key may have no ".csv" suffix: a CSV is still read as CSV."""
    path = tmp_path / name
    _frame().to_csv(path, index=False)
    assert len(load_retraining_input(str(path))) == 2


@pytest.mark.parametrize("name", ["train.parquet", "upload-0f3a"])
def test_parquet_with_or_without_extension(tmp_path, name):
    """Parquet is detected by its magic bytes."""
    path = tmp_path / name
    _frame().to_parquet(path, index=False)
    assert len(load_retraining_input(str(path))) == 2


def test_missing_columns_is_a_clear_error(tmp_path):
    """The contract error names the missing columns."""
    path = tmp_path / "train.csv"
    _frame().drop(columns=["Grado_Infeccion"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Grado_Infeccion"):
        load_retraining_input(str(path))


# --- Feasibility of the 70/15/15 stratified series split (QA: raw sklearn errors) -----------

from app.domain.services.exceptions import InsufficientDataError  # noqa: E402
from app.plugins.ml3_wine_disease_pest_forecast import training  # noqa: E402
from app.plugins.ml3_wine_disease_pest_forecast.constants import (  # noqa: E402
    MIN_SERIES_PER_STRATUM,
    WINDOW_SIZE,
)

_CLASSES = ["ALTICA", "BLACK_ROT", "BOTRYTIS", "EMPOASCA", "ERINOSIS", "ESCA",
            "HEALTHY", "LOBESIA", "MILDIU", "OIDIO", "RED_MITE"]


def _series_frame(series_per_class: dict[str, int], rows_per_series: int = 3) -> pd.DataFrame:
    """One row block per series; Etiqueta_Clase constant per series, HEALTHY rows first."""
    rows, sid = [], 0
    for label, n_series in series_per_class.items():
        for _ in range(n_series):
            sid += 1
            for r in range(rows_per_series):
                rows.append({"ID_Serie": sid, "Etiqueta_Clase": label,
                             "Clase_Entrenamiento": "HEALTHY" if r == 0 else label})
    return pd.DataFrame(rows)


def test_missing_etiqueta_clase_is_a_clear_error(tmp_path):
    """Etiqueta_Clase is required: without it the original split fails with a KeyError."""
    path = tmp_path / "train.csv"
    _frame().drop(columns=["Etiqueta_Clase"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Etiqueta_Clase"):
        load_retraining_input(str(path))


@pytest.mark.parametrize("n_classes", [2, 5, 11])
@pytest.mark.parametrize("extra", [0, 1, 3, 8])
def test_minimum_series_per_class_always_splits(n_classes, extra):
    """With >= MIN_SERIES_PER_STRATUM series per class the original split never fails."""
    counts = {c: MIN_SERIES_PER_STRATUM + (extra if i % 2 else 0)
              for i, c in enumerate(_CLASSES[:n_classes])}
    _, frames, n_ids = training._encode_and_split(_series_frame(counts))
    assert all(len(f) for f in frames)
    assert sum(n_ids) == sum(counts.values())


@pytest.mark.parametrize("series", [1, 2, 5, MIN_SERIES_PER_STRATUM - 1])
def test_too_few_series_is_a_clear_error(series):
    """The QA case (2 series -> 'n_samples=1, test_size=0.5') now names the rule and the class."""
    counts = {"MILDIU": MIN_SERIES_PER_STRATUM, "OIDIO": series}
    with pytest.raises(InsufficientDataError, match=rf"al menos {MIN_SERIES_PER_STRATUM} series.*OIDIO: {series}"):
        training._encode_and_split(_series_frame(counts))


def test_etiqueta_clase_must_be_constant_per_series():
    """Etiqueta_Clase labels the whole series; a series with two values is rejected."""
    df = _series_frame({"MILDIU": MIN_SERIES_PER_STRATUM, "OIDIO": MIN_SERIES_PER_STRATUM})
    df.loc[df.index[-1], "Etiqueta_Clase"] = "MILDIU"
    with pytest.raises(InsufficientDataError, match="constante"):
        training._encode_and_split(df)


def test_a_single_training_class_is_a_clear_error():
    """The classifier needs at least two Clase_Entrenamiento values."""
    df = _series_frame({"MILDIU": MIN_SERIES_PER_STRATUM, "OIDIO": MIN_SERIES_PER_STRATUM})
    df["Clase_Entrenamiento"] = "HEALTHY"
    with pytest.raises(InsufficientDataError, match="2 clases"):
        training._encode_and_split(df)


def test_split_without_windows_is_a_clear_error():
    """Series of <= WINDOW_SIZE rows produce no window: say so instead of training on nothing."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    empty = (np.zeros((0, WINDOW_SIZE, 3)), np.zeros(0), np.zeros(0))
    full = (np.zeros((4, WINDOW_SIZE, 3)), np.zeros(4), np.zeros(4))
    with pytest.raises(InsufficientDataError, match=rf"validación.*{WINDOW_SIZE + 1} filas"):
        training._require_windows({"entrenamiento": full, "validación": empty, "test": full})
    training._require_windows({"entrenamiento": full, "validación": full, "test": full})
