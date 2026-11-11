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
