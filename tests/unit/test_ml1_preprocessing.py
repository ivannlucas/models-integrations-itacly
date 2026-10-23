"""Pure-logic regression tests for ml1's preprocessing/training helpers.

These three cases are real bugs found by external QA against the deployed PR, reproduced
directly against the real functions (no FakePlugin — the wiring tests never exercise this code):

1. A non-numeric sensor value escaped as a bare ValueError, which predict_batch's per-row
   except (DataContractError, InvalidImageError) does not catch — it aborted the whole batch,
   discarding every already-computed valid row.
2. predict_batch picked a single images_dir (the ZIP root, or an images/ subfolder if one
   existed) instead of searching both — a photo at the root was silently ignored whenever the
   ZIP also had an images/ subfolder for other rows.
3. training.metrics_returned in the manifest promises f1_weighted, but train_tabular_experts()
   never computed or returned it.
4. A sensor value of inf (or a finite-but-huge value like 1e100, which overflows to +inf once
   cast to float32) passed validation and reached RandomForestRegressor.predict(), which raises
   a bare ValueError for non-finite input — same uncaught-abort-the-whole-batch failure as bug 1.
"""
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.domain.services.exceptions import DataContractError
from app.plugins.ml1_wine_pest_disease_detection import preprocessing, training
from app.plugins.ml1_wine_pest_disease_detection.constants import CLASSES, SENSOR_COLS

_SENSORS_OK = {
    "Mes": 7, "Temp_Max_Media_5d_C": 30.0, "Temp_Min_Media_5d_C": 18.0,
    "Hum_Rel_Max_5d_Pct": 60.0, "Hum_Rel_Min_5d_Pct": 25.0, "Hum_Suelo_Pct": 30.0,
    "Lluvia_Acum_5d_mm": 0.5, "Horas_Humedad_Foliar": 4.0, "GDD_Acumulado": 500.0,
}


def test_non_numeric_sensor_raises_data_contract_error_not_value_error():
    """A bad value must map to a domain exception predict_batch already catches per row,
    never a bare ValueError that would abort the whole batch."""
    bad_sensors = {**_SENSORS_OK, "Mes": "oops"}
    with pytest.raises(DataContractError, match="no numérico"):
        preprocessing.validate_and_align_sensors(bad_sensors)


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), 1e100])
def test_non_finite_sensor_raises_data_contract_error(bad_value):
    """inf, -inf and values that overflow to inf under float32 (1e100) must all be rejected
    as a per-row DataContractError, never reach RandomForestRegressor.predict() as non-finite
    input (sklearn's bare ValueError there is not caught by predict_batch's per-row except)."""
    with pytest.raises(DataContractError, match="no finito"):
        preprocessing.validate_and_align_sensors({**_SENSORS_OK, "Mes": bad_value})


def test_resolve_image_path_for_batch_finds_root_photo_even_with_images_subfolder(tmp_path):
    """A photo at the ZIP root must still be found when an images/ subfolder also exists for
    other rows — predict_dto.py's own docstring says both locations are searched by filename."""
    images_subdir = tmp_path / "images"
    images_subdir.mkdir()
    (images_subdir / "other_photo.jpg").write_bytes(b"not-decoded")
    (tmp_path / "root_photo.jpg").write_bytes(b"not-decoded")

    search_dirs = [images_subdir, tmp_path]
    result = preprocessing.resolve_image_path_for_batch("root_photo.jpg", search_dirs)

    assert result == tmp_path / "root_photo.jpg"


def test_resolve_image_path_for_batch_still_finds_nested_photo(tmp_path):
    images_subdir = tmp_path / "images"
    images_subdir.mkdir()
    (images_subdir / "leaf.jpg").write_bytes(b"not-decoded")

    result = preprocessing.resolve_image_path_for_batch("leaf.jpg", [images_subdir, tmp_path])

    assert result == images_subdir / "leaf.jpg"


def test_train_tabular_experts_returns_f1_weighted():
    """manifest.yaml training.metrics_returned lists f1_weighted — train() must actually
    compute and return it, not just accuracy/f1_macro/mae_riesgo."""
    rows = []
    for i, label in enumerate(CLASSES + CLASSES[:5]):  # 20 rows, every class represented
        rows.append({**_SENSORS_OK, "Mes": (i % 12) + 1, "Etiqueta": label,
                     "Indice_Riesgo_Cientifico": 0.1 + (i % 10) / 20})
    df = pd.DataFrame(rows)[[*SENSOR_COLS, "Etiqueta", "Indice_Riesgo_Cientifico"]]

    result = training.train_tabular_experts(df)

    assert isinstance(result.rf_clase, RandomForestClassifier)
    assert isinstance(result.rf_riesgo, RandomForestRegressor)
    assert 0.0 <= result.f1_weighted <= 1.0
