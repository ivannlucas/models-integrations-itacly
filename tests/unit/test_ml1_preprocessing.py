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
    """A photo only present in a nested images/ folder is still found."""
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


def test_train_tabular_experts_rejects_csv_missing_a_class():
    """manifest constraints_entrenamiento: the training CSV must bring all 15 classes.
    Without one, train() fails with a ValueError (→ 400) naming the missing class instead of
    training a classifier that silently never predicts it."""
    rows = []
    for i, label in enumerate(CLASSES[:-1] * 2):  # 28 rows, RED_MITE never appears
        rows.append({**_SENSORS_OK, "Mes": (i % 12) + 1, "Etiqueta": label,
                     "Indice_Riesgo_Cientifico": 0.1 + (i % 10) / 20})
    df = pd.DataFrame(rows)[[*SENSOR_COLS, "Etiqueta", "Indice_Riesgo_Cientifico"]]

    with pytest.raises(ValueError, match="RED_MITE"):
        training.train_tabular_experts(df)


def test_training_child_trains_without_loading_the_bundle(tmp_path, monkeypatch):
    """The async training child calls load_for_training(), not load(): the bundle (YOLO, the 3
    CNN and the ~1 GiB RandomForest risk regressor) is already held by the serving process, and
    loading it again in the child got the pod OOMKilled at 2Gi. train() uses none of it."""
    # pylint: disable-next=import-outside-toplevel  # heavy (keras/torch/ultralytics)
    from app.plugins.ml1_wine_pest_disease_detection import plugin as ml1

    def _no_bundle():
        raise AssertionError("the training child must not load the bundle")

    monkeypatch.setattr(ml1.model_loader, "load_model_bundle", _no_bundle)
    monkeypatch.setattr(ml1, "require_train_tracker", lambda run_id: object())
    monkeypatch.setattr(ml1, "log_run_metadata", lambda tracker, metrics: None)
    written = {}

    def fake_upload(tracker, write, prefix):
        del tracker, prefix
        write(str(tmp_path))
        written["files"] = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".joblib")

    monkeypatch.setattr(ml1, "upload_model_to_mlflow", fake_upload)
    rows = []
    for i, label in enumerate(CLASSES + CLASSES[:5]):  # 20 rows, every class represented
        rows.append({**_SENSORS_OK, "Mes": (i % 12) + 1, "Etiqueta": label,
                     "Indice_Riesgo_Cientifico": 0.1 + (i % 10) / 20})
    csv = tmp_path / "train.csv"
    pd.DataFrame(rows)[[*SENSOR_COLS, "Etiqueta", "Indice_Riesgo_Cientifico"]].to_csv(
        csv, index=False)

    plugin = ml1.Ml1WinePestDiseaseDetectionPlugin()
    plugin.load_for_training(data_path=str(csv), mlflow_run_id="run-1")
    assert not plugin.is_loaded()
    resp = plugin.train(data_path=str(csv), mlflow_run_id="run-1")

    assert resp.n_train + resp.n_test == len(rows)
    assert written["files"] == [ml1.RF_RIESGO_MLFLOW_FILENAME]
