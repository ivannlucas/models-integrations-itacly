"""Regression tests for ml1's model loading and batch contract, found in the a01 review against
the AI team's full data folder:

1. EfficientNetB0 was rebuilt with weights=None, which in Keras 3 drops the Rescaling layer that
   weights='imagenet' (the original's build) inserts after the stem Normalization. The layer has
   no variables, so load_weights() never complained — but the backbone collapsed to predicting
   one class for every image and the ensemble no longer matched the audited 0.9469.
2. Both RF experts were mandatory at load time, although the decoupled mode never uses the RF
   classifier (~370 MB of RSS) and the original serves image requests without either RF.
3. predict_batch treated every non-Path_Imagen column as a sensor, so a CSV with an ID or label
   column failed every row — the original only reads the sensor columns.
"""
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import pandas as pd
import pytest

# keras is only reached through model_loader, which sets KERAS_BACKEND=torch before importing it
# (production config). Importing keras here first would pin the backend for the whole session.
from app.domain.services.exceptions import DataContractError
from app.plugins.ml1_wine_pest_disease_detection import inference, model_loader, plugin, preprocessing
from app.plugins.ml1_wine_pest_disease_detection.constants import (
    ENSEMBLE_WEIGHTS_FILENAMES,
    IMAGENET_STDDEV_RGB,
    SENSOR_COLS,
)
from keras.src.utils.naming import to_snake_case  # after model_loader: backend already set  # noqa: I100

_DELIVERED_ARTIFACTS = Path(__file__).resolve().parents[2] / "inbox" / "a01" / "codigo" / "models" / "artifacts"
_SENSORS_OK = {
    "Mes": 7, "Temp_Max_Media_5d_C": 30.0, "Temp_Min_Media_5d_C": 18.0,
    "Hum_Rel_Max_5d_Pct": 60.0, "Hum_Rel_Min_5d_Pct": 25.0, "Hum_Suelo_Pct": 30.0,
    "Lluvia_Acum_5d_mm": 0.5, "Horas_Humedad_Foliar": 4.0, "GDD_Acumulado": 500.0,
}


def _backbone(backbone_type: str):
    hybrid = model_loader.build_hybrid_model(backbone_type)
    return next(layer for layer in hybrid.layers if isinstance(layer, model_loader.models.Model))


def test_all_backbones_build_in_load_order_from_a_fresh_session():
    """load_model_bundle builds the backbones one after another in one process: a fixed layer
    name in one of them must not collide with Keras' auto-numbered names from the previous one."""
    model_loader.keras.backend.clear_session()
    for backbone_type in ENSEMBLE_WEIGHTS_FILENAMES:
        model_loader.build_hybrid_model(backbone_type)


def test_efficientnet_has_imagenet_rescaling_after_normalization():
    backbone = _backbone("efficientnet")
    names = [type(layer).__name__ for layer in backbone.layers]
    norm_idx = names.index("Normalization")
    assert names[norm_idx - 1] == "Rescaling"   # 1/255
    assert names[norm_idx + 1] == "Rescaling"   # 1/sqrt(stddev), only with weights='imagenet'
    extra = backbone.layers[norm_idx + 1]
    assert extra.scale == pytest.approx([1.0 / math.sqrt(s) for s in IMAGENET_STDDEV_RGB])


@pytest.mark.parametrize("backbone_type", list(ENSEMBLE_WEIGHTS_FILENAMES))
def test_backbone_layers_match_delivered_checkpoint(backbone_type):
    """Every layer stored in the delivered .weights.h5 (including variable-less ones like
    Rescaling) must exist in the rebuilt backbone — load_weights() alone does not check this.
    Keras keys each stored layer by its snake_cased class plus a counter (rescaling_1...)."""
    weights_path = _DELIVERED_ARTIFACTS / ENSEMBLE_WEIGHTS_FILENAMES[backbone_type]
    if not weights_path.exists():
        pytest.skip(f"Artefacto entregado no disponible: {weights_path}")
    with h5py.File(weights_path, "r") as fh:
        stored = Counter(re.sub(r"_\d+$", "", key) for key in fh["layers/functional/layers"].keys())
    rebuilt = Counter(to_snake_case(type(layer).__name__) for layer in _backbone(backbone_type).layers)
    assert rebuilt == stored


def test_missing_rf_riesgo_does_not_block_load(monkeypatch):
    monkeypatch.setattr(model_loader, "_load_class_mapping", dict)
    monkeypatch.setattr(model_loader, "YOLO", lambda _path: object())
    monkeypatch.setattr(model_loader, "build_hybrid_model", lambda _name: _NoWeights())
    monkeypatch.setattr(model_loader._store, "path", _missing_joblib)  # pylint: disable=protected-access

    bundle = model_loader.load_model_bundle()

    assert bundle["rf_riesgo"] is None
    assert "rf_clase" not in bundle


def _missing_joblib(filename):
    if filename.endswith(".joblib"):
        raise FileNotFoundError(filename)
    return Path(filename)


class _NoWeights:
    def load_weights(self, _path):
        return None


def test_tabular_only_without_rf_riesgo_raises_data_contract_error():
    sensors = preprocessing.validate_and_align_sensors(_SENSORS_OK)
    with pytest.raises(DataContractError, match="Envía una imagen"):
        inference.run_inference_tabular_only({"rf_riesgo": None}, sensors)


class _ConstantRegressor:
    def predict(self, x):
        assert x.shape == (1, len(SENSOR_COLS))
        return [0.42]


def test_predict_batch_ignores_non_sensor_columns(monkeypatch, tmp_path):
    """The team's own test_split.csv format (ID, label, cached path...) must not fail rows."""
    monkeypatch.setattr(plugin.model_loader, "load_model_bundle", lambda: {
        "yolo_model": None, "yolo_device": "cpu", "ensemble": {}, "class_mapping": {},
        "rf_riesgo": _ConstantRegressor(),
    })
    instance = plugin.Ml1WinePestDiseaseDetectionPlugin()
    instance.load()

    row = {"ID": 7, "Etiqueta": "MILDIU", "Cached_Path": "x.jpg", "Label_ID": 11,
           "CO2_ppm": 400.0, "Path_Imagen": "N/A", **_SENSORS_OK}
    batch_zip = tmp_path / "batch.zip"
    with zipfile.ZipFile(batch_zip, "w") as archive:
        archive.writestr("data.csv", pd.DataFrame([row]).to_csv(index=False))

    response = instance.predict_batch(data_path=str(batch_zip))

    assert response.n_predictions == 1
    assert response.predictions[0].error is None
    assert response.predictions[0].riesgo == pytest.approx(0.42)
