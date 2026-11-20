"""ml10 «desconocido»: detections whose crop does not look like any training vector.

QA: with photos unrelated to the model (vine leaves, cows, poultry droppings…) the detector fired
and the classifier answered fly/mos/tick at ~0.9. The gate (ood.py) scores each accepted crop
with the classifier's pooled features; above the threshold the detection's species is
«desconocido», it stops counting as a vector and gets no Grad-CAM crop. The classifier's own
outputs never change.
"""
# pylint: disable=protected-access  # exercises the plugin's private helpers directly
import base64
import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from app.domain.services.feature_ood import OodReference, build_reference
from app.plugins.ml10_dairy_disease_vector_detection import mlflow_utils, model_loader, ood
from app.plugins.ml10_dairy_disease_vector_detection import plugin as ml10
from app.plugins.ml10_dairy_disease_vector_detection.postprocessing import (
    build_heatmap_crops,
    build_inline_result,
    classify_crop,
    render_annotated_image,
)
from app.plugins.ml10_dairy_disease_vector_detection.preprocessing import crop_to_tensor

_NAMES = ["fly", "mos", "tick"]
_LEFT = (0, 0, 100, 100)    # red half of _scene()
_RIGHT = (100, 0, 200, 100)  # noise half of _scene()


@pytest.fixture(scope="module")
def classifier() -> torch.nn.Module:
    """A real MobileNetV3-Large (random weights) like the served one, in eval mode."""
    torch.manual_seed(0)
    return model_loader._build_mobilenetv3_classifier(len(_NAMES)).eval()


def _scene() -> Image.Image:
    """200x100 image: left half plain red, right half random noise."""
    rng = np.random.default_rng(0)
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[:, :100] = (200, 30, 30)
    arr[:, 100:] = rng.integers(0, 255, (100, 100, 3))
    return Image.fromarray(arr)


def _b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _features(classifier, box) -> np.ndarray:
    return ood.classify_crop_with_features(
        classifier, crop_to_tensor(_scene(), *box), _NAMES, "cpu")[2]


def _knn_reference(bank: np.ndarray, threshold: float = 0.02) -> OodReference:
    """k-NN gate whose bank is *bank*: crops (almost) equal to it pass, the rest do not."""
    return OodReference(bank=bank / np.linalg.norm(bank, axis=1, keepdims=True),
                        threshold=threshold, keep=0.95)


def _detector(boxes) -> MagicMock:
    mocked = []
    for xyxy in boxes:
        box = MagicMock()
        box.conf = torch.tensor([0.9])
        box.xyxy = torch.tensor([list(xyxy)])
        mocked.append(box)
    detector = MagicMock()
    detector.predict.return_value = [SimpleNamespace(boxes=mocked)]
    return detector


def _plugin(classifier, base_ref, boxes=(_LEFT, _RIGHT)):
    with patch.object(ml10, "load_detector_and_classifier",
                      return_value=(_detector(boxes), classifier, _NAMES)), \
         patch.object(ml10, "load_base_reference", return_value=base_ref):
        plugin = ml10.Ml10DairyDiseaseVectorDetectionPlugin()
        plugin._device = torch.device("cpu")
        plugin.load()
    return plugin


def test_features_path_gives_the_same_prediction_as_the_plain_forward(classifier):
    """The gate reads features from the same forward pass: species/confidence unchanged."""
    tensor = crop_to_tensor(_scene(), *_RIGHT)
    species, conf, feats = ood.classify_crop_with_features(classifier, tensor, _NAMES, "cpu")
    assert (species, conf) == pytest.approx(classify_crop(classifier, tensor, _NAMES, "cpu"))
    assert feats.shape == (1, 960)


def test_detection_unlike_training_crops_is_desconocido(classifier):
    """Same image: the crop like the bank is a vector, the other one is «desconocido»."""
    plugin = _plugin(classifier, _knn_reference(_features(classifier, _LEFT)))
    result = plugin.predict_inline(features={"image_base64": _b64(_scene()),
                                             "cls_conf_thresh": 0.0})

    known, unknown = result.detections
    assert known["es_desconocido"] is False and known["species"] in _NAMES
    assert unknown["es_desconocido"] is True and unknown["species"] == "desconocido"
    assert unknown["species_modelo"] in _NAMES  # raw network label kept apart
    assert unknown["puntuacion_ood"] > result.umbral_ood == pytest.approx(0.02)
    assert result.prediction == known["species"] and result.confidence == known["cls_conf"]
    assert (result.vectors_count, result.unknown_count) == (1, 1)
    assert result.species_summary == {known["species"]: 1}
    assert result.es_desconocido is False
    assert [c["species"] for c in result.heatmap_crops] == [known["species"]]


def test_image_with_only_unknown_detections_is_desconocido(classifier):
    """Every crop unlike the training ones: prediction «desconocido», no vectors, no Grad-CAM."""
    far = np.zeros((1, 960), np.float32)
    far[0, 0] = 1.0
    plugin = _plugin(classifier, _knn_reference(far, threshold=1e-6))
    result = plugin.predict_inline(features={"image_base64": _b64(_scene()),
                                             "cls_conf_thresh": 0.0})
    assert result.prediction == "desconocido" and result.es_desconocido is True
    assert result.confidence == 0.0
    assert (result.vectors_count, result.unknown_count) == (0, 2)
    assert result.species_summary == {}
    assert result.heatmap_crops == []  # pylint: disable=use-implicit-booleaness-not-comparison
    assert result.annotated_image  # unknown boxes are still drawn (grey, «desconocido»)


def test_without_reference_predictions_are_as_before(classifier):
    """No ood_reference.npz (or a run retrained before the gate): no detection is unknown."""
    plugin = _plugin(classifier, None)
    result = plugin.predict_inline(features={"image_base64": _b64(_scene()),
                                             "cls_conf_thresh": 0.0})
    assert result.vectors_count == 2 and result.unknown_count == 0
    assert result.umbral_ood is None
    assert all(d["es_desconocido"] is False and d["puntuacion_ood"] is None
               for d in result.detections)


def test_user_model_uses_its_own_reference(classifier, tmp_path):
    """A retrained run is gated with the reference /train computed for it, not the base one."""
    plugin = _plugin(classifier, None)
    far = np.zeros((1, 960), np.float32)
    far[0, 0] = 1.0
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    loaded = (classifier, _NAMES, _knn_reference(far, threshold=1e-6), str(user_dir))
    with patch.object(ml10, "download_user_classifier_from_mlflow", return_value=loaded):
        result = plugin.predict_inline(features={"image_base64": _b64(_scene()),
                                                 "cls_conf_thresh": 0.0}, mlflow_run_id="r1")
    assert result.prediction == "desconocido"
    assert not user_dir.exists()


def test_batch_rows_carry_the_gate(classifier, tmp_path):
    """Batch rows get the same fields; prediction «desconocido» when nothing is recognised."""
    far = np.zeros((1, 960), np.float32)
    far[0, 0] = 1.0
    plugin = _plugin(classifier, _knn_reference(far, threshold=1e-6), boxes=[_RIGHT])
    _scene().save(tmp_path / "a.png")
    with patch.object(ml10, "DEFAULT_CLS_CONF", 0.0):
        rows = plugin.predict_batch(data_path=str(tmp_path)).predictions
    assert rows[0]["prediction"] == "desconocido" and rows[0]["unknown_count"] == 1
    assert rows[0]["umbral_ood"] == pytest.approx(1e-6)


def test_inline_result_and_heatmaps_ignore_unknown_detections():
    """A higher-confidence unknown detection never becomes the dominant species."""
    dets = [
        {"species": "tick", "cls_conf": 0.7, "det_conf": 0.5, "es_desconocido": False,
         "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        {"species": "desconocido", "species_modelo": "fly", "cls_conf": 0.99, "det_conf": 0.9,
         "es_desconocido": True, "bbox": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}},
    ]
    result = build_inline_result("ml10", dets)
    assert (result["prediction"], result["confidence"]) == ("tick", 0.7)
    assert result["species_summary"] == {"tick": 1} and result["unknown_count"] == 1
    image = Image.new("RGB", (30, 30))
    assert [c["species"] for c in build_heatmap_crops(image, dets)] == ["tick"]
    assert render_annotated_image(image, dets)


def test_retrained_reference_needs_the_crops_zip(classifier):
    """Without ood_crops.zip a retrained run has no gate (its crops alone would over-flag)."""
    with patch.object(ml10, "base_artifact_file", return_value=None):
        assert ml10._retrained_reference(classifier, "cpu", None, None) is None


def test_retrained_reference_is_built_from_run_and_zip_crops(classifier, tmp_path):
    """/train: bank = run train crops + id_train, threshold on val + id_val, outliers fitted."""
    rng = np.random.default_rng(1)

    def png(color=None) -> bytes:
        arr = (np.full((40, 40, 3), color, np.uint8) if color is not None
               else rng.integers(0, 255, (40, 40, 3), dtype=np.uint8))
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    zip_path = tmp_path / ood.OOD_CROPS_FILENAME
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(4):
            zf.writestr(f"{ood.ID_TRAIN_DIR}/{i}.png", png((200, 30 + i, 30)))
            zf.writestr(f"{ood.ID_VAL_DIR}/{i}.png", png((200, 32 + i, 30)))
            zf.writestr(f"{ood.OUTLIERS_DIR}/ruido/{i}.png", png())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train, val = [], []
    for i in range(4):
        (run_dir / f"t{i}.png").write_bytes(png((190, 40 + i, 40)))
        (run_dir / f"v{i}.png").write_bytes(png((190, 42 + i, 40)))
        train.append((str(run_dir / f"t{i}.png"), 0))
        val.append((str(run_dir / f"v{i}.png"), 0))

    with patch.object(ml10, "base_artifact_file", return_value=str(zip_path)):
        ref = ml10._retrained_reference(classifier, "cpu", SimpleNamespace(samples=train),
                                        SimpleNamespace(samples=val))
    assert ref.method == "logreg" and len(ref.bank) == 8
    assert np.isfinite(ref.threshold)


def test_mlflow_run_reference_round_trip(classifier, tmp_path):
    """A run with ood_reference.npz loads it; one trained before the gate loads None."""
    torch.save(classifier.state_dict(), tmp_path / "best_classifier.pth")
    (tmp_path / "class_names.json").write_text(json.dumps(_NAMES))
    assert mlflow_utils._load(str(tmp_path))[2] is None

    ref = build_reference(np.eye(4, 960, dtype=np.float32), np.eye(4, 960, k=1, dtype=np.float32))
    ref.save(str(tmp_path / "ood_reference.npz"))
    loaded = mlflow_utils._load(str(tmp_path))[2]
    assert loaded is not None and loaded.threshold == pytest.approx(ref.threshold)


def test_base_reference_is_optional(tmp_path):
    """The base artifact may lack ood_reference.npz: the plugin then serves without the gate."""
    with patch.object(model_loader._store, "optional_path", return_value=None):
        assert model_loader.load_base_reference() is None
    build_reference(np.eye(3, 960, dtype=np.float32), np.eye(3, 960, k=1, dtype=np.float32)
                    ).save(str(tmp_path / "ood_reference.npz"))
    with patch.object(model_loader._store, "optional_path",
                      return_value=tmp_path / "ood_reference.npz"):
        assert model_loader.load_base_reference() is not None
