"""ml8 «desconocido»: images unlike the training ones are not given a cereal/category.

QA: with vine leaves, insects or grain photos the model answered «insectos» + trigo/maíz with
0.9+ confidence. The gate (ood.py) scores the network's pooled features; it never changes the
network's own outputs, only reports «desconocido» and keeps the raw labels apart.
"""
# pylint: disable=protected-access  # exercises the plugin's private helpers directly
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app.plugins.ml8_cereals_img_anomaly_detector import mlflow_utils, ood
from app.plugins.ml8_cereals_img_anomaly_detector import plugin as ml8
from app.plugins.ml8_cereals_img_anomaly_detector.model_loader import MultiTaskMobileNetV3Large
from app.plugins.ml8_cereals_img_anomaly_detector.predict_dto import PredictInlineResponse

_D = 16


def _cloud(center: int, n: int, seed: int) -> np.ndarray:
    """n feature vectors around one axis (a 'domain')."""
    rng = np.random.default_rng(seed)
    base = np.zeros(_D, dtype=np.float32)
    base[center] = 1.0
    return base + 0.15 * rng.standard_normal((n, _D)).astype(np.float32)


def _raw_result() -> dict:
    return {"categoria": "insectos", "cereal": "trigo", "confianza_categoria": 0.97,
            "confianza_cereal": 0.95, "probabilidades_categoria": {"insectos": 0.97},
            "probabilidades_cereal": {"trigo": 0.95}, "model_id": "ml8"}


@pytest.mark.parametrize("with_outliers", [False, True])
def test_gate_keeps_real_images_and_flags_other_domains(with_outliers):
    """Threshold keeps ~95 % of held-out real images; another domain becomes «desconocido»."""
    bank, held_out = _cloud(0, 200, 1), _cloud(0, 100, 2)
    outliers = np.r_[_cloud(3, 60, 3), _cloud(5, 60, 4)] if with_outliers else None
    ref = ood.build_reference(bank, held_out, outliers)
    assert ref.method == ("logreg" if with_outliers else "knn")
    assert (ref.score(held_out) <= ref.threshold).mean() >= 0.95
    unseen_domain = _cloud(9, 50, 5)
    assert (ref.score(unseen_domain) > ref.threshold).mean() > 0.9

    result = ood.apply_gate(_raw_result(), ref, torch.from_numpy(unseen_domain[:1]))
    assert result["es_desconocido"] is True
    assert (result["categoria"], result["cereal"]) == ("desconocido", "desconocido")
    assert (result["categoria_modelo"], result["cereal_modelo"]) == ("insectos", "trigo")
    assert result["confianza_categoria"] == 0.97  # raw network output, untouched
    assert result["puntuacion_ood"] > result["umbral_ood"]

    real = ood.apply_gate(_raw_result(), ref, torch.from_numpy(held_out[:1] * 0 + bank.mean(0)))
    assert real["es_desconocido"] is False and real["categoria"] == "insectos"


def test_without_reference_the_prediction_is_unchanged():
    """Retrained runs from before the gate (no ood_reference.npz) serve as before."""
    result = ood.apply_gate(_raw_result(), None, torch.zeros(1, _D))
    assert result["categoria"] == "insectos" and result["es_desconocido"] is False
    assert result["puntuacion_ood"] is None and result["umbral_ood"] is None
    PredictInlineResponse(**result)


def test_reference_round_trips_through_the_npz(tmp_path):
    """Saved next to the checkpoint and loaded back with the same decisions."""
    ref = ood.build_reference(_cloud(0, 120, 1), _cloud(0, 60, 2), _cloud(4, 80, 3))
    ref.save(str(tmp_path / ood.OOD_REFERENCE_FILENAME))
    loaded = ood.load_reference(str(tmp_path))
    probe = np.r_[_cloud(0, 20, 7), _cloud(4, 20, 8), _cloud(11, 20, 9)]
    assert loaded.method == "logreg" and loaded.threshold == pytest.approx(ref.threshold)
    np.testing.assert_allclose(loaded.score(probe), ref.score(probe), atol=1e-3)
    assert ood.load_reference(str(tmp_path / "missing")) is None


def test_bank_rows_are_scored_leave_one_out_when_fitting():
    """Each training image must not match itself, or its k-NN distance would always be 0."""
    bank = _cloud(0, 30, 1)
    ref = ood.fit_reference(bank)
    assert np.all(ref.knn(bank) < 1e-5)
    assert np.all(ref.knn(bank, exclude_self=True) > 1e-3)


class _TinyMultiTask(nn.Module):
    """Same attribute layout as MultiTaskMobileNetV3Large (features/avgpool/neck/heads)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, _D, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(_D, 8), nn.Hardswish())
        self.classifier_categoria = nn.Linear(8, 4)
        self.classifier_cereal = nn.Linear(8, 4)

    def forward(self, x):
        """(categoría, cereal) logits, like the served model."""
        h = self.neck(torch.flatten(self.avgpool(self.features(x)), 1))
        return self.classifier_categoria(h), self.classifier_cereal(h)


def test_forward_with_features_matches_the_model_forward():
    """The gate's single pass gives the exact logits of the served forward()."""
    model = _TinyMultiTask().eval()
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        cat, cer = model(x)
        cat2, cer2, feats = ood.forward_with_features(model, x)
    assert torch.equal(cat, cat2) and torch.equal(cer, cer2) and feats.shape == (2, _D)


def test_plugin_rows_carry_the_gate_fields(tmp_path):
    """Batch rows and inline responses come out of the same _classify (gate applied)."""
    model = _TinyMultiTask().eval()
    names = ["hongos", "insectos", "otros", "sano"]
    cereals = ["arroz", "maiz", "sorgo", "trigo"]
    feats = []
    for i in range(40):
        Image.new("RGB", (16, 16), (i * 5, 80, 40)).save(tmp_path / f"{i}.png")
        feats.append(ml8._features(model, [(str(tmp_path / f"{i}.png"),)], "cpu")[0])
    feats = np.array(feats)
    ref = ood.build_reference(feats[:30], feats[30:])
    bundle = {"model": model, "device": "cpu", "image_size": 224, "model_id": "ml8",
              "idx_to_class": dict(enumerate(names)), "idx_to_cereal": dict(enumerate(cereals)),
              "ood": ref}
    row = ml8._predict_image(bundle, tmp_path / "0.png")
    assert {"es_desconocido", "categoria_modelo", "puntuacion_ood", "umbral_ood"} <= row.keys()
    assert row["categoria"] in names + ["desconocido"]
    Image.new("RGB", (16, 16), (255, 255, 255)).save(tmp_path / "white.png")
    far = ml8._predict_image(bundle, tmp_path / "white.png")
    assert far["puntuacion_ood"] >= row["puntuacion_ood"]


def test_retrained_run_saves_and_serves_its_own_reference(tmp_path):
    """train() writes ood_reference.npz next to the checkpoint; the MLflow loader picks it up."""
    model = MultiTaskMobileNetV3Large(4, 4)
    ref = ood.build_reference(_cloud(0, 40, 1), _cloud(0, 20, 2))
    ml8._write_model(str(tmp_path), ml8._checkpoint(model), ref)
    bundle = mlflow_utils._load(str(tmp_path))
    assert bundle["ood"] is not None and bundle["ood"].threshold == pytest.approx(ref.threshold)

    (tmp_path / ood.OOD_REFERENCE_FILENAME).unlink()  # run trained before the gate existed
    assert mlflow_utils._load(str(tmp_path))["ood"] is None
