"""Direct unit tests for modelo10-lácteo helper functions.

Tests the pure-logic functions in preprocessing, postprocessing,
and the real plugin class (with mocked model loading).
"""

from __future__ import annotations

import csv
import os
import tempfile
import io
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from app.domain.services.exceptions import InvalidImageError
from app.plugins.ml10_dairy_disease_vector_detection.postprocessing import build_inline_result, classify_crop
from app.plugins.ml10_dairy_disease_vector_detection.preprocessing import (
    CLASSIFIER_TRANSFORM,
    build_eval_transform,
    crop_to_tensor,
    image_base64_to_pil,
)
from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin, _cls_transforms


def _make_mock_classifier(class_idx=0, confidence=0.9):
    """Create a mock classifier that returns the given class index with the given confidence."""
    classifier = MagicMock()
    probs = [0.1, 0.1, 0.1]
    probs[class_idx] = confidence
    logits = torch.tensor(probs).unsqueeze(0)
    classifier.return_value = logits
    return classifier


# ── preprocessing tests ───────────────────────────────────────────────────

class TestImageBase64ToPil:
    """Tests for image_base64_to_pil decoding."""

    def test_decodes_valid_base64(self):
        """Verify a valid base64 image is decoded to a PIL Image."""
        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = image_base64_to_pil(b64)
        assert result.mode == "RGB"
        assert result.size == (10, 10)

    def test_invalid_base64_raises(self):
        """Verify invalid base64 raises an exception."""
        with pytest.raises(Exception):
            image_base64_to_pil("not-valid-base64!!")

    def test_valid_base64_non_image_bytes_raises(self):
        """Verify InvalidImageError when base64 is valid but bytes are not an image."""
        b64 = base64.b64encode(b"these are definitely not image bytes").decode()
        with pytest.raises(InvalidImageError, match="Failed to decode image from base64"):
            image_base64_to_pil(b64)


class TestCropToTensor:
    """Tests for crop_to_tensor shape and edge cases."""

    def test_returns_correct_shape(self):
        """Verify crop_to_tensor returns a tensor of expected shape."""
        img = Image.new("RGB", (100, 100), color="blue")
        tensor = crop_to_tensor(img, 10, 10, 50, 50)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_zero_area_crop_still_produces_tensor(self):
        """Verify even a zero-area crop produces a valid tensor."""
        img = Image.new("RGB", (100, 100), color="green")
        tensor = crop_to_tensor(img, 50, 50, 51, 51)
        assert tensor.shape == (1, 3, 224, 224)

    def test_full_image_crop(self):
        """Verify cropping the full image produces a valid tensor."""
        img = Image.new("RGB", (224, 224), color="white")
        tensor = crop_to_tensor(img, 0, 0, 224, 224)
        assert tensor.shape == (1, 3, 224, 224)

    def test_crop_raises_on_os_error(self):
        """Verify InvalidImageError is raised when the crop operation raises OSError."""
        mock_img = MagicMock(spec=Image.Image)
        mock_img.crop.side_effect = OSError("simulated disk error")
        with pytest.raises(InvalidImageError, match="Failed to crop image"):
            crop_to_tensor(mock_img, 0, 0, 100, 100)


class TestClassifierTransform:
    """Tests for the CLASSIFIER_TRANSFORM preprocessing transform."""

    def test_transform_applies_without_error(self):
        """Verify CLASSIFIER_TRANSFORM applies to a PIL image without error."""
        img = Image.new("RGB", (300, 300), color="gray")
        tensor = CLASSIFIER_TRANSFORM(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_has_center_crop_matching_production_pipeline(self):
        """Verify CLASSIFIER_TRANSFORM uses Resize(256)+CenterCrop(224), matching the real
        delivered production inference code.

        Regression test (Oct-2026 manifest-extraction audit of inbox/a10): a previous version
        of this transform dropped CenterCrop, reasoning (incorrectly) that it diverged from
        training. That reasoning only accounted for ``src/training/trainer.py``/
        ``predict_classifier`` (which evaluate the classifier in isolation on already-cropped
        ImageFolder images with a plain resize) and missed that the two places in the delivered
        code that actually run the real detect-crop-classify flow on a raw image —
        ``predict_full_pipeline`` in ``src/predict/predictor.py`` and
        ``notebooks/evaluation/pipeline.ipynb`` — both use Resize(256)+CenterCrop(224).
        Verified empirically against the one real delivered end-to-end example
        (``inbox/a10/codigo/.../data/test.jpg`` + ``data/predictions/test_results.json``): the
        tick crop at bbox (755,280,840,374), run through the vendored best_classifier.pth,
        scores cls_conf=0.9996 with Resize(256)+CenterCrop(224) (matches the delivered JSON to
        4 decimals) vs. 0.9991 without CenterCrop — see inbox/a10/manifest.yaml golden_cases.
        """
        from torchvision import transforms as T
        assert any(isinstance(t, T.CenterCrop) for t in CLASSIFIER_TRANSFORM.transforms)
        assert any(
            isinstance(t, T.Resize) and t.size == 256 for t in CLASSIFIER_TRANSFORM.transforms
        )

    def test_eval_tfm_intentionally_differs_from_classifier_transform(self):
        """Verify training's eval_tfm and production inference preprocessing are DIFFERENT.

        They serve different purposes in the original delivered code: eval_tfm (used by this
        plugin's own train()) replicates trainer.py's plain Resize((imgsz, imgsz)), used to
        validate the classifier on already-cropped ImageFolder images; CLASSIFIER_TRANSFORM
        replicates predict_full_pipeline's Resize(256)+CenterCrop(224), applied to a raw
        detector crop at real inference. Uses a non-uniform (quadrant-colored) image on
        purpose — a solid-color image can't distinguish a stretch-resize from a
        resize+center-crop, since every pixel is identical either way.
        """
        img = Image.new("RGB", (300, 150), color=(30, 60, 90))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 149, 74], fill=(220, 20, 20))
        draw.rectangle([150, 75, 299, 149], fill=(20, 220, 20))
        _, eval_tfm = _cls_transforms(224)
        assert not torch.equal(eval_tfm(img), CLASSIFIER_TRANSFORM(img))

    def test_build_eval_transform_is_single_source_of_truth(self):
        """Verify CLASSIFIER_TRANSFORM is built via build_eval_transform(224), not a copy."""
        img = Image.new("RGB", (180, 260), color=(200, 10, 90))
        assert torch.equal(CLASSIFIER_TRANSFORM(img), build_eval_transform(224)(img))


# ── postprocessing tests ──────────────────────────────────────────────────

class TestClassifyCrop:
    """Tests for classify_crop prediction accuracy."""

    def test_returns_expected_class(self):
        """Verify classify_crop returns the species with the highest probability."""
        classifier = MagicMock()
        classifier.return_value = torch.tensor([[0.1, 0.8, 0.1]])
        tensor = torch.randn(1, 3, 224, 224)
        species, conf = classify_crop(classifier, tensor, ["fly", "mos", "tick"], "cpu")
        assert species == "mos"

    def test_all_classes_predictable(self):
        """Verify all classes can be predicted correctly."""
        classifier = MagicMock()
        classifier.return_value = torch.tensor([[0.9, 0.05, 0.05]])
        tensor = torch.randn(1, 3, 224, 224)
        species, conf = classify_crop(classifier, tensor, ["fly", "mos", "tick"], "cpu")
        assert species == "fly"

    def test_device_moving(self):
        """Verify classify_crop works with a torch.device argument."""
        classifier = MagicMock()
        classifier.return_value = torch.tensor([[0.2, 0.2, 0.6]])
        tensor = torch.randn(1, 3, 224, 224)
        species, conf = classify_crop(
            classifier, tensor, ["fly", "mos", "tick"], torch.device("cpu")
        )
        assert species == "tick"


class TestBuildInlineResult:
    """Tests for build_inline_result output structure."""

    def test_empty_detections(self):
        """Verify empty detections produce a 'no_vectors' result."""
        result = build_inline_result("ml10-dairy-disease-vector-detection", [])
        assert result == {
            "model_id": "ml10-dairy-disease-vector-detection",
            "prediction": "no_vectors",
            "confidence": 0.0,
            "vectors_count": 0,
            "detections": [],
            "species_summary": {},
        }

    def test_single_detection(self):
        """Verify a single detection produces the correct result."""
        dets = [
            {"species": "fly", "det_conf": 0.9, "cls_conf": 0.95, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        ]
        result = build_inline_result("ml10-dairy-disease-vector-detection", dets)
        assert result["prediction"] == "fly"
        assert result["confidence"] == 0.95
        assert result["vectors_count"] == 1
        assert result["species_summary"] == {"fly": 1}

    def test_multiple_detections_picks_dominant(self):
        """Verify the dominant species is selected from multiple detections."""
        dets = [
            {"species": "mos", "det_conf": 0.7, "cls_conf": 0.6, "bbox": {}},
            {"species": "fly", "det_conf": 0.8, "cls_conf": 0.9, "bbox": {}},
            {"species": "tick", "det_conf": 0.5, "cls_conf": 0.4, "bbox": {}},
        ]
        result = build_inline_result("ml10-dairy-disease-vector-detection", dets)
        assert result["prediction"] == "fly"
        assert result["confidence"] == 0.9
        assert result["vectors_count"] == 3
        assert result["species_summary"] == {"mos": 1, "fly": 1, "tick": 1}

    def test_species_summary_counts(self):
        """Verify species_summary correctly counts detections per species."""
        dets = [
            {"species": "fly", "det_conf": 0.9, "cls_conf": 0.95, "bbox": {}},
            {"species": "fly", "det_conf": 0.8, "cls_conf": 0.85, "bbox": {}},
            {"species": "tick", "det_conf": 0.7, "cls_conf": 0.75, "bbox": {}},
        ]
        result = build_inline_result("ml10-dairy-disease-vector-detection", dets)
        assert result["species_summary"] == {"fly": 2, "tick": 1}


# ── Real plugin tests (without loading ML models) ─────────────────────────

class TestMl10DairyDiseaseVectorDetectionPluginDirect:
    """Tests the real plugin class with mocked model loading."""

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_plugin_initial_state(self, mock_load):
        """Verify the plugin starts in an unloaded state with zero stats."""
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        assert plugin.is_loaded() is False
        assert plugin._predict_count == 0
        assert plugin._total_latency_ms == 0.0
        assert plugin._last_predict_at is None

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_plugin_load_sets_loaded(self, mock_load):
        """Verify calling load() makes is_loaded() return True."""
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        plugin.load()
        assert plugin.is_loaded() is True

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_stats_structure(self, mock_load):
        """Verify stats() returns a StatsResponse with expected fields."""
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        plugin.load()
        stats = plugin.stats()
        assert stats.model_name == "ml10-dairy-disease-vector-detection"
        assert stats.task_type == "object-detection+classification"
        assert stats.framework == "pytorch+ultralytics"
        assert isinstance(stats.inputs, list)
        assert isinstance(stats.outputs, list)
        assert stats.runtime_stats.total_predictions == 0
        assert stats.runtime_stats.avg_latency_ms is None

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_stats_with_predictions(self, mock_load):
        """Verify predict_count reflects the number of predictions made."""
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        plugin.load()
        plugin._update_stats(latency_ms=100.0)
        plugin._update_stats(latency_ms=200.0)
        stats = plugin.stats()
        assert stats.runtime_stats.total_predictions == 2

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_update_stats_tracking(self, mock_load):
        """Verify _update_stats increments predict_count and records latency."""
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        plugin.load()
        assert plugin._predict_count == 0
        assert plugin._last_predict_at is None
        plugin._update_stats(latency_ms=50.0)
        assert plugin._predict_count == 1
        assert plugin._total_latency_ms == 50.0
        assert plugin._last_predict_at is not None

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_assert_loaded_raises_when_not_loaded(self, mock_load):
        """Verify _assert_loaded raises ModelNotLoadedError when not loaded."""
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        from app.domain.services.exceptions import ModelNotLoadedError
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        with pytest.raises(ModelNotLoadedError):
            plugin._assert_loaded()

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_assert_loaded_passes_when_loaded(self, mock_load):
        """Verify _assert_loaded does not raise after load()."""
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        plugin.load()
        plugin._assert_loaded()  # should not raise

    def test_image_paths_from_csv(self):
        """Verify _image_paths_from_csv extracts paths from a CSV."""
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["image_path"])
            writer.writerow(["/data/img001.jpg"])
            writer.writerow(["/data/img002.png"])
            csv_path = f.name
        try:
            paths = plugin._image_paths_from_csv(csv_path)
            assert len(paths) == 2
            assert paths[0] == Path("/data/img001.jpg")
            assert paths[1] == Path("/data/img002.png")
        finally:
            os.unlink(csv_path)

    def test_image_paths_from_csv_empty_rows_skipped(self):
        """Verify empty rows in CSV are skipped."""
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["image_path"])
            writer.writerow(["/data/img001.jpg"])
            writer.writerow([""])
            writer.writerow(["/data/img002.png"])
            csv_path = f.name
        try:
            paths = plugin._image_paths_from_csv(csv_path)
            assert len(paths) == 2
        finally:
            os.unlink(csv_path)


# ── Training helper function tests ────────────────────────────────────────

class TestClsTransforms:
    """Tests for _cls_transforms output shapes."""

    def test_returns_two_transforms(self):
        """Verify _cls_transforms returns train and eval transforms."""
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import _cls_transforms
        train_tfm, eval_tfm = _cls_transforms(224)
        assert train_tfm is not None
        assert eval_tfm is not None

    def test_transform_produces_correct_shape(self):
        """Verify eval transform produces a tensor of the expected shape."""
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import _cls_transforms
        import PIL.Image
        _, eval_tfm = _cls_transforms(224)
        img = PIL.Image.new("RGB", (300, 300), color="red")
        tensor = eval_tfm(img)
        assert tensor.shape == (3, 224, 224)

    def test_custom_size(self):
        """Verify _cls_transforms supports a custom image size."""
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import _cls_transforms
        _, eval_tfm = _cls_transforms(128)
        img = Image.new("RGB", (200, 200), color="blue")
        tensor = eval_tfm(img)
        assert tensor.shape == (3, 128, 128)


class TestCreateSplits:
    """Tests for _create_splits folder structure and empty class handling."""

    def test_creates_expected_structure(self):
        """Verify _create_splits creates train/val/test folders with correct proportions.

        Ratios are 70/15/15, matching ``splits.p_train/p_val/p_test`` in the original
        delivered config/config.yaml (see inbox/a10/manifest.yaml training.hyperparams) — not
        the 80/10/10 a previous version of this function used, which diverged from the
        original delivered split logic.
        """
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import _create_splits, _CLASSES
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            for cls in _CLASSES:
                (data_root / cls).mkdir(parents=True)
                for i in range(10):
                    (data_root / cls / f"img_{i:03d}.jpg").touch()

            with tempfile.TemporaryDirectory() as tmp2:
                splits = _create_splits(data_root, _CLASSES, tmp2)
                for phase in ("train", "val", "test"):
                    for cls in _CLASSES:
                        assert (splits / phase / cls).exists(), f"Missing {splits / phase / cls}"
                        files = list((splits / phase / cls).iterdir())
                        if phase == "train":
                            assert len(files) == 7
                        elif phase == "val":
                            assert len(files) == 1
                        elif phase == "test":
                            assert len(files) == 2

    def test_empty_class_handling(self):
        """Verify _create_splits handles classes with no files gracefully."""
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import _create_splits
        with tempfile.TemporaryDirectory() as tmp:
            # One class has no files
            data_root = Path(tmp) / "data"
            (data_root / "fly").mkdir(parents=True)
            (data_root / "mos").mkdir()
            (data_root / "tick").mkdir()
            for i in range(5):
                (data_root / "fly" / f"img_{i:03d}.jpg").touch()
            (data_root / "tick" / "img_000.jpg").touch()

            with tempfile.TemporaryDirectory() as tmp2:
                # 'bee' class has no directory → triggers continue at line 82
                splits = _create_splits(data_root, ["fly", "mos", "tick", "bee"], tmp2)
                assert (splits / "train" / "fly").exists()
                assert (splits / "train" / "mos").exists()
                assert (splits / "train" / "tick").exists()


# ── Training method error paths ───────────────────────────────────────────

# ── Predict inline with mocked models ────────────────────────────────────

class TestPredictInlineWithMocks:
    """Tests predict_inline with real plugin + mocked detector/classifier."""

    def _make_plugin(self, detector_mock=None, classifier_mock=None):
        """Create a plugin instance with mocked detector and classifier."""
        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (
                detector_mock or MagicMock(),
                classifier_mock or MagicMock(),
                ["fly", "mos", "tick"],
            )
            from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()
            return plugin

    def _make_mock_detector(self, boxes_data=None):
        """Create a mock YOLO detector that returns the given boxes.

        Each item in *boxes_data* is (conf, [x1, y1, x2, y2]).
        We build a list of per-box mocks so ``for box in res.boxes:`` works.
        """
        if boxes_data is None:
            boxes_data = [(0.85, [10, 20, 80, 90])]

        box_mocks = []
        for conf, xyxy in boxes_data:
            b = MagicMock()
            b.conf = torch.tensor([conf])
            b.xyxy = torch.tensor([xyxy])
            box_mocks.append(b)

        mock_box_container = MagicMock()
        mock_box_container.boxes = box_mocks  # list makes it iterable
        mock_detector = MagicMock()
        mock_detector.predict.return_value = [mock_box_container]
        return mock_detector

    def test_predict_inline_with_base64(self):
        """Verify predict_inline works with a base64-encoded image."""
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        classifier = _make_mock_classifier(class_idx=0, confidence=0.95)
        detector = self._make_mock_detector()
        plugin = self._make_plugin(detector, classifier)

        result = plugin.predict_inline(features={"image_base64": b64})
        assert result.model_id == "ml10-dairy-disease-vector-detection"
        assert result.prediction == "fly"
        assert result.vectors_count == 1
        # annotated_image (bbox overlay) must be present on inline responses too — without
        # it, the platform falls back to running Grad-CAM against the full scene with a
        # classifier that only ever saw per-vector crops in training (see plugin.py).
        assert isinstance(result.annotated_image, str) and len(result.annotated_image) > 0
        # heatmap_crops (the exact per-detection crops the classifier itself was fed) must be
        # present too — Grad-CAM needs to run against these crops, not the full uncropped
        # scene, or the resulting heatmap has no real spatial correspondence to any detection.
        assert len(result.heatmap_crops) == 1
        crop = result.heatmap_crops[0]
        assert crop["species"] == "fly"
        assert isinstance(crop["crop_base64"], str) and len(crop["crop_base64"]) > 0
        assert crop["bbox"] == {"x1": 10.0, "y1": 20.0, "x2": 80.0, "y2": 90.0}

    def test_predict_inline_multiple_detections(self):
        """Verify predict_inline handles multiple detections correctly."""
        img = Image.new("RGB", (200, 200), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        boxes_data = [
            (0.85, [10, 20, 50, 60]),
            (0.75, [70, 80, 120, 140]),
        ]
        classifier = _make_mock_classifier(class_idx=1, confidence=0.88)
        detector = self._make_mock_detector(boxes_data)
        plugin = self._make_plugin(detector, classifier)

        result = plugin.predict_inline(features={"image_base64": b64})
        assert result.model_id == "ml10-dairy-disease-vector-detection"
        assert result.vectors_count == 2
        # Both detections get their own crop (2 <= max_crops), not just the top one.
        assert len(result.heatmap_crops) == 2

    def test_predict_inline_heatmap_crops_capped_at_five(self):
        """Verify heatmap_crops caps at the 5 highest-cls_conf detections."""
        img = Image.new("RGB", (300, 300), color="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        boxes_data = [(0.9, [x, x, x + 20, x + 20]) for x in range(0, 210, 30)]  # 7 boxes
        classifier = _make_mock_classifier(class_idx=0, confidence=0.9)
        detector = self._make_mock_detector(boxes_data)
        plugin = self._make_plugin(detector, classifier)

        result = plugin.predict_inline(features={"image_base64": b64})
        assert result.vectors_count == 7
        assert len(result.heatmap_crops) == 5

    def test_predict_inline_no_detections(self):
        """Verify predict_inline returns no_vectors when no detections are found."""
        img = Image.new("RGB", (100, 100), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        mock_detector = MagicMock()
        mock_detector.predict.return_value = []
        plugin = self._make_plugin(mock_detector, MagicMock())

        result = plugin.predict_inline(features={"image_base64": b64})
        assert result.prediction == "no_vectors"
        assert result.vectors_count == 0
        assert result.heatmap_crops == []

    def test_predict_inline_with_image_path(self):
        """Verify predict_inline works with an image_path field."""
        import tempfile
        from pathlib import Path
        detector = self._make_mock_detector()
        classifier = _make_mock_classifier()
        plugin = self._make_plugin(detector, classifier)
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.jpg"
            Image.new("RGB", (50, 50), color="red").save(str(img_path))
            result = plugin.predict_inline(features={"image_path": str(img_path)})
            assert result.model_id == "ml10-dairy-disease-vector-detection"
            assert result.vectors_count == 1

    def test_predict_inline_no_image_field_raises(self):
        """Verify predict_inline raises ValueError when no image field is provided."""
        plugin = self._make_plugin()
        with pytest.raises(ValueError, match="image_path' o 'image_base64"):
            plugin.predict_inline(features={})

    def test_predict_inline_unsupported_extension_raises(self):
        """Verify predict_inline raises on unsupported file extensions."""
        plugin = self._make_plugin()
        with pytest.raises(Exception):
            plugin.predict_inline(features={"image_path": "/tmp/test.gif"})

    def test_predict_inline_updates_stats(self):
        """Verify predict_inline updates predict_count."""
        import io
        import base64
        img = Image.new("RGB", (100, 100), color="yellow")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        plugin = self._make_plugin()
        assert plugin._predict_count == 0
        plugin.predict_inline(features={"image_base64": b64})
        assert plugin._predict_count == 1

    def test_predict_inline_skips_zero_area_detection(self):
        """Verify predict_inline skips boxes with zero area."""
        img = Image.new("RGB", (100, 100), color="cyan")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        box_mocks = []
        b = MagicMock()
        b.conf = torch.tensor([0.8])
        b.xyxy = torch.tensor([[50, 50, 50, 50]])  # zero area: x2==x1, y2==y1
        box_mocks.append(b)
        c = MagicMock()
        c.boxes = box_mocks
        d = MagicMock()
        d.predict.return_value = [c]

        classifier = _make_mock_classifier()
        plugin = self._make_plugin(d, classifier)
        result = plugin.predict_inline(features={"image_base64": b64})
        assert result.vectors_count == 0  # zero-area box skipped


# ── Predict batch with mocked models ─────────────────────────────────────

class TestPredictBatchWithMocks:
    """Tests predict_batch with real plugin + mocked detector/classifier."""

    def _make_mock_detector(self, boxes_data=None):
        """Create a mock YOLO detector returning the given boxes."""
        if boxes_data is None:
            boxes_data = [(0.9, [10, 20, 50, 60])]
        box_mocks = []
        for conf, xyxy in boxes_data:
            b = MagicMock()
            b.conf = torch.tensor([conf])
            b.xyxy = torch.tensor([xyxy])
            box_mocks.append(b)
        c = MagicMock()
        c.boxes = box_mocks
        d = MagicMock()
        d.predict.return_value = [c]
        return d

    def test_predict_batch_directory_mode(self):
        """Verify predict_batch works with a directory of images."""
        import tempfile
        from pathlib import Path

        mock_detector = self._make_mock_detector()
        mock_classifier = _make_mock_classifier()

        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (mock_detector, mock_classifier, ["fly", "mos", "tick"])
            from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                img_path = Path(tmp) / "test.jpg"
                Image.new("RGB", (50, 50), color="red").save(str(img_path))
                result = plugin.predict_batch(data_path=tmp)
                assert result.model_id == "ml10-dairy-disease-vector-detection"
                assert len(result.predictions) == 1
                assert plugin._predict_count == 1

    def test_predict_batch_with_csv(self):
        """Verify predict_batch works with a CSV file listing images."""
        mock_detector = self._make_mock_detector([(0.9, [10, 20, 50, 60])])
        mock_classifier = _make_mock_classifier()
        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (mock_detector, mock_classifier, ["fly", "mos", "tick"])

            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                # Create a real image for the CSV to reference
                img_path = Path(tmp) / "test.jpg"
                Image.new("RGB", (50, 50), color="red").save(str(img_path))
                csv_path = Path(tmp) / "images.csv"
                with open(str(csv_path), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["image_path"])
                    writer.writerow([str(img_path)])
                result = plugin.predict_batch(data_path=str(csv_path))
                assert result.model_id == "ml10-dairy-disease-vector-detection"
                assert len(result.predictions) == 1
                assert plugin._predict_count == 1

    def test_predict_batch_empty_directory_raises(self):
        """Verify predict_batch raises ValueError on empty directories."""
        import tempfile

        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
            from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                with pytest.raises(ValueError, match="No se encontraron imágenes"):
                    plugin.predict_batch(data_path=tmp)

    def test_predict_batch_unsupported_type_raises(self):
        """Verify predict_batch raises ValueError for paths that are not CSV, ZIP, or directory."""
        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
            from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()
            with pytest.raises(ValueError, match="data_path debe ser CSV, ZIP o directorio"):
                plugin.predict_batch(data_path="/fake/path/not_a_dir")

    def test_predict_batch_csv_image_not_found(self):
        """Verify predict_batch handles missing images gracefully (error entry)."""
        mock_detector = self._make_mock_detector()
        mock_classifier = _make_mock_classifier()
        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (mock_detector, mock_classifier, ["fly", "mos", "tick"])
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                csv_path = Path(tmp) / "bad.csv"
                with open(str(csv_path), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["image_path"])
                    writer.writerow([str(Path(tmp) / "nonexistent.jpg")])
                result = plugin.predict_batch(data_path=str(csv_path))
                assert len(result.predictions) == 1
                assert result.predictions[0]["status"] == "error"

    def test_predict_batch_first_row_has_heatmap_crops_like_inline(self):
        """Verify the first batch row carries the same heatmap_crops predict_inline returns.

        The platform's explainability panel only reads dataSample[0] and, without
        heatmap_crops there, falls back to a Grad-CAM request with no image_base64.
        """
        mock_detector = self._make_mock_detector([(0.9, [10, 20, 50, 60])])
        mock_classifier = _make_mock_classifier()
        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (mock_detector, mock_classifier, ["fly", "mos", "tick"])
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                for name in ("a.jpg", "b.jpg"):
                    Image.new("RGB", (100, 100), color="red").save(str(Path(tmp) / name))
                result = plugin.predict_batch(data_path=tmp)
                inline = plugin.predict_inline(features={"image_path": str(Path(tmp) / "a.jpg")})

        first, second = result.predictions
        assert first["filename"] == "a.jpg"
        assert first["heatmap_crops"] == inline.heatmap_crops
        # Only row 0 is read for XAI — don't repeat the crops on every row.
        assert "heatmap_crops" not in second
        # annotated_image is unchanged and still on every row.
        assert first["annotated_image"] and second["annotated_image"]

    def test_predict_batch_heatmap_crops_are_classifier_crops_not_annotated_image(self):
        """Verify each crop is the raw bbox region fed to the classifier, with the fields
        runModelo10CompositeExplain reads (crop_base64, species, cls_conf, bbox)."""
        mock_detector = self._make_mock_detector([(0.9, [10, 20, 50, 60])])
        mock_classifier = _make_mock_classifier(class_idx=2, confidence=0.8)
        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (mock_detector, mock_classifier, ["fly", "mos", "tick"])
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                Image.new("RGB", (100, 100), color="red").save(str(Path(tmp) / "a.jpg"))
                row = plugin.predict_batch(data_path=tmp).predictions[0]

        assert len(row["heatmap_crops"]) == 1
        crop = row["heatmap_crops"][0]
        assert set(crop) == {"species", "cls_conf", "crop_base64", "bbox"}
        assert crop["species"] == "tick"
        assert crop["bbox"] == {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 60.0}
        assert crop["crop_base64"] != row["annotated_image"]
        crop_img = Image.open(io.BytesIO(base64.b64decode(crop["crop_base64"])))
        assert crop_img.format == "JPEG"
        assert crop_img.size == (40, 40)  # the bbox region, not the 100x100 scene


# ── Predict batch zip mode ──────────────────────────────────────────────

class TestPredictBatchZip:
    """Tests predict_batch with ZIP file inputs."""

    def _make_mock_detector(self, boxes_data=None):
        """Create a mock YOLO detector returning the given boxes."""
        if boxes_data is None:
            boxes_data = [(0.9, [10, 20, 50, 60])]
        box_mocks = []
        for conf, xyxy in boxes_data:
            b = MagicMock()
            b.conf = torch.tensor([conf])
            b.xyxy = torch.tensor([xyxy])
            box_mocks.append(b)
        c = MagicMock()
        c.boxes = box_mocks
        d = MagicMock()
        d.predict.return_value = [c]
        return d

    def test_predict_batch_with_zip(self):
        """Verify predict_batch works with a ZIP file of images."""
        import zipfile
        import tempfile
        from pathlib import Path

        mock_detector = self._make_mock_detector()
        mock_classifier = _make_mock_classifier()

        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (mock_detector, mock_classifier, ["fly", "mos", "tick"])
            from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                img_dir = Path(tmp) / "images"
                img_dir.mkdir()
                Image.new("RGB", (50, 50), color="red").save(str(img_dir / "img001.jpg"))
                Image.new("RGB", (50, 50), color="blue").save(str(img_dir / "img002.png"))
                zip_path = Path(tmp) / "dataset.zip"
                with zipfile.ZipFile(str(zip_path), "w") as zf:
                    for f in img_dir.rglob("*"):
                        zf.write(str(f), arcname=f"images/{f.name}")
                result = plugin.predict_batch(data_path=str(zip_path))
                assert result.model_id == "ml10-dairy-disease-vector-detection"
                assert len(result.predictions) == 2

    def test_predict_batch_zip_no_images_raises(self):
        """Verify predict_batch raises ValueError when ZIP has no image files."""
        import zipfile
        import tempfile
        from pathlib import Path

        with patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
            from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
            plugin = Ml10DairyDiseaseVectorDetectionPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                zip_path = Path(tmp) / "empty.zip"
                with zipfile.ZipFile(str(zip_path), "w") as zf:
                    zf.writestr("readme.txt", "no images here")
                with pytest.raises(ValueError, match="No se encontraron imágenes"):
                    plugin.predict_batch(data_path=str(zip_path))


# ── Training error paths ────────────────────────────────────────────────

class TestTrainHelpers:
    """Tests for training helper functions (_cls_train_epoch, _cls_validate)."""

    def test_cls_train_epoch(self):
        """Verify _cls_train_epoch returns float loss and accuracy."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import _cls_train_epoch
        from app.plugins.ml10_dairy_disease_vector_detection.model_loader import _build_mobilenetv3_classifier

        model = _build_mobilenetv3_classifier(3)
        model.train()
        X = torch.randn(16, 3, 224, 224)
        y = torch.randint(0, 3, (16,))
        loader = DataLoader(TensorDataset(X, y), batch_size=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss, acc = _cls_train_epoch(model, loader, optimizer, criterion, "cpu")
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0.0

    def test_cls_validate(self):
        """Verify _cls_validate returns float loss and accuracy."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import _cls_validate
        from app.plugins.ml10_dairy_disease_vector_detection.model_loader import _build_mobilenetv3_classifier

        model = _build_mobilenetv3_classifier(3)
        model.eval()
        X = torch.randn(16, 3, 224, 224)
        y = torch.randint(0, 3, (16,))
        loader = DataLoader(TensorDataset(X, y), batch_size=8)
        criterion = nn.CrossEntropyLoss()

        loss, acc = _cls_validate(model, loader, criterion, "cpu")
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0.0


class TestTrainErrorPaths:
    """Tests for training error handling paths."""

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_train_rejects_non_zip(self, mock_load):
        """Verify train raises ValueError for non-zip paths."""
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        plugin.load()
        with pytest.raises(ValueError, match="debe ser un fichero .zip"):
            plugin.train(data_path="/some/dir/", mlflow_run_id="test-run-id")

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_train_rejects_invalid_zip_structure(self, mock_load):
        """Verify train raises ValueError for ZIP without expected folder structure."""
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        plugin.load()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = f.name
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("random_file.txt", "not a valid dataset")
            with pytest.raises(ValueError, match="ZIP sin estructura"):
                plugin.train(data_path=zip_path, mlflow_run_id="test-run-id")
        finally:
            os.unlink(zip_path)

    @patch("app.plugins.ml10_dairy_disease_vector_detection.plugin.load_detector_and_classifier")
    def test_train_rejects_empty_zip(self, mock_load):
        """Verify train raises ValueError for empty ZIP files."""
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.ml10_dairy_disease_vector_detection.plugin import Ml10DairyDiseaseVectorDetectionPlugin
        plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        plugin.load()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = f.name
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, "w"):
                pass  # empty zip
            with pytest.raises(ValueError, match="ZIP sin estructura"):
                plugin.train(data_path=zip_path, mlflow_run_id="test-run-id")
        finally:
            os.unlink(zip_path)


# ── model_loader tests ────────────────────────────────────────────────────

class TestLoadDetectorAndClassifier:
    """Tests for load_detector_and_classifier() with mocked external I/O."""

    def test_loads_successfully(self, tmp_path):
        """Verify load_detector_and_classifier returns detector, classifier and class names."""
        import json
        import sys
        import torch

        class_names = ["fly", "mos", "tick"]
        (tmp_path / "class_names.json").write_text(json.dumps(class_names))
        (tmp_path / "detector_best.pt").write_bytes(b"fake detector")
        (tmp_path / "best_classifier.pth").write_bytes(b"fake classifier")

        mock_detector = MagicMock()
        mock_classifier = MagicMock()
        mock_state_dict = MagicMock()

        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_detector

        with patch("app.plugins.ml10_dairy_disease_vector_detection.model_loader._store.download_all_if_needed"), \
             patch("app.plugins.ml10_dairy_disease_vector_detection.model_loader._store.path",
                   side_effect=lambda f: tmp_path / f), \
             patch("app.plugins.ml10_dairy_disease_vector_detection.model_loader._build_mobilenetv3_classifier",
                   return_value=mock_classifier), \
             patch("app.plugins.ml10_dairy_disease_vector_detection.model_loader.torch.load",
                   return_value=mock_state_dict), \
             patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            from app.plugins.ml10_dairy_disease_vector_detection.model_loader import load_detector_and_classifier
            det, clf, names = load_detector_and_classifier(torch.device("cpu"))

        assert names == class_names
        assert det is mock_detector
        mock_classifier.load_state_dict.assert_called_once_with(mock_state_dict)
        mock_classifier.eval.assert_called_once()

    def test_raises_when_artifact_file_missing(self, tmp_path):
        """Verify FileNotFoundError when an artifact path does not exist on disk."""
        import json
        import torch

        (tmp_path / "class_names.json").write_text(json.dumps(["fly"]))
        # detector_best.pt and best_classifier.pth deliberately absent

        with patch("app.plugins.ml10_dairy_disease_vector_detection.model_loader._store.download_all_if_needed"), \
             patch("app.plugins.ml10_dairy_disease_vector_detection.model_loader._store.path",
                   side_effect=lambda f: tmp_path / f):
            from app.plugins.ml10_dairy_disease_vector_detection.model_loader import load_detector_and_classifier
            with pytest.raises(FileNotFoundError):
                load_detector_and_classifier(torch.device("cpu"))
