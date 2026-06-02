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

from app.plugins.modelo10_lacteo.postprocessing import build_inline_result, classify_crop
from app.plugins.modelo10_lacteo.preprocessing import CLASSIFIER_TRANSFORM, crop_to_tensor, image_base64_to_pil


# ── preprocessing tests ───────────────────────────────────────────────────

class TestImageBase64ToPil:
    def test_decodes_valid_base64(self):
        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = image_base64_to_pil(b64)
        assert result.mode == "RGB"
        assert result.size == (10, 10)

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            image_base64_to_pil("not-valid-base64!!")


class TestCropToTensor:
    def test_returns_correct_shape(self):
        img = Image.new("RGB", (100, 100), color="blue")
        tensor = crop_to_tensor(img, 10, 10, 50, 50)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_zero_area_crop_still_produces_tensor(self):
        img = Image.new("RGB", (100, 100), color="green")
        tensor = crop_to_tensor(img, 50, 50, 51, 51)
        assert tensor.shape == (1, 3, 224, 224)

    def test_full_image_crop(self):
        img = Image.new("RGB", (224, 224), color="white")
        tensor = crop_to_tensor(img, 0, 0, 224, 224)
        assert tensor.shape == (1, 3, 224, 224)


class TestClassifierTransform:
    def test_transform_applies_without_error(self):
        img = Image.new("RGB", (300, 300), color="gray")
        tensor = CLASSIFIER_TRANSFORM(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)


# ── postprocessing tests ──────────────────────────────────────────────────

class TestClassifyCrop:
    def test_returns_expected_class(self):
        classifier = MagicMock()
        classifier.return_value = torch.tensor([[0.1, 0.8, 0.1]])
        tensor = torch.randn(1, 3, 224, 224)
        species, conf = classify_crop(classifier, tensor, ["fly", "mos", "tick"], "cpu")
        assert species == "mos"

    def test_all_classes_predictable(self):
        classifier = MagicMock()
        classifier.return_value = torch.tensor([[0.9, 0.05, 0.05]])
        tensor = torch.randn(1, 3, 224, 224)
        species, conf = classify_crop(classifier, tensor, ["fly", "mos", "tick"], "cpu")
        assert species == "fly"

    def test_device_moving(self):
        classifier = MagicMock()
        classifier.return_value = torch.tensor([[0.2, 0.2, 0.6]])
        tensor = torch.randn(1, 3, 224, 224)
        species, conf = classify_crop(
            classifier, tensor, ["fly", "mos", "tick"], torch.device("cpu")
        )
        assert species == "tick"


class TestBuildInlineResult:
    def test_empty_detections(self):
        result = build_inline_result("modelo10-lacteo", [])
        assert result == {
            "model_id": "modelo10-lacteo",
            "prediction": "no_vectors",
            "confidence": 0.0,
            "vectors_count": 0,
            "detections": [],
            "species_summary": {},
        }

    def test_single_detection(self):
        dets = [
            {"species": "fly", "det_conf": 0.9, "cls_conf": 0.95, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        ]
        result = build_inline_result("modelo10-lacteo", dets)
        assert result["prediction"] == "fly"
        assert result["confidence"] == 0.95
        assert result["vectors_count"] == 1
        assert result["species_summary"] == {"fly": 1}

    def test_multiple_detections_picks_dominant(self):
        dets = [
            {"species": "mos", "det_conf": 0.7, "cls_conf": 0.6, "bbox": {}},
            {"species": "fly", "det_conf": 0.8, "cls_conf": 0.9, "bbox": {}},
            {"species": "tick", "det_conf": 0.5, "cls_conf": 0.4, "bbox": {}},
        ]
        result = build_inline_result("modelo10-lacteo", dets)
        assert result["prediction"] == "fly"
        assert result["confidence"] == 0.9
        assert result["vectors_count"] == 3
        assert result["species_summary"] == {"mos": 1, "fly": 1, "tick": 1}

    def test_species_summary_counts(self):
        dets = [
            {"species": "fly", "det_conf": 0.9, "cls_conf": 0.95, "bbox": {}},
            {"species": "fly", "det_conf": 0.8, "cls_conf": 0.85, "bbox": {}},
            {"species": "tick", "det_conf": 0.7, "cls_conf": 0.75, "bbox": {}},
        ]
        result = build_inline_result("modelo10-lacteo", dets)
        assert result["species_summary"] == {"fly": 2, "tick": 1}


# ── Real plugin tests (without loading ML models) ─────────────────────────

class TestModelo10LacteoPluginDirect:
    """Tests the real plugin class with mocked model loading."""

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_plugin_initial_state(self, mock_load):
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        assert plugin.is_loaded() is False
        assert plugin._predict_count == 0
        assert plugin._total_latency_ms == 0.0
        assert plugin._last_predict_at is None

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_plugin_load_sets_loaded(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        plugin.load()
        assert plugin.is_loaded() is True

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_stats_structure(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        plugin.load()
        stats = plugin.stats()
        assert stats.model_name == "modelo10-lacteo"
        assert stats.model_type == "classification"
        assert stats.framework == "pytorch+ultralytics"
        assert isinstance(stats.artifact_path, str)
        assert isinstance(stats.input_schema, dict)
        assert isinstance(stats.output_schema, dict)
        assert stats.predict_count == 0
        assert stats.last_predict_at is None

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_stats_with_predictions(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        plugin.load()
        plugin._update_stats(latency_ms=100.0)
        plugin._update_stats(latency_ms=200.0)
        stats = plugin.stats()
        assert stats.predict_count == 2

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_update_stats_tracking(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        plugin.load()
        assert plugin._predict_count == 0
        assert plugin._last_predict_at is None
        plugin._update_stats(latency_ms=50.0)
        assert plugin._predict_count == 1
        assert plugin._total_latency_ms == 50.0
        assert plugin._last_predict_at is not None

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_assert_loaded_raises_when_not_loaded(self, mock_load):
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        from app.domain.services.exceptions import ModelNotLoadedError
        plugin = Modelo10LacteoPlugin()
        with pytest.raises(ModelNotLoadedError):
            plugin._assert_loaded()

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_assert_loaded_passes_when_loaded(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        plugin.load()
        plugin._assert_loaded()  # should not raise

    def test_image_paths_from_csv(self):
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
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
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
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
    def test_returns_two_transforms(self):
        from app.plugins.modelo10_lacteo.plugin import _cls_transforms
        train_tfm, eval_tfm = _cls_transforms(224)
        assert train_tfm is not None
        assert eval_tfm is not None

    def test_transform_produces_correct_shape(self):
        from app.plugins.modelo10_lacteo.plugin import _cls_transforms
        import PIL.Image
        _, eval_tfm = _cls_transforms(224)
        img = PIL.Image.new("RGB", (300, 300), color="red")
        tensor = eval_tfm(img)
        assert tensor.shape == (3, 224, 224)

    def test_custom_size(self):
        from app.plugins.modelo10_lacteo.plugin import _cls_transforms
        _, eval_tfm = _cls_transforms(128)
        img = Image.new("RGB", (200, 200), color="blue")
        tensor = eval_tfm(img)
        assert tensor.shape == (3, 128, 128)


class TestCreateSplits:
    def test_creates_expected_structure(self):
        from app.plugins.modelo10_lacteo.plugin import _create_splits, _CLASSES
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
                            assert len(files) == 8
                        elif phase == "val":
                            assert len(files) == 1
                        elif phase == "test":
                            assert len(files) == 1

    def test_empty_class_handling(self):
        from app.plugins.modelo10_lacteo.plugin import _create_splits
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
                splits = _create_splits(data_root, ["fly", "mos", "tick"], tmp2)
                assert (splits / "train" / "fly").exists()
                assert (splits / "train" / "mos").exists()
                assert (splits / "train" / "tick").exists()


# ── Training method error paths ───────────────────────────────────────────

# ── Predict inline with mocked models ────────────────────────────────────

class TestPredictInlineWithMocks:
    """Tests predict_inline with real plugin + mocked detector/classifier."""

    def _make_plugin(self, detector_mock=None, classifier_mock=None):
        with patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (
                detector_mock or MagicMock(),
                classifier_mock or MagicMock(),
                ["fly", "mos", "tick"],
            )
            from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
            plugin = Modelo10LacteoPlugin()
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

    def _make_mock_classifier(self, class_idx=0, confidence=0.9):
        classifier = MagicMock()
        probs = [0.1, 0.1, 0.1]
        probs[class_idx] = confidence
        logits = torch.tensor(probs).unsqueeze(0)
        classifier.return_value = logits
        return classifier

    def test_predict_inline_with_base64(self):
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        classifier = self._make_mock_classifier(class_idx=0, confidence=0.95)
        detector = self._make_mock_detector()
        plugin = self._make_plugin(detector, classifier)

        result = plugin.predict_inline(features={"image_base64": b64})
        assert result["model_id"] == "modelo10-lacteo"
        assert result["prediction"] == "fly"
        assert result["vectors_count"] == 1

    def test_predict_inline_multiple_detections(self):
        img = Image.new("RGB", (200, 200), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        boxes_data = [
            (0.85, [10, 20, 50, 60]),
            (0.75, [70, 80, 120, 140]),
        ]
        classifier = self._make_mock_classifier(class_idx=1, confidence=0.88)
        detector = self._make_mock_detector(boxes_data)
        plugin = self._make_plugin(detector, classifier)

        result = plugin.predict_inline(features={"image_base64": b64})
        assert result["model_id"] == "modelo10-lacteo"
        assert result["vectors_count"] == 2

    def test_predict_inline_no_detections(self):
        img = Image.new("RGB", (100, 100), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        mock_detector = MagicMock()
        mock_detector.predict.return_value = []
        plugin = self._make_plugin(mock_detector, MagicMock())

        result = plugin.predict_inline(features={"image_base64": b64})
        assert result["prediction"] == "no_vectors"
        assert result["vectors_count"] == 0

    def test_predict_inline_no_image_field_raises(self):
        plugin = self._make_plugin()
        with pytest.raises(ValueError, match="image_path' o 'image_base64"):
            plugin.predict_inline(features={})

    def test_predict_inline_unsupported_extension_raises(self):
        plugin = self._make_plugin()
        with pytest.raises(Exception):
            plugin.predict_inline(features={"image_path": "/tmp/test.gif"})

    def test_predict_inline_updates_stats(self):
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


# ── Predict batch with mocked models ─────────────────────────────────────

class TestPredictBatchWithMocks:
    def _make_mock_detector(self, boxes_data=None):
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
        import tempfile
        from pathlib import Path

        mock_detector = self._make_mock_detector()

        with patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (mock_detector, MagicMock(), ["fly", "mos", "tick"])
            from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
            plugin = Modelo10LacteoPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                img_path = Path(tmp) / "test.jpg"
                Image.new("RGB", (50, 50), color="red").save(str(img_path))
                result = plugin.predict_batch(data_path=tmp)
                assert result["model_id"] == "modelo10-lacteo"
                assert len(result["predictions"]) == 1
                assert plugin._predict_count == 1

    def test_predict_batch_with_csv(self):
        with patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
            from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
            plugin = Modelo10LacteoPlugin()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                writer = csv.writer(f)
                writer.writerow(["image_path"])
                writer.writerow(["/fake/path/img.jpg"])
                csv_path = f.name
            try:
                with pytest.raises(Exception):
                    plugin.predict_batch(data_path=csv_path)
            finally:
                os.unlink(csv_path)

    def test_predict_batch_empty_directory_raises(self):
        import tempfile

        with patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
            from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
            plugin = Modelo10LacteoPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                with pytest.raises(ValueError, match="No se encontraron imágenes"):
                    plugin.predict_batch(data_path=tmp)


# ── Predict batch zip mode ──────────────────────────────────────────────

class TestPredictBatchZip:
    def test_predict_batch_with_zip(self):
        import zipfile
        import tempfile
        from pathlib import Path

        mock_detector = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.conf = torch.tensor([[0.9]])
        mock_boxes.xyxy = torch.tensor([[[10, 20, 50, 60]]])
        mock_box_container = MagicMock()
        mock_box_container.boxes = mock_boxes
        mock_detector.predict.return_value = [mock_box_container]

        with patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (mock_detector, MagicMock(), ["fly", "mos", "tick"])
            from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
            plugin = Modelo10LacteoPlugin()
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
                assert result["model_id"] == "modelo10-lacteo"
                assert len(result["predictions"]) == 2

    def test_predict_batch_zip_no_images_raises(self):
        import zipfile
        import tempfile
        from pathlib import Path

        with patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
            from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
            plugin = Modelo10LacteoPlugin()
            plugin.load()

            with tempfile.TemporaryDirectory() as tmp:
                zip_path = Path(tmp) / "empty.zip"
                with zipfile.ZipFile(str(zip_path), "w") as zf:
                    zf.writestr("readme.txt", "no images here")
                with pytest.raises(ValueError, match="No se encontraron imágenes"):
                    plugin.predict_batch(data_path=str(zip_path))


# ── Training error paths ────────────────────────────────────────────────

class TestTrainErrorPaths:
    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_train_rejects_non_zip(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        plugin.load()
        with pytest.raises(ValueError, match="debe ser un fichero .zip"):
            plugin.train(data_path="/some/dir/")

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_train_rejects_invalid_zip_structure(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        plugin.load()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = f.name
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("random_file.txt", "not a valid dataset")
            with pytest.raises(ValueError, match="ZIP sin estructura"):
                plugin.train(data_path=zip_path)
        finally:
            os.unlink(zip_path)

    @patch("app.plugins.modelo10_lacteo.plugin.load_detector_and_classifier")
    def test_train_rejects_empty_zip(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), ["fly", "mos", "tick"])
        from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
        plugin = Modelo10LacteoPlugin()
        plugin.load()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = f.name
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, "w"):
                pass  # empty zip
            with pytest.raises(ValueError, match="ZIP sin estructura"):
                plugin.train(data_path=zip_path)
        finally:
            os.unlink(zip_path)
