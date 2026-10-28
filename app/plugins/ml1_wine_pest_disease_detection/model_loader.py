"""Model loading for ml1 — YOLOv8 detector + 3-backbone Keras Deep Ensemble + RandomForest
risk regressor, production variant (Desacoplado + Sin Gases).

CRITICAL (faithful to src/training/train.py, inbox/a01/codigo/): KERAS_BACKEND=torch MUST be set
in the environment BEFORE keras is imported for the first time anywhere in the process, so Keras
3 uses the PyTorch backend the delivered .weights.h5 files were produced with. The delivered
src/predict/predictor.py gets this transitively by importing `from src.training.train import
build_hybrid_model` (train.py sets the env var at module level before its own `import keras`) —
here we set it explicitly at the top of this module instead, since the plugin does not import
the original train.py.
"""
from __future__ import annotations

import json
import logging
import math
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402  pylint: disable=wrong-import-position
import keras  # noqa: E402  pylint: disable=wrong-import-position
import torch  # noqa: E402  pylint: disable=wrong-import-position
from keras import layers, models  # noqa: E402  pylint: disable=wrong-import-position
from ultralytics import YOLO  # noqa: E402  pylint: disable=wrong-import-position

from app.infrastructure.artifact_store import ArtifactStore  # noqa: E402  pylint: disable=wrong-import-position
from app.plugins.ml1_wine_pest_disease_detection.constants import (  # noqa: E402  pylint: disable=wrong-import-position
    ARTIFACT_FOLDER_NAME,
    CLASS_MAPPING_FILENAME,
    CLASSES,
    ENSEMBLE_WEIGHTS_FILENAMES,
    IMAGE_SIZE,
    IMAGENET_STDDEV_RGB,
    RF_RIESGO_FILENAME,
    SENSOR_COLS,
    YOLO_WEIGHTS_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _yolo_device() -> str:
    """Explicit device string for ultralytics — never let it silently guess (known repo pitfall:
    a detector that defaults to the wrong device returns a real 500 instead of CPU fallback)."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _efficientnet_b0_imagenet_topology(img_shape: tuple) -> "models.Model":
    """EfficientNetB0 with the exact layer graph Keras builds for weights='imagenet', without
    downloading the ImageNet checkpoint.

    The original builds it with weights='imagenet' (src/training/train.py:84-85), and Keras 3
    only then inserts an extra Rescaling(1/sqrt(stddev)) right after the stem Normalization
    (keras/src/applications/efficientnet.py, "if weights == 'imagenet'"). That layer has no
    variables, so building with weights=None still loads the delivered .weights.h5 without any
    error — but feeds the backbone a differently scaled input, and the model collapses to a
    single class. The delivered checkpoint does contain the layer ("rescaling_1").
    """
    base_model = keras.applications.EfficientNetB0(
        input_shape=img_shape, include_top=False, weights=None,
    )

    def _call(layer, *args, **kwargs):
        out = layer(*args, **kwargs)
        if isinstance(layer, layers.Normalization):
            # No explicit name: Keras numbers auto-names with a process-wide counter, so any fixed
            # name can collide with a backbone built earlier (MobileNet already takes
            # "rescaling"). The .weights.h5 keys layers by class + counter, not by layer.name.
            out = layers.Rescaling([1.0 / math.sqrt(stddev) for stddev in IMAGENET_STDDEV_RGB])(out)
        return out

    return keras.models.clone_model(base_model, clone_function=lambda layer: layer, call_function=_call)


def build_hybrid_model(backbone_type: str) -> "models.Model":
    """Faithful port of src/training/train.py::build_hybrid_model, decoupled-mode branch only
    (use_fusion_layer=False — the production config): class comes exclusively from the image
    embedding, risk exclusively from the tabular embedding. Layer order/shapes must match
    exactly for load_weights to align correctly with the delivered .weights.h5 files.
    """
    num_classes = len(CLASSES)
    num_sensors = len(SENSOR_COLS)
    img_shape = (*IMAGE_SIZE, 3)

    input_img = layers.Input(shape=img_shape, name="input_image")
    input_tab = layers.Input(shape=(num_sensors,), name="input_tabular")

    if backbone_type == "mobilenet":
        base_model = keras.applications.MobileNetV3Small(
            input_shape=img_shape, include_top=False, weights=None,
        )
    elif backbone_type == "efficientnet":
        base_model = _efficientnet_b0_imagenet_topology(img_shape)
    elif backbone_type == "densenet":
        base_model = keras.applications.DenseNet121(
            input_shape=img_shape, include_top=False, weights=None,
        )
    else:
        raise ValueError(f"Backbone {backbone_type} no soportado.")
    # weights=None (not 'imagenet'): load_weights() below overwrites every parameter right
    # after with the delivered fine-tuned checkpoint — downloading ImageNet weights first would
    # only waste a network call during every pod startup. EfficientNetB0 is the one backbone
    # whose TOPOLOGY also depends on that argument — see _efficientnet_b0_imagenet_topology.

    v = base_model(input_img)
    v = layers.GlobalAveragePooling2D()(v)
    v = layers.Dense(128, activation="swish")(v)
    v = layers.BatchNormalization()(v)
    v = layers.Dropout(0.3)(v)

    t = layers.Dense(128, activation="swish")(input_tab)
    t = layers.BatchNormalization()(t)
    t = layers.Dense(64, activation="swish")(t)
    t = layers.GaussianNoise(0.05)(t)
    t = layers.Dense(32, activation="swish")(t)
    t = layers.Dropout(0.4)(t)

    output_clase = layers.Dense(num_classes, activation="softmax", name="clase")(v)
    output_riesgo = layers.Dense(1, activation="linear", name="riesgo")(t)

    return models.Model(inputs=[input_img, input_tab], outputs=[output_clase, output_riesgo])


def _load_class_mapping() -> dict[int, str]:
    path = _store.path(CLASS_MAPPING_FILENAME)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    mapping = {int(k): v for k, v in raw.items()}
    expected = dict(enumerate(CLASSES))
    if mapping != expected:
        raise ValueError(
            f"class_mapping_no_gases.json ({mapping}) no coincide con el orden vinculante de "
            f"CLASSES en constants.py ({expected}) — el modelo entrenó con un orden de clases "
            "distinto al que asume este plugin."
        )
    return mapping


def load_model_bundle() -> dict:
    """Load YOLO, the 3-backbone Deep Ensemble (weights only, architecture rebuilt in-process)
    and the RandomForest risk regressor (no_gases variant, production config)."""
    class_mapping = _load_class_mapping()

    yolo_model = YOLO(str(_store.path(YOLO_WEIGHTS_FILENAME)))

    ensemble: dict[str, "models.Model"] = {}
    for name, filename in ENSEMBLE_WEIGHTS_FILENAMES.items():
        model = build_hybrid_model(name)
        model.load_weights(str(_store.path(filename)))
        ensemble[name] = model

    # Only the RF regressor is loaded: the RF classifier is never used in the decoupled production
    # mode (it would cost ~370 MB of RSS for nothing). Like predictor.py:137-144, a missing RF does
    # not stop the service — image requests keep working; only the no-image path needs it.
    try:
        rf_riesgo = joblib.load(_store.path(RF_RIESGO_FILENAME))
    except FileNotFoundError:
        logger.warning(
            "ml1: %s no disponible — sin imagen no se podrá estimar el riesgo", RF_RIESGO_FILENAME,
        )
        rf_riesgo = None

    logger.info(
        "ml1 bundle loaded — backbones=%s, yolo_device=%s, classes=%d, rf_riesgo=%s",
        list(ensemble), _yolo_device(), len(class_mapping), rf_riesgo is not None,
    )

    return {
        "yolo_model": yolo_model,
        "yolo_device": _yolo_device(),
        "ensemble": ensemble,
        "rf_riesgo": rf_riesgo,
        "class_mapping": class_mapping,
    }
