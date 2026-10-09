"""Loads the Detectron2 detector and SlowFast classifier from ArtifactStore.

Detectron2, pytorchvideo and ``torch.hub`` are imported lazily inside
``load_model_bundle`` so that importing this module (e.g. during test
collection) does not require those heavy, source-installed dependencies.
"""
import logging

import torch
from torch import nn

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml5_meat_cow_behaviour.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASSIFIER_FILENAME,
    DETECTOR_FILENAME,
    TRAINING_BEHAVIOR_TO_IDX,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


class SlowFastCowBehavior(nn.Module):
    """SlowFast-R50 backbone with the projection head resized to the cow classes.

    Shared architecture builder: used both to load the served checkpoint (below) and,
    from ``training.py``, to build the module trained/retrained by ``plugin.train()`` and
    to rebuild a user-retrained checkpoint downloaded from MLflow (``mlflow_utils.py``) —
    one definition, so the serving and training code paths can never drift apart.
    """

    def __init__(self, n_classes: int) -> None:
        """Build the SlowFast-R50 backbone with a projection head for ``n_classes``."""
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=False,
        )
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, n_classes)

    def forward(self, x: list) -> torch.Tensor:
        """Run the SlowFast forward pass on a ``[slow, fast]`` pathway list."""
        return self.model(x)


def build_classifier_module(num_classes: int, device: str) -> "SlowFastCowBehavior":
    """Build an un-trained (random head, no checkpoint loaded) SlowFast classifier module."""
    return SlowFastCowBehavior(num_classes).to(device)


def _safe_device() -> str:
    """Return ``"cuda"`` only if it can execute a conv op; otherwise ``"cpu"``."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.nn.Conv2d(1, 1, 1).cuda()(torch.zeros(1, 1, 4, 4).cuda())
        return "cuda"
    except Exception:
        logger.warning("CUDA detectada pero no funcional para operaciones de red — usando CPU.")
        return "cpu"


def _load_detector(detector_path: str, device: str):
    """Build a Detectron2 Faster R-CNN predictor with the trained weights."""
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = detector_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def _load_classifier(classifier_path: str, device: str):
    """Build the SlowFast-R50 classifier and load its trained state dict.

    Returns ``(classifier, behavior_to_idx, idx_to_behavior, num_classes)``.
    """
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=True)

    # BUGFIX (see constants.TRAINING_BEHAVIOR_TO_IDX for the full writeup): the checkpoint's
    # own embedded `behavior_to_idx` comes from configs/slowfast_cow_behavior.py, which is NOT
    # the class order the model's output head actually learned during training (that was
    # utils/data_utils.py's order). Decoding with the checkpoint's embedded dict silently
    # mislabels 10 of 12 classes (only grazing=0 and ruminating-standing=6 happen to agree).
    # We deliberately ignore checkpoint["behavior_to_idx"] for decoding and always use the
    # verified-correct TRAINING_BEHAVIOR_TO_IDX constant instead.
    behavior_to_idx: dict[str, int] = dict(TRAINING_BEHAVIOR_TO_IDX)
    idx_to_behavior: dict[int, str] = {v: k for k, v in behavior_to_idx.items()}

    # Read num_classes from the final projection layer shape.
    proj_weight_key = "model.blocks.6.proj.weight"
    num_classes: int = checkpoint["model_state_dict"][proj_weight_key].shape[0]

    for i in range(num_classes):
        idx_to_behavior.setdefault(i, f"unknown_{i}")

    classifier = build_classifier_module(num_classes, device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    return classifier, behavior_to_idx, idx_to_behavior, num_classes


def load_model_bundle() -> dict:
    """Load detector + classifier from ArtifactStore and return a runtime bundle dict.

    Each file is fetched lazily via _store.path(filename) below, which only reaches out
    to S3 for a file that isn't already present locally (and only if STORAGE_BUCKET is
    set) — unlike the unconditional _store.download_all_if_needed() this used to call,
    which raised EnvironmentError immediately whenever STORAGE_BUCKET was unset, even
    with artifacts already vendored locally (same bug found and fixed for ml23; see
    inbox/a23/manifest.yaml known_issues).
    """
    detector_path = _store.path(DETECTOR_FILENAME)
    classifier_path = _store.path(CLASSIFIER_FILENAME)

    for path in (detector_path, classifier_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Artefacto no encontrado: {path}. "
                "Copia los ficheros .pth en artifacts/ antes de arrancar el servicio."
            )

    device = _safe_device()
    logger.info("Loading Detectron2 detector from %s (device=%s)", detector_path, device)
    detector = _load_detector(str(detector_path), device)

    logger.info("Loading SlowFast classifier from %s", classifier_path)
    classifier, behavior_to_idx, idx_to_behavior, num_classes = _load_classifier(
        str(classifier_path), device
    )

    logger.info(
        "Ml5MeatCowBehaviour bundle ready — %d classes, device=%s, behaviors=%s",
        num_classes, device, list(behavior_to_idx.keys()),
    )

    return {
        "detector": detector,
        "classifier": classifier,
        "behavior_to_idx": behavior_to_idx,
        "idx_to_behavior": idx_to_behavior,
        "num_classes": num_classes,
        "device": device,
    }
