import logging
from typing import Any

import torch

from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_store = ArtifactStore("cow_behavior")

DETECTOR_FILENAME = "model_final.pth"
CLASSIFIER_FILENAME = "best_model.pth"
CLIP_LENGTH = 32
ALPHA = 4
CROP_SIZE = 224


def load_artifacts() -> tuple[Any, Any, dict[str, int], dict[int, str], int]:
    detector_path = _store.path(DETECTOR_FILENAME)
    classifier_path = _store.path(CLASSIFIER_FILENAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(detector_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = device
    detector = DefaultPredictor(cfg)
    logger.info("Detector loaded.")

    checkpoint = torch.load(str(classifier_path), map_location=device)
    behavior_to_idx: dict[str, int] = checkpoint.get("behavior_to_idx", {})
    idx_to_behavior: dict[int, str] = {v: k for k, v in behavior_to_idx.items()}

    proj_weight_key = "model.blocks.6.proj.weight"
    num_classes: int = checkpoint["model_state_dict"][proj_weight_key].shape[0]

    for i in range(num_classes):
        idx_to_behavior.setdefault(i, f"unknown_{i}")

    import torch.nn as nn

    class SlowFastCowBehavior(nn.Module):
        def __init__(self, n_classes: int) -> None:
            super().__init__()
            self.model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=False)
            in_features = self.model.blocks[-1].proj.in_features
            self.model.blocks[-1].proj = nn.Linear(in_features, n_classes)

        def forward(self, x: list) -> torch.Tensor:
            return self.model(x)

    classifier = SlowFastCowBehavior(num_classes).to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    logger.info("Classifier loaded — %d classes, device=%s", num_classes, device)
    return detector, classifier, behavior_to_idx, idx_to_behavior, num_classes
