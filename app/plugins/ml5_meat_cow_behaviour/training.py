"""Fixed training pipeline for the SlowFast cow-behaviour classifier.

Ports scripts/training/train_classifier.py + utils/data_utils.py +
scripts/data/format.py from inbox/a05/codigo/ into the plugin, with the 4 bugs
documented in inbox/a05/manifest.yaml known_issues ("[NUEVO Ciclo 2 ... 4 bugs en el
código de ENTRENAMIENTO entregado]") corrected — and those same 4 fixes applied back to
the delivered training repo copy itself (inbox/a05/codigo/), so it is also correct for
anyone running it standalone outside this platform:

  1. LR scheduler: ``scheduler.step()`` is now called once per BATCH inside the training
     loop below, matching ``lr_lambda``'s step-counting formula
     (``total_steps = len(train_loader) * epochs``). Before, it was called once per EPOCH,
     so the LR never left the warmup ramp during a real multi-epoch run.
  2. Train/val/test split (``split_clips``): splits by CLIP/video identity, never by
     individual frame — no clip can appear in more than one of train/val/test.
  3. ``track_id`` (``load_all_clips``): namespaced by ``clip_name`` so it is globally
     unique across clips/videos (CVAT exports it as a small per-clip animal index that
     collides across different videos otherwise).
  4. ColorJitter (``ClipClassificationDataset``): only part of the default transform
     pipeline when ``is_train=True`` — the eval/val path never augments.

Only the classifier is in scope here. The detector (Faster R-CNN) remains out of
``/train``'s scope (bbox-annotation-only ZIP contract mismatch, unrelated to these 4 bugs
— see inbox/a05/manifest.yaml training.reason).
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from typing import Any, Callable

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from app.plugins.ml5_meat_cow_behaviour.constants import (
    CROP_SIZE,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_SEED,
    TRAIN_SPLIT_RATIOS,
    TRAIN_WARMUP_EPOCHS,
    TRAIN_WEIGHT_DECAY,
    TRAINING_BEHAVIOR_TO_IDX,
)
from app.plugins.ml5_meat_cow_behaviour.model_loader import build_classifier_module
from app.plugins.ml5_meat_cow_behaviour.preprocessing import batched_slowfast_pathways

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance — ported verbatim from utils/focal_loss.py (not a bug)."""

    def __init__(self, alpha: float = FOCAL_LOSS_ALPHA, gamma: float = FOCAL_LOSS_GAMMA) -> None:
        """Store the focal-loss balancing (``alpha``) and focusing (``gamma``) factors."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the mean focal loss over a batch of logits and integer targets."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def load_all_clips(raw_frames_dir: str, annotations_dir: str) -> dict:
    """Consolidate all per-clip CVAT/COCO annotation exports into one COCO-style dict.

    Ported from scripts/data/format.py::load_all_clips with BUGFIX #3: ``track_id`` is
    namespaced by ``clip_name`` so it is globally unique across clips (CVAT exports it
    as a small per-clip animal index — e.g. only 23 distinct values for 502 real clips
    before this fix — which silently merges different videos' animals into one "track").
    """
    clips = sorted(
        d for d in os.listdir(raw_frames_dir) if os.path.isdir(os.path.join(raw_frames_dir, d))
    )

    coco_data: dict[str, Any] = {"images": [], "annotations": []}
    image_id_counter = 0
    annotation_id_counter = 0

    for clip_name in clips:
        ann_file = os.path.join(annotations_dir, clip_name, "annotations", "instances_default.json")
        if not os.path.exists(ann_file):
            logger.warning("No se encontró %s — clip omitido.", ann_file)
            continue

        with open(ann_file, "r", encoding="utf-8") as f:
            clip_data = json.load(f)

        old_to_new_image_id: dict[int, int] = {}
        for img in clip_data["images"]:
            old_to_new_image_id[img["id"]] = image_id_counter
            coco_data["images"].append({
                "id": image_id_counter,
                "original_filename": os.path.basename(img["file_name"]),
                "clip_name": clip_name,
            })
            image_id_counter += 1

        for ann in clip_data["annotations"]:
            if ann["image_id"] not in old_to_new_image_id:
                continue
            attributes = dict(ann.get("attributes", {}))
            if "track_id" in attributes:
                # BUGFIX #3: namespace by clip so track_id is globally unique.
                attributes["track_id"] = f"{clip_name}__track{attributes['track_id']}"
            coco_data["annotations"].append({
                "id": annotation_id_counter,
                "image_id": old_to_new_image_id[ann["image_id"]],
                "bbox": ann.get("bbox", [0, 0, 0, 0]),
                "attributes": attributes,
            })
            annotation_id_counter += 1

    return coco_data


def split_clips(data: dict, seed: int = TRAIN_SEED) -> dict[str, set[int]]:
    """Split by CLIP identity (BUGFIX #2) — ported from format.py::split_dataset.

    Returns a dict of {"train"|"val"|"test": set of image ids}; every image belonging to
    a given clip always lands in the same split — no clip/video ever appears in more
    than one split.
    """
    _, val_ratio, test_ratio = TRAIN_SPLIT_RATIOS
    clip_names = sorted({img["clip_name"] for img in data["images"]})

    try:
        train_clips, temp_clips = train_test_split(
            clip_names, test_size=(val_ratio + test_ratio), random_state=seed
        )
        val_ratio_adj = val_ratio / (val_ratio + test_ratio)
        val_clips, test_clips = train_test_split(
            temp_clips, test_size=(1 - val_ratio_adj), random_state=seed
        )
    except ValueError as exc:  # sklearn: "With n_samples=1, test_size=... the resulting train set will be empty"
        raise ValueError(
            f"El ZIP trae {len(clip_names)} clip(s)/vídeo(s) distintos y el split 70/15/15 se hace "
            "por clip: hacen falta al menos 4 (con menos, el 30 % reservado no se puede repartir "
            "entre validación y test)."
        ) from exc

    train_clips, val_clips, test_clips = set(train_clips), set(val_clips), set(test_clips)
    return {
        "train": {img["id"] for img in data["images"] if img["clip_name"] in train_clips},
        "val": {img["id"] for img in data["images"] if img["clip_name"] in val_clips},
        "test": {img["id"] for img in data["images"] if img["clip_name"] in test_clips},
    }


def build_clips(data: dict, image_ids: set[int], clip_length: int) -> list[dict]:
    """Group annotations by (globally-unique, post-BUGFIX-#3) track_id and window them
    into clips — ported from utils/data_utils.py's ``_organize_by_tracks`` + ``_create_clips``,
    restricted to one split's ``image_ids``.
    """
    img_by_id = {img["id"]: img for img in data["images"] if img["id"] in image_ids}
    tracks: dict[str, list[dict]] = {}
    for ann in data["annotations"]:
        img = img_by_id.get(ann["image_id"])
        if img is None:
            continue
        track_id = ann["attributes"]["track_id"]
        tracks.setdefault(track_id, []).append({
            "image_file": img["original_filename"],
            "clip_name": img["clip_name"],
            "bbox": ann["bbox"],
            "behavior": ann["attributes"]["behavior"],
            "image_id": ann["image_id"],
        })
    for detections in tracks.values():
        detections.sort(key=lambda d: d["image_id"])

    clips: list[dict] = []
    stride = max(1, clip_length // 2)
    for detections in tracks.values():
        for i in range(0, len(detections) - clip_length + 1, stride):
            window = detections[i:i + clip_length]
            behaviors = [d["behavior"] for d in window]
            majority = max(set(behaviors), key=behaviors.count)
            if majority not in TRAINING_BEHAVIOR_TO_IDX:
                continue
            clips.append({
                "clip_name": window[0]["clip_name"], "detections": window, "behavior": majority
            })
    return clips


class ClipClassificationDataset(Dataset):
    """Ported from utils/data_utils.py::BehaviorClassificationDataset with BUGFIX #4.

    Reads frames directly from ``raw_frames_dir/<clip_name>/<original_filename>`` (no
    split-copy step — format.py::copy_images_to_splits — since the plugin works off the
    extracted ZIP in place). ``is_train=False`` (validation/test) never applies
    ColorJitter; before the fix, both splits fell into the same default-transform branch.
    """

    def __init__(
        self, clips: list[dict], raw_frames_dir: str, clip_length: int, is_train: bool
    ) -> None:
        """Store the windowed clips and whether this split gets training augmentation."""
        self.clips = clips
        self.raw_frames_dir = raw_frames_dir
        self.clip_length = clip_length
        self.is_train = is_train
        self.transforms = None

    def __len__(self) -> int:
        """Return the number of windowed clips in this split."""
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # pylint: disable=too-many-locals
        """Load, crop and stack one clip's frames into a ``[C, T, H, W]`` tensor + label."""
        clip_info = self.clips[idx]
        frames: list[np.ndarray] = []
        for detection in clip_info["detections"]:
            img_path = os.path.join(
                self.raw_frames_dir, clip_info["clip_name"], detection["image_file"]
            )
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]
            x, y, w, h = (int(v) for v in detection["bbox"])
            if w <= 0 or h <= 0:
                continue
            x, y = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            if x2 <= x or y2 <= y:
                continue
            roi = cv2.resize(img[y:y2, x:x2], (CROP_SIZE, CROP_SIZE))
            frames.append(roi)

        if len(frames) < self.clip_length:
            while len(frames) < self.clip_length:
                pad = frames[-1] if frames else np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
                frames.append(pad)

        frames_np = np.array(frames)
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

        # BUGFIX #4: default transform depends on is_train — only the training split
        # gets ColorJitter; validation/test is deterministic (identity transform).
        if self.transforms is None:
            if self.is_train:
                self.transforms = transforms.Compose([
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
                ])
            else:
                self.transforms = transforms.Compose([])
        frames_t = self.transforms(frames_t)

        frames_t = frames_t.permute(1, 0, 2, 3)  # [C, T, H, W], as SlowFast expects
        label = TRAINING_BEHAVIOR_TO_IDX[clip_info["behavior"]]
        return frames_t, label


# pylint: disable-next=too-many-locals,too-many-arguments,too-many-positional-arguments,too-many-statements
def train_classifier(
    train_clips: list[dict],
    val_clips: list[dict],
    raw_frames_dir: str,
    clip_length: int,
    device: str,
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    mlflow_log_metrics: Callable[[dict, int], None] | None = None,
) -> tuple[nn.Module, dict]:
    """Run the fixed SlowFast training loop and return ``(trained_model, metrics)``.

    Real architecture/hyperparameters from the delivered train_classifier.py / memoria
    §6.4 (AdamW, FocalLoss(alpha=0.5, gamma=1.5), WeightedRandomSampler, warmup+cosine
    LR) — only BUGFIX #1 (scheduler stepped per batch, see inline comment below) changes
    behaviour versus the original. ``batch_size`` is clamped to the dataset size so this
    also works as a short wiring smoke test on a tiny real data slice.
    """
    if not train_clips or not val_clips:
        # Without validation clips val_acc would stay 0 and best_state would just be the last
        # epoch, silently. plugin.train() checks it too; this guards direct callers.
        raise ValueError(
            f"Hacen falta clips de entrenamiento y de validación (train={len(train_clips)}, "
            f"val={len(val_clips)})."
        )
    torch.manual_seed(TRAIN_SEED)

    train_ds = ClipClassificationDataset(train_clips, raw_frames_dir, clip_length, is_train=True)
    val_ds = ClipClassificationDataset(val_clips, raw_frames_dir, clip_length, is_train=False)

    behavior_counts = Counter(c["behavior"] for c in train_clips)
    total = len(train_clips)
    class_weight = {b: total / (len(behavior_counts) * n) for b, n in behavior_counts.items()}
    sample_weights = [class_weight[c["behavior"]] for c in train_clips]

    train_batch_size = max(1, min(batch_size, len(train_ds)))
    val_batch_size = max(1, min(batch_size, len(val_ds)))
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)

    model = build_classifier_module(num_classes=len(TRAINING_BEHAVIOR_TO_IDX), device=device)
    criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)

    warmup_steps = len(train_loader) * TRAIN_WARMUP_EPOCHS
    total_steps = len(train_loader) * epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        # The delivered train_classifier.py builds a tensor with 3.14159; a float with math.pi
        # is what LambdaLR expects (its own grid_search.py uses np.pi). Relative LR change ≤ 1e-6.
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    best_state = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = torch.as_tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(batched_slowfast_pathways(frames))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # BUGFIX #1: scheduler.step() once per BATCH — matches lr_lambda's
            # total_steps = len(train_loader) * epochs. Calling it once per epoch (the
            # original bug) left the LR stuck in the warmup ramp for the whole run.
            scheduler.step()

            tr_loss += loss.item()
            tr_correct += (outputs.argmax(1) == labels).sum().item()
            tr_total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = torch.as_tensor(labels).to(device)
                outputs = model(batched_slowfast_pathways(frames))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = 100.0 * tr_correct / max(tr_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        lr_now = float(scheduler.get_last_lr()[0])
        epoch_metrics = {
            "train_loss": tr_loss / max(len(train_loader), 1),
            "train_accuracy": train_acc,
            "val_loss": val_loss / max(len(val_loader), 1),
            "val_accuracy": val_acc,
            "lr": lr_now,
        }
        history.append({"epoch": epoch, **epoch_metrics})
        logger.info(
            "Epoch %d/%d — train_acc=%.2f val_acc=%.2f lr=%.6f",
            epoch, epochs, train_acc, val_acc, lr_now,
        )
        if mlflow_log_metrics is not None:
            mlflow_log_metrics(epoch_metrics, epoch)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
        "epochs_run": epochs,
        "best_val_acc": round(best_val_acc, 2),
        "lr_history": [round(h["lr"], 8) for h in history],
    }
    return model, metrics
