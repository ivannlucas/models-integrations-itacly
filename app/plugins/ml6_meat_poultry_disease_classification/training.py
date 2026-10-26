"""Retraining of one ensemble — ported from the delivered training code.

- Split: src/utils/io.py::ensure_data_splits (every class folder in name order, files in name
  order, one ``random.Random(42)`` shuffling each folder in turn, ``int(n*0.7)`` train,
  ``int(n*0.15)`` val, the rest test). Computed in memory instead of copying files.
- Sample order: torchvision ImageFolder (folders by name, then files by name), which is what
  the WeightedRandomSampler indexes.
- Loop: src/training/trainer.py::train_model — Adam(lr, weight_decay) over all parameters (only
  the head has gradients), CrossEntropyLoss, StepLR(7, 0.1) stepped per epoch, early stopping on
  val loss (patience 10, min_delta 1e-4), best checkpoint by val accuracy (strictly greater).
- RNG: the original seeds the global RNG once (set_seed) and trains the three models in a row;
  the sampler and Dropout draw from it. Here the same sequence runs inside
  ``torch.random.fork_rng`` so the shared server's global RNG is left untouched.

Divergence (documented in manifest.training): the original starts from ImageNet weights with a
fresh head; this starts from the served checkpoint (the repo rule: fine-tuning always starts from
the base model). The frozen backbone / trainable head split is the same.
"""
from __future__ import annotations

import copy
import logging
import os
import random
from typing import Callable

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from app.plugins.ml6_meat_poultry_disease_classification.constants import (
    CLASS_NAMES,
    FOLDER_MAP,
    IMAGE_EXTENSIONS,
    TRAIN_EPOCHS,
    TRAIN_GAMMA,
    TRAIN_MIN_DELTA,
    TRAIN_PATIENCE,
    TRAIN_SEED,
    TRAIN_SPLIT,
    TRAIN_STEP_SIZE,
)
from app.plugins.ml6_meat_poultry_disease_classification.postprocessing import majority_vote
from app.plugins.ml6_meat_poultry_disease_classification.preprocessing import (
    get_transform,
    load_image,
)

logger = logging.getLogger(__name__)

Sample = tuple[str, int]


def class_index_of_folder(folder_name: str, task: str) -> int | None:
    """Class index of a folder: its logical name or its raw-dataset name, case-insensitive."""
    name = folder_name.lower()
    for idx, logical in enumerate(CLASS_NAMES[task]):
        if name in (logical, FOLDER_MAP[task][logical].lower()):
            return idx
    return None


def _images_in(folder: str) -> list[str]:
    paths = (os.path.join(folder, f) for f in os.listdir(folder))
    return sorted(
        p for p in paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )


def split_dataset(  # pylint: disable=too-many-locals
    data_root: str, task: str,
) -> dict[str, list[Sample]]:
    """Split a raw dataset (one sub-folder per class) exactly like ensure_data_splits.

    Raises ValueError when a class is missing or too small to give every split one image.
    """
    folders = sorted(
        (d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))),
    )
    indexed = ((class_index_of_folder(d, task), d) for d in folders)
    found = {idx: d for idx, d in indexed if idx is not None}
    missing = [CLASS_NAMES[task][i] for i in range(len(CLASS_NAMES[task])) if i not in found]
    if missing:
        expected = [FOLDER_MAP[task][c] for c in CLASS_NAMES[task]]
        raise ValueError(
            f"data_path debe contener una subcarpeta por clase del ensemble '{task}' ({expected}); "
            f"faltan: {missing}. Encontradas: {folders}"
        )

    rng = random.Random(TRAIN_SEED)
    total = sum(TRAIN_SPLIT.values())
    train_p, val_p = TRAIN_SPLIT["train"] / total, TRAIN_SPLIT["val"] / total
    per_folder: dict[str, dict[str, list[str]]] = {}
    # Every sub-folder consumes the RNG in name order, as in the original (even unknown ones).
    for folder in folders:
        images = _images_in(os.path.join(data_root, folder))
        rng.shuffle(images)
        n_train, n_val = int(len(images) * train_p), int(len(images) * val_p)
        per_folder[folder] = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

    for idx, folder in found.items():
        counts = {part: len(files) for part, files in per_folder[folder].items()}
        if min(counts.values()) < 1:
            raise ValueError(
                f"La clase '{CLASS_NAMES[task][idx]}' ({folder}) tiene muy pocas imágenes para el "
                f"split 70/15/15 (train/val/test = {counts}); hacen falta al menos 7 por clase."
            )

    # ImageFolder order: class folders by name, then files by name.
    splits: dict[str, list[Sample]] = {}
    for part in ("train", "val", "test"):
        splits[part] = [
            (path, idx)
            for idx, folder in sorted(found.items(), key=lambda kv: kv[1])
            for path in sorted(per_folder[folder][part], key=os.path.basename)
        ]
    return splits


class ImageListDataset(Dataset):
    """(path, label) pairs with the ensemble's (train = test) transform."""

    def __init__(self, samples: list[Sample], task: str) -> None:
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = get_transform(task)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return self.transform(load_image(path)), label


def make_loaders(
    splits: dict[str, list[Sample]], task: str, batch_size: int,
) -> dict[str, DataLoader]:
    """load_data.py::get_dataloaders — weighted sampler for train, ordered val/test."""
    datasets = {part: ImageListDataset(samples, task) for part, samples in splits.items()}
    targets = torch.tensor(datasets["train"].targets)
    class_counts = torch.bincount(targets, minlength=len(CLASS_NAMES[task])).double()
    sample_weights = (1.0 / class_counts)[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


def train_model(  # pylint: disable=too-many-locals,too-many-statements
    model: nn.Module, loaders: dict[str, DataLoader], hyperparams: dict, device: torch.device,
    log_epoch: Callable[[int, dict], None] | None = None,
) -> tuple[nn.Module, float, int]:
    """trainer.py::train_model on an already-built model.

    Returns ``(best model, best val accuracy, epochs run)``.
    """
    sizes = {part: len(loader.dataset) for part, loader in loaders.items()}
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"],
                           weight_decay=hyperparams["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=TRAIN_STEP_SIZE, gamma=TRAIN_GAMMA)

    best_state = copy.deepcopy(model.state_dict())
    best_acc, best_val_loss = 0.0, float("inf")
    epochs_no_improve, early_stop, epochs_run = 0, False, 0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        if early_stop:
            break
        epochs_run = epoch
        epoch_log: dict[str, float] = {}
        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()  # pylint: disable=expression-not-assigned
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += int(torch.sum(preds == labels.data))
            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects / sizes[phase]
            epoch_log[f"{phase}_loss"], epoch_log[f"{phase}_acc"] = epoch_loss, epoch_acc
            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_state = copy.deepcopy(model.state_dict())
                if best_val_loss - epoch_loss > TRAIN_MIN_DELTA:
                    best_val_loss, epochs_no_improve = epoch_loss, 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= TRAIN_PATIENCE:
                        early_stop = True
        if log_epoch:
            log_epoch(epoch, epoch_log)
    model.load_state_dict(best_state)
    model.eval()
    return model, float(best_acc), epochs_run


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> list[int]:
    """Class index per sample, in loader order (predictor.py::run_inference)."""
    preds: list[int] = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            output = model(inputs.to(device))
            preds.extend(int(p) for p in torch.max(torch.exp(output), 1)[1].cpu())
    return preds


def ensemble_predictions(per_model: dict[str, list[int]]) -> list[int]:
    """Majority vote per sample across the base models (dict order = config order)."""
    return [majority_vote(list(votes)) for votes in zip(*per_model.values())]
