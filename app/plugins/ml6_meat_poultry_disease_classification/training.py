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
from dataclasses import dataclass
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


def _class_folders(data_root: str, task: str) -> tuple[list[str], dict[int, str]]:
    """Sub-folders of ``data_root`` (sorted) and ``{class index: folder}`` of the task's classes.

    Raises ValueError when a class folder is missing.
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
    return folders, found


def _check_split_sizes(
    found: dict[int, str], per_folder: dict[str, dict[str, list[str]]], task: str,
) -> None:
    """Raise ValueError when a class leaves some split (train/val/test) without images."""
    for idx, folder in found.items():
        counts = {part: len(files) for part, files in per_folder[folder].items()}
        if min(counts.values()) < 1:
            raise ValueError(
                f"La clase '{CLASS_NAMES[task][idx]}' ({folder}) tiene muy pocas imágenes para el "
                f"split 70/15/15 (train/val/test = {counts}); hacen falta al menos 7 por clase."
            )


def split_dataset(data_root: str, task: str) -> dict[str, list[Sample]]:
    """Split a raw dataset (one sub-folder per class) exactly like ensure_data_splits.

    Raises ValueError when a class is missing or too small to give every split one image.
    """
    folders, found = _class_folders(data_root, task)

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

    _check_split_sizes(found, per_folder, task)

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


def _run_epoch_phase(model: nn.Module, loader: DataLoader, phase: str, step: tuple) -> tuple:
    """One train/val pass of trainer.py::train_model: ``(epoch loss, epoch accuracy)``.

    ``step`` is ``(optimizer, criterion, device)``; weights are only updated in "train".
    """
    optimizer, criterion, device = step
    model.train(phase == "train")
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in loader:
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
    size = len(loader.dataset)
    return running_loss / size, running_corrects / size


@dataclass
class _EarlyStopping:
    """Best checkpoint by val accuracy (strictly greater) + early stop on val loss."""

    best_state: dict
    best_acc: float = 0.0
    best_val_loss: float = float("inf")
    epochs_no_improve: int = 0
    stop: bool = False

    def update(self, model: nn.Module, epoch_acc: float, epoch_loss: float) -> None:
        """Account for one validation pass."""
        if epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            self.best_state = copy.deepcopy(model.state_dict())
        if self.best_val_loss - epoch_loss > TRAIN_MIN_DELTA:
            self.best_val_loss, self.epochs_no_improve = epoch_loss, 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= TRAIN_PATIENCE:
                self.stop = True


def train_model(
    model: nn.Module, loaders: dict[str, DataLoader], hyperparams: dict, device: torch.device,
    log_epoch: Callable[[int, dict], None] | None = None,
) -> tuple[nn.Module, float, int]:
    """trainer.py::train_model on an already-built model.

    Returns ``(best model, best val accuracy, epochs run)``.
    """
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"],
                           weight_decay=hyperparams["weight_decay"])
    step = (optimizer, nn.CrossEntropyLoss(), device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=TRAIN_STEP_SIZE, gamma=TRAIN_GAMMA)

    tracker = _EarlyStopping(best_state=copy.deepcopy(model.state_dict()))
    epochs_run = 0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        if tracker.stop:
            break
        epochs_run = epoch
        epoch_log: dict[str, float] = {}
        for phase in ("train", "val"):
            epoch_loss, epoch_acc = _run_epoch_phase(model, loaders[phase], phase, step)
            if phase == "train":
                scheduler.step()
            epoch_log[f"{phase}_loss"], epoch_log[f"{phase}_acc"] = epoch_loss, epoch_acc
            if phase == "val":
                tracker.update(model, epoch_acc, epoch_loss)
        if log_epoch:
            log_epoch(epoch, epoch_log)
    model.load_state_dict(tracker.best_state)
    model.eval()
    return model, float(tracker.best_acc), epochs_run


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
