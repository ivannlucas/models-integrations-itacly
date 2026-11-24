"""ml7 training checkpoints the YOLO layers on CPU, only while Ultralytics trains.

The checkpointing itself (bit-identical training) is tested in test_activation_checkpointing.py;
these tests pin when ml7 switches it on and off around Ultralytics' trainer.
"""
# pylint: disable=protected-access  # unit tests of the private training helper
from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace

import torch
from torch import nn

from app.plugins.ml7_cereals_grain_pest_detection import plugin as ml7


class _FakeYolo:
    """Records add_callback like ultralytics.YOLO and lets the test fire the events."""

    def __init__(self) -> None:
        self.callbacks: dict[str, list] = {}

    def add_callback(self, event: str, func) -> None:
        """Register ``func`` for ``event``."""
        self.callbacks.setdefault(event, []).append(func)

    def fire(self, event: str, trainer) -> None:
        """Run the callbacks of ``event``, as the trainer does."""
        for func in self.callbacks.get(event, []):
            func(trainer)


def _trainer() -> SimpleNamespace:
    layers = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.SiLU())
    return SimpleNamespace(model=SimpleNamespace(model=layers))


def _shadowed(trainer) -> list[bool]:
    return ["forward" in vars(layer) for layer in trainer.model.model]


def test_cpu_training_checkpoints_layers_between_start_and_end():
    """Layers are checkpointed from on_train_start (after the EMA copy) to on_train_end."""
    yolo, trainer = _FakeYolo(), _trainer()
    with ExitStack() as stack:
        ml7._checkpoint_layers_on_cpu(yolo, torch.device("cpu"), stack)
        assert _shadowed(trainer) == [False] * 3  # the trainer copies the model before start
        yolo.fire("on_train_start", trainer)
        assert _shadowed(trainer) == [True] * 3
        yolo.fire("on_train_end", trainer)
        assert _shadowed(trainer) == [False] * 3


def test_training_that_fails_mid_way_restores_the_layers():
    """Without on_train_end (Ultralytics raised), closing the stack undoes the checkpointing."""
    yolo, trainer = _FakeYolo(), _trainer()
    with ExitStack() as stack:
        ml7._checkpoint_layers_on_cpu(yolo, torch.device("cpu"), stack)
        yolo.fire("on_train_start", trainer)
    assert _shadowed(trainer) == [False] * 3


def test_cuda_training_is_left_as_is():
    """On CUDA activations are not in the pod's RAM: no callback is registered."""
    yolo = _FakeYolo()
    with ExitStack() as stack:
        ml7._checkpoint_layers_on_cpu(yolo, torch.device("cuda"), stack)
    assert not yolo.callbacks
