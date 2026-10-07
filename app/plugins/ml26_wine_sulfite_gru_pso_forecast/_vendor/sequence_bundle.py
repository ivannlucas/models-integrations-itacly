"""SequenceBundle + restricted unpickler for the AI team's gru_pso.pkl.

gru_pso.pkl is a pickle of the dataclass ``src.training.models.train_sequence.SequenceBundle``
(the AI team's module path). We vendor the dataclass (same field names) and load the pickle with
an allow-list Unpickler: only the handful of globals the file actually references are resolved
(checked with pickletools on the delivered artifact), everything else raises. Tensors stored via
``torch.storage._load_from_bytes`` are re-read with ``torch.load(weights_only=True)`` on CPU.
"""

from __future__ import annotations

import collections
import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ORIGINAL_BUNDLE_MODULE = "src.training.models.train_sequence"


@dataclass
class SequenceBundle:  # pylint: disable=too-many-instance-attributes
    """Serialized sequence model: weights + feature/target contract + z-score statistics +
    config.
    """

    model_name: str
    model_state: dict[str, Any]
    feature_names: list[str]
    target_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    config: dict[str, Any]


def _load_storage_from_bytes(payload: bytes) -> Any:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


_NUMPY_ALLOWED = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
}


class RestrictedBundleUnpickler(pickle.Unpickler):
    """Unpickler that only resolves the globals present in the delivered gru_pso.pkl."""

    def find_class(self, module: str, name: str) -> Any:
        """Resolve an allow-listed global or refuse."""
        if (module, name) == (ORIGINAL_BUNDLE_MODULE, "SequenceBundle"):
            return SequenceBundle
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _load_storage_from_bytes
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return torch._utils._rebuild_tensor_v2  # pylint: disable=protected-access
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) in _NUMPY_ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Global not allowed in a26 model bundle: {module}.{name}")


def load_sequence_bundle(path: str | Path) -> SequenceBundle:
    """Load gru_pso.pkl (AI team format) into a vendored SequenceBundle on CPU."""
    with Path(path).open("rb") as handle:
        bundle = RestrictedBundleUnpickler(handle).load()
    if not isinstance(bundle, SequenceBundle):
        raise TypeError(f"Unexpected object in model bundle: {type(bundle).__name__}")
    return bundle
