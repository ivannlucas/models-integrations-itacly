"""Device selection shared by the PyTorch plugins."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def safe_torch_device() -> torch.device:
    """Return CUDA only if a conv actually runs on it (module AND tensor on the GPU), else CPU.

    ``torch.cuda.is_available()`` is not enough: on a GPU the installed build does not support,
    or with a broken driver, it is True and the first real op fails. The probe moves the module
    too — with only the tensor on CUDA the probe always failed and 7 plugins ran on CPU.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.nn.Conv2d(1, 1, 1).cuda()(torch.zeros(1, 1, 4, 4).cuda())
        return torch.device("cuda")
    except Exception:  # pylint: disable=broad-exception-caught  # any CUDA failure → CPU
        logger.warning("CUDA detectada pero no funcional — usando CPU.")
        return torch.device("cpu")
