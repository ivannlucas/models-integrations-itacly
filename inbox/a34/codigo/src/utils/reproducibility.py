"""
Reproducibility control and device detection utilities.

Used across all notebooks to ensure deterministic results.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 1) -> None:
    """
    Fix all random seeds to ensure exact reproducibility.

    Parameters
    ----------
    seed : int
        Seed value for all random number generators (default: 1).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Detect and return the execution device (CUDA if available, otherwise CPU).

    Returns
    -------
    torch.device
        Selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
