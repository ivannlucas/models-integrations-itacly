"""Training stage exports."""

from pathlib import Path

from src.reproducibility.runtime import ensure_optional_dependency, repo_root

ensure_optional_dependency("sklearn", repo_root_path=repo_root())

from .pipeline import run_training
from .policy_simulation import run_policy_simulation
from .reproducibility import run_reproducibility_training
from .train_purchase_trigger import run_purchase_trigger_training
from .train_quantity_optimizer import run_quantity_optimizer_training
from .train_upstream_predictor import run_upstream_training

__all__ = [
    "run_policy_simulation",
    "run_purchase_trigger_training",
    "run_quantity_optimizer_training",
    "run_reproducibility_training",
    "run_training",
    "run_upstream_training",
]
