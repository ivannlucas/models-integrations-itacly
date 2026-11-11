"""Process-wide Ultralytics settings for the plugins that use YOLO (ml1, ml7, ml10).

Ultralytics reads its settings from a per-user ``settings.json`` whose defaults suit a laptop,
not this service:

- ``mlflow``: its MLflow callback logs every ``model.train()`` to ``MLFLOW_TRACKING_URI`` as an
  extra experiment named after ``project`` (a temp dir here) and uploads best.pt/last.pt to it —
  junk next to the run that ``train()`` persists. Without ``MLFLOW_TRACKING_URI`` it writes to
  ``runs/mlflow`` and switches the process-wide tracking URI.
- ``sync``: sends usage telemetry on every train/val/predict call.
- the other integrations (tensorboard, comet, clearml…), for the same reasons.
- ``weights_dir``: where the AMP check of ``model.train()`` downloads ``yolo26n.pt``. It defaults
  to ``<git root or cwd>/weights`` — inside the repo locally, ``/app/weights`` in the container.

The overrides live in memory only: ``SETTINGS.update()`` would persist them to the
``settings.json`` of whoever runs the service. ``events.enabled`` and ``WEIGHTS_DIR`` are computed
once at import, so they are patched too. ``train()``/``val()`` always pass ``project=`` to a temp
dir, so ``runs_dir`` is never used. Idempotent; call it before any YOLO use.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

_INTEGRATIONS_OFF = (
    "sync", "mlflow", "tensorboard", "wandb", "clearml", "comet", "dvc", "raytune",
)


def ultralytics_weights_dir() -> Path:
    """Directory for the weights Ultralytics downloads by itself (AMP check), outside the repo."""
    return Path(tempfile.gettempdir()) / "ultralytics" / "weights"


def configure_ultralytics() -> None:
    """Disable Ultralytics' integrations and telemetry and move its weights dir out of the repo."""
    # pylint: disable=import-outside-toplevel  # heavy, and only the YOLO plugins need it
    import ultralytics.utils as yolo_utils
    from ultralytics.utils.events import events

    weights_dir = ultralytics_weights_dir()
    overrides = {key: False for key in _INTEGRATIONS_OFF if key in yolo_utils.SETTINGS}
    overrides["weights_dir"] = str(weights_dir)
    # dict.update, not SETTINGS.update(): the latter writes the user's settings.json.
    dict.update(yolo_utils.SETTINGS, overrides)
    yolo_utils.WEIGHTS_DIR = weights_dir
    events.enabled = False
