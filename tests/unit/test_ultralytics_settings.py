"""configure_ultralytics(): no MLflow callback, telemetry or weights download inside the repo.

Ultralytics' defaults made each ml7 /train create an extra MLflow experiment (named after the
temp project dir) and download yolo26n.pt into <repo>/weights; its telemetry ran on every
train/val/predict of ml1, ml7 and ml10.
"""
import tempfile
from pathlib import Path

import ultralytics.utils as yolo_utils
from ultralytics.utils.events import events

from app.domain.services.ultralytics_settings import (
    configure_ultralytics,
    ultralytics_weights_dir,
)


def test_disables_integrations_and_telemetry():
    """The MLflow callback, the other integrations and the telemetry end up off."""
    configure_ultralytics()
    for key in ("mlflow", "sync", "tensorboard", "wandb", "clearml", "comet", "dvc", "raytune"):
        assert yolo_utils.SETTINGS[key] is False, key
    assert events.enabled is False


def test_weights_dir_outside_the_repo():
    """The AMP check reads WEIGHTS_DIR at call time; it must point to the temp dir."""
    configure_ultralytics()
    weights_dir = ultralytics_weights_dir()
    assert weights_dir.is_relative_to(Path(tempfile.gettempdir()))
    assert yolo_utils.WEIGHTS_DIR == weights_dir
    assert Path(yolo_utils.SETTINGS["weights_dir"]) == weights_dir


def test_does_not_write_the_users_settings_file():
    """The overrides stay in memory: the settings.json of whoever runs the service is untouched."""
    settings_file = Path(yolo_utils.SETTINGS.file)
    before = settings_file.read_bytes() if settings_file.exists() else None
    configure_ultralytics()
    after = settings_file.read_bytes() if settings_file.exists() else None
    assert after == before
