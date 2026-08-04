"""Logging helpers for stage-based execution."""

from __future__ import annotations

import logging
from pathlib import Path

from .project import ensure_directory


def setup_stage_logger(stage_name: str, logs_dir: str | Path, run_id: str) -> tuple[logging.Logger, Path]:
    """Create a logger with stream and file handlers for a pipeline stage."""
    log_directory = ensure_directory(Path(logs_dir))
    log_path = log_directory / f"{stage_name}_{run_id}.log"
    logger_name = f"pipeline.{stage_name}.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger, log_path
