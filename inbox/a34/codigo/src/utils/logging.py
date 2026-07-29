"""
Centralized logging utility for DATAGIA.

Provides a configured logger with consistent formatting across all modules.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger instance.

    Uses a StreamHandler to stdout with a format that includes timestamp,
    module name and log level. If the logger already has handlers (e.g.
    from a previous call), it is returned as-is to avoid duplicate output.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    level : int
        Logging level (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger ready to use.

    Example
    -------
    >>> from src.utils.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started")
    2026-03-02 10:00:00 | INFO | src.training.trainer | Training started
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
