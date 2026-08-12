"""Configuración de logging para el proyecto."""

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Crea logger con formato estándar para consola.

    Args:
        name: Nombre del logger (normalmente __name__).
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))

        formatter = logging.Formatter(
            "%(asctime)s. %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
