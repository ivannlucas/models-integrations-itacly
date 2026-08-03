"""Script para ejecutar la generación del dataset sintético."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import generate_dataset, generate_dataset_for_xai
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Punto de entrada para generación de dataset."""
    parser = argparse.ArgumentParser(
        description="Genera el dataset sintético de secadoras industriales."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración (default: config/config.yaml)",
    )
    args = parser.parse_args()
    logger.info("Generando dataset de entrenamiento...")
    generate_dataset(config_path=args.config)
    logger.info("Generando todos los datasets necesarios para la capa de explicabilidad XAI...")
    generate_dataset_for_xai(config_path=args.config)


if __name__ == "__main__":
    main()
