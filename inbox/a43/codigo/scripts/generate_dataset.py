"""Script para ejecutar la generación del dataset sintético (entrenamiento y capa XAI)."""

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
        description="Genera el dataset sintético de hornos industriales."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración (default: config/config.yaml)",
    )

    parser.add_argument(
        "--xai",
        action="store_true",
        help="Activa la generación de datasets vinculados a la capa de explicabilidad XAI.",
    )
    args = parser.parse_args()

    try:
        if args.xai:
            logger.info("Generando dataset de entrenamiento...")
            generate_dataset(config_path=args.config)
            logger.info("Generando todos los datasets necesarios para la capa de explicabilidad XAI...")
            generate_dataset_for_xai(config_path=args.config)
        else:
            logger.info("Generando dataset de entrenamiento...")
            generate_dataset(config_path=args.config)

            return 0
    except Exception as e:
        logger.error(f"Error durante la generación del dataset: {e}")
        return 1


if __name__ == "__main__":
    main()