"""Script para ejecutar el pipeline de procesamiento de datos."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import data_processing


def main() -> None:
    """Punto de entrada para procesamiento de datos."""
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de procesamiento de datos."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración (default: config/config.yaml)",
    )
    args = parser.parse_args()
    data_processing(config_path=args.config)


if __name__ == "__main__":
    main()
