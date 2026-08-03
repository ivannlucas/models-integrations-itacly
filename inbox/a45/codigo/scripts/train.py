"""Script para ejecutar el entrenamiento del modelo."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import train


def main() -> None:
    """Punto de entrada para entrenamiento."""
    parser = argparse.ArgumentParser(
        description="Entrena el modelo Deep Neuro-Fuzzy."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración (default: config/config.yaml)",
    )
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
