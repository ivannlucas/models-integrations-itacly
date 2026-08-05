"""Script para generar estadísticas y reportes."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import get_stats


def main() -> None:
    """Punto de entrada para estadísticas."""
    parser = argparse.ArgumentParser(
        description="Genera estadísticas y métricas de evaluación."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración (default: config/config.yaml)",
    )
    args = parser.parse_args()
    get_stats(config_path=args.config)


if __name__ == "__main__":
    main()
