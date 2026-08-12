"""Script para ejecutar inferencia con XAI integrado."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import predict
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Punto de entrada único para inferencia."""
    parser = argparse.ArgumentParser(
        description="Ejecuta inferencia con el modelo entrenado y explicabilidad XAI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    # Inferencia con configuración por defecto
  python scripts/predict.py

    # Inferencia con archivo específico
  python scripts/predict.py --input data/input/datos.csv
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración YAML (default: config/config.yaml)",
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Ruta al archivo CSV de entrada (sobreescribe config).",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directorio para guardar predicciones y reportes XAI.",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar mensajes de debug detallados",
    )

    args = parser.parse_args()

    try:
        logger.info("Ejecutando inferencia...")
        logger.info("Config: %s", args.config)
        logger.info("Input: %s", args.input or "default from config")
        logger.info("Output: %s", args.output or "default (data/predictions/)")

        results = predict(
            config_path=args.config,
            input_path=args.input,
            output_dir=args.output,
        )

        return 0

    except Exception as e:
        logger.error("Error durante predicción: %s", str(e), exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
