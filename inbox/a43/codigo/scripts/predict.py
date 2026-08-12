"""Script para ejecutar predicciones con el modelo entrenado.

Por defecto ejecuta predicción estándar.
Con --xai activa el flujo de explicabilidad completo.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import predict
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Punto de entrada para inferencia con XAI integrado."""
    parser = argparse.ArgumentParser(
        description="Ejecuta predicciones con el modelo entrenado + explicabilidad XAI automática.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    # Predicción estándar (default)
    python scripts/predict.py

    # Predicción estándar con archivo específico
    python scripts/predict.py --input data/input/datos.csv

    # Predicción + XAI (parámetros leídos de config.yaml)
    python scripts/predict.py --xai
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
        "--xai",
        action="store_true",
        help="Activa la explicabilidad XAI (predicción + explicaciones)",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar mensajes de debug detallados",
    )

    args = parser.parse_args()

    try:
        if not args.xai:
            # Solo predicción, sin explicabilidad
            logger.info("Ejecutando predicción (sin XAI)...")
            logger.info("=" * 30)
            predict(
                config_path=args.config,
                input_path=args.input,
                enable_xai=False,
            )
            return 0
        else:
            # Predicción + XAI integrado (flujo unificado)
            logger.info("Ejecutando predicción + explicabilidad XAI (flujo integrado)...")
            logger.info("=" * 30)

            results = predict(
                config_path=args.config,
                input_path=args.input,
                enable_xai=True,
                output_dir=args.output,
            )
            return 0

    except Exception as e:
        logger.error("✗ Error durante predicción: %s", str(e), exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
