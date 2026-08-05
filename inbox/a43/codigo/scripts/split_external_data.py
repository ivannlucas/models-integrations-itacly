"""Script para dividir un dataset externo en splits train/val/test.

Carga un CSV externo (por ejemplo, datos reales del cliente),
lo divide en train/val/test siguiendo las proporciones definidas
en config/config.yaml (data_processing.external_data_split) y guarda los
splits en data/splits/ listos para el pipeline de entrenamiento.

Ejemplo:
    python scripts/split_external_data.py --input data/raw/mi_dataset.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import split_external_data
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Punto de entrada para dividir dataset externo en splits."""
    parser = argparse.ArgumentParser(
        description="Divide un dataset externo en train/val/test para entrenamiento.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
El dataset externo debe contener al menos:
  - timestamp: para ordenar cronológicamente.
  - Las 13 columnas de sensores (ver data_generation.sensors en config.yaml).
  - Una columna de etiquetas de fallo (fault_name por defecto).

Columna opcional:
  - cycle_id: para agrupar ventanas por ciclo y evitar mezclar datos de distintos ciclos.

Ejemplo:
    python scripts/split_external_data.py --input data/raw/mi_dataset.csv
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuracion YAML (default: config/config.yaml)",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al archivo CSV externo a dividir.",
    )

    args = parser.parse_args()

    try:
        split_external_data(
            input_path=args.input,
            config_path=args.config,
        )
        logger.info(
            "Siguientes pasos: ejecuta 'python scripts/data_processing.py' "
            "y luego 'python scripts/train.py'."
        )
        return 0

    except Exception as e:
        logger.error("Error al dividir dataset externo: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
