"""Script para dividir un dataset externo en splits train/val/test.

Carga un CSV externo (por ejemplo, datos reales del cliente),
lo divide en train/val/test siguiendo las proporciones definidas
en config/config.yaml (data_processing.external_data_split) y guarda los
splits en data/processed/ listos para el pipeline de entrenamiento.

Ejemplo:
    python scripts/split_external_data.py --input data/raw/mi_dataset.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data_processing.load_data import load_raw_data
from src.data_processing.preprocess import split_train_val_test_by_id
from src.utils.common import ensure_dir, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Punto de entrada para dividir dataset externo en splits."""
    parser = argparse.ArgumentParser(
        description="Divide un dataset externo en train/val/test para entrenamiento.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
El dataset externo debe contener al menos:
  - Las 11 columnas de sensores (ver data_generation.sensors en config.yaml).
  - Una columna de etiquetas de fallo (fault_name por defecto).

Columnas opcionales:
  - cycle_id: para split por ciclo (recomendado si los datos tienen ciclos).
  - timestamp: para ordenar cronologicamente.

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
        config = load_config(args.config)
        data_cfg = config["data_processing"]
        gen_cfg = config.get("data_generation", {})
        paths = config["paths"]

        # 1. Cargar datos
        if not os.path.exists(args.input):
            logger.error("Archivo no encontrado: %s", args.input)
            return 1

        df = load_raw_data(args.input)
        df.columns = df.columns.astype(str).str.strip().str.lower()

        # 2. Validar columnas minimas
        target_column = data_cfg.get("target_column", "fault_name").lower()
        id_column = data_cfg.get("id_column", "cycle_id")
        id_column = id_column.lower() if id_column else None
        normal_tokens = data_cfg.get("normal_tokens", [])
        expected_sensors = [
            s["name"].lower() for s in gen_cfg.get("sensors", [])
        ]

        errors: list[str] = []

        # Columna objetivo
        if target_column not in df.columns:
            errors.append(
                f"Falta la columna objetivo '{target_column}'. "
                f"Columnas disponibles: {sorted(df.columns.tolist())}."
            )

        # Sensores
        missing_sensors = [s for s in expected_sensors if s not in df.columns]
        if missing_sensors:
            errors.append(
                f"Columnas de sensores faltantes: {missing_sensors}."
            )

        if errors:
            for err in errors:
                logger.error(err)
            return 1

        # 3. Calcular proporciones de split desde external_data_split
        split_cfg = data_cfg.get("external_data_split", {})
        train_pct = float(split_cfg.get("train_pct", 66.7))
        val_pct = float(split_cfg.get("val_pct", 13.3))
        test_pct = float(split_cfg.get("test_pct", 20.0))
        total_pct = train_pct + val_pct + test_pct

        if abs(total_pct - 100.0) > 0.01:
            logger.error(
                "train_pct + val_pct + test_pct debe sumar 100%%. "
                "Actual: %.1f + %.1f + %.1f = %.1f",
                train_pct, val_pct, test_pct, total_pct,
            )
            return 1

        val_size = val_pct / 100.0
        test_size = test_pct / 100.0

        logger.info(
            "Proporciones de split (desde external_data_split): "
            "train=%.1f%%, val=%.1f%%, test=%.1f%%",
            train_pct, val_pct, test_pct,
        )

        # 4. Ejecutar split
        split_data, _ = split_train_val_test_by_id(
            df,
            id_col=id_column if id_column else "cycle_id",
            target_column=target_column,
            val_size=val_size,
            test_size=test_size,
            normal_tokens=normal_tokens,
        )

        # 5. Guardar splits en data/splits/
        splits_dir = ensure_dir(paths["splits"])

        for split_name in ["train", "val", "test"]:
            df_split = split_data[split_name]
            split_path = os.path.join(splits_dir, f"{split_name}.csv")
            df_split.to_csv(split_path, index=False)
            logger.info(
                "Split '%s' guardado: %d filas -> %s",
                split_name, len(df_split), split_path,
            )

        logger.info(
            "Dataset externo dividido correctamente: %d filas totales "
            "-> train=%d, val=%d, test=%d",
            len(df),
            len(split_data["train"]),
            len(split_data["val"]),
            len(split_data["test"]),
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
