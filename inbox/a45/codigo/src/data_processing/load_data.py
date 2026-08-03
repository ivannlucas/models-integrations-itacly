"""Carga y preparación inicial de datos."""

from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw_data(
    path: str,
    columns_to_drop: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Carga CSV, normaliza columnas a minúsculas y elimina columnas indicadas.

    Args:
        path: Ruta al archivo CSV.
        columns_to_drop: Lista de columnas a eliminar del DataFrame.

    Returns:
        DataFrame limpio con columnas en minúsculas.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    logger.info("Cargando datos desde %s", path)
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip().str.lower()

    if columns_to_drop:
        cols_to_drop = [c.lower() for c in columns_to_drop]
        existing = [c for c in cols_to_drop if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            logger.info("Columnas eliminadas: %s", existing)

    logger.info("Dataset cargado: %d filas x %d columnas", len(df), len(df.columns))
    return df
