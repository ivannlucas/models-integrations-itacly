import pandas as pd
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger("LOAD_DATA")

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Carga el dataset raw, valida su existencia y asegura el orden 
    cronológico por ciclo (run_id).
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"Archivo no encontrado en la ruta: {file_path}")
        raise FileNotFoundError(f"No se pudo encontrar {file_path}")

    try:
        logger.info(f"Cargando dataset desde: {file_path}...")
        df = pd.read_csv(path)
        
        # Validación mínima de columnas requeridas por ambos sistemas
        base_cols = ['run_id', 'time_min']
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            logger.warning(f"Faltan columnas de control temporal: {missing}")
        
        # Crucial para los lags: ordenar por run y luego por tiempo
        if 'run_id' in df.columns and 'time_min' in df.columns:
            df = df.sort_values(['run_id', 'time_min']).reset_index(drop=True)
            logger.info("Dataset ordenado por 'run_id' y 'time_min'.")
            
        logger.info(f"Carga exitosa. Dimensiones: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error crítico al leer el CSV: {e}")
        raise

def get_data_by_run(df: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """
    Extrae un ciclo de funcionamiento específico (Run) para análisis individual.
    """
    run_data = df[df['run_id'] == run_id]
    if run_data.empty:
        logger.warning(f"El run_id {run_id} no existe en el dataset.")
    return run_data

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Guarda los datos tras el feature engineering en la carpeta de procesados.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Datos procesados guardados en: {output_path}")