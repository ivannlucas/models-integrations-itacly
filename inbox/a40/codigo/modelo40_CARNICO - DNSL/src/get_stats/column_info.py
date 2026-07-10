import pandas as pd
import numpy as np
from src.utils.logging import get_logger

logger = get_logger("COLUMN_INFO")

def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un informe detallado sobre la calidad y el tipo de cada 
    columna en el dataset procesado.
    """
    logger.info("Generando resumen de columnas...")
    
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'nulls': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df)) * 100,
        'unique_values': df.nunique(),
        'mean': df.select_dtypes(include=[np.number]).mean(),
        'std': df.select_dtypes(include=[np.number]).std(),
        'min': df.select_dtypes(include=[np.number]).min(),
        'max': df.select_dtypes(include=[np.number]).max()
    })
    
    # Identificar columnas con varianza cero (constantes) que no aportan información
    const_cols = summary[summary['unique_values'] == 1].index.tolist()
    if const_cols:
        logger.warning(f"Columnas constantes detectadas (sin valor predictivo): {const_cols}")
        
    return summary

def get_feature_groups(df: pd.DataFrame):
    """
    Clasifica las columnas por su rol funcional en el sistema.
    Útil para auditorías rápidas de ingeniería.
    """
    groups = {
        'metadata': ['run_id', 'time_min'],
        'targets': [col for col in df.columns if 'fault' in col or 'target' in col],
        'sensors_raw': [col for col in df.columns if '_cab' in col or '_amb' in col or '_bar' in col],
        'engineered_features': [col for col in df.columns if '_lag' in col or '_delta' in col or '_Index' in col]
    }
    
    for group, cols in groups.items():
        existing_cols = [c for c in cols if c in df.columns]
        logger.info(f"Grupo '{group}': {len(existing_cols)} columnas encontradas.")
        
    return groups

def check_data_leakage(df: pd.DataFrame, target_col: str):
    """
    Verifica si hay columnas con una correlación sospechosamente 
    perfecta con el target (Data Leakage).
    """
    if target_col not in df.columns:
        logger.error(f"Target {target_col} no encontrado para check de leakage.")
        return
        
    corr = df.select_dtypes(include=[np.number]).corr()[target_col].abs().sort_values(ascending=False)
    leakage_suspects = corr[(corr > 0.99) & (corr.index != target_col)].index.tolist()
    
    if leakage_suspects:
        logger.warning(f"⚠️ POSIBLE DATA LEAKAGE: {leakage_suspects} tienen correlación > 0.99 con {target_col}")
    else:
        logger.info("No se detectó leakage evidente en las variables numéricas que serán utilizadas para el entrenamiento.")
        
    return corr