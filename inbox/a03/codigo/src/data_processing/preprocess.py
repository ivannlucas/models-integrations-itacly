import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _calcular_gdd_serie(temp_series: np.ndarray, t_base: float = 10.0) -> np.ndarray:
    """
    Calcula la integral térmica (Grados-Día de Crecimiento) para una serie.
    Fórmula idéntica a vid_simulator.calcular_gdd_acumulado.
    """
    n_steps = len(temp_series)
    gdd = np.zeros(n_steps)
    acumulado = 0.0
    pasos_dia = 24
    dias = n_steps // pasos_dia

    for d in range(dias):
        i0 = d * pasos_dia
        i1 = i0 + pasos_dia
        t_media = np.mean(temp_series[i0:i1])
        acumulado += max(0, t_media - t_base)
        gdd[i0:i1] = acumulado

    # Horas residuales (serie no divisible por 24)
    resto = n_steps % pasos_dia
    if resto > 0:
        gdd[dias * pasos_dia:] = acumulado

    return gdd


def _calcular_horas_mojado_serie(lluvia_series: np.ndarray, hr_series: np.ndarray) -> np.ndarray:
    """
    Calcula las horas consecutivas de mojado foliar para una serie.
    La hoja está mojada si llueve (>0.1 mm) O si HR > 90%.
    Fórmula idéntica a vid_simulator.calcular_horas_mojado.
    """
    n_steps = len(lluvia_series)
    wetness = np.zeros(n_steps)
    count = 0

    for t in range(n_steps):
        if lluvia_series[t] > 0.1 or hr_series[t] > 90.0:
            count += 1
        else:
            count = 0
        wetness[t] = count

    return wetness


def apply_feature_engineering(df, date_column: str = "Fecha", strict: bool = True,
                              series_column: str = "ID_Serie"):
    """
    Aplica la ingeniería de variables temporales y físicas al DataFrame.
    Función compartida entre preprocesamiento e inferencia para evitar train-serving skew.
    
    Genera automáticamente las siguientes features si no están presentes:
      - Hora_Sin / Hora_Cos: Codificación cíclica de la hora del día
      - GDD_Acumulado: Integral térmica acumulada (base 10°C), calculada por serie
      - Horas_Humedad_Foliar: Horas consecutivas con mojado foliar, calculada por serie
    
    Args:
        df: DataFrame con los datos de entrada.
        date_column: Nombre de la columna temporal (configurable desde config.yaml).
        strict: Si True, lanza error si falta la columna temporal. Si False, la omite.
        series_column: Nombre de la columna de ID de serie temporal.
    
    Returns:
        DataFrame con las features generadas.
    """
    # --- 1. Features temporales cíclicas ---
    if date_column in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        df['Hora'] = df[date_column].dt.hour
        df['Hora_Sin'] = np.sin(2 * np.pi * df['Hora'] / 24).astype('float32')
        df['Hora_Cos'] = np.cos(2 * np.pi * df['Hora'] / 24).astype('float32')
    elif strict:
        raise ValueError(
            f"Columna '{date_column}' no encontrada en el input. Es imprescindible para "
            "generar Hora_Sin y Hora_Cos."
        )

    # --- 2. GDD Acumulado (si no viene precalculado) ---
    if 'GDD_Acumulado' not in df.columns:
        if 'Temp_Amb_C' not in df.columns:
            raise ValueError(
                "Se necesita 'Temp_Amb_C' para calcular 'GDD_Acumulado', pero no está en el input."
            )
        logger.debug("Generando GDD_Acumulado a partir de Temp_Amb_C...")
        
        if series_column in df.columns:
            # Calcular por serie temporal independiente
            gdd_results = []
            for _, group in df.groupby(series_column):
                gdd_results.append(_calcular_gdd_serie(group['Temp_Amb_C'].values))
            df['GDD_Acumulado'] = np.concatenate(gdd_results).astype('float32')
        else:
            # Serie única (inferencia con un solo bloque)
            df['GDD_Acumulado'] = _calcular_gdd_serie(df['Temp_Amb_C'].values).astype('float32')

    # --- 3. Horas Humedad Foliar (si no viene precalculado) ---
    if 'Horas_Humedad_Foliar' not in df.columns:
        if 'Lluvia_mm' not in df.columns or 'Hum_Rel_Pct' not in df.columns:
            raise ValueError(
                "Se necesitan 'Lluvia_mm' y 'Hum_Rel_Pct' para calcular "
                "'Horas_Humedad_Foliar', pero no están en el input."
            )
        logger.debug("Generando Horas_Humedad_Foliar a partir de Lluvia_mm y Hum_Rel_Pct...")
        
        if series_column in df.columns:
            mojado_results = []
            for _, group in df.groupby(series_column):
                mojado_results.append(
                    _calcular_horas_mojado_serie(group['Lluvia_mm'].values, group['Hum_Rel_Pct'].values)
                )
            df['Horas_Humedad_Foliar'] = np.concatenate(mojado_results).astype('float32')
        else:
            df['Horas_Humedad_Foliar'] = _calcular_horas_mojado_serie(
                df['Lluvia_mm'].values, df['Hum_Rel_Pct'].values
            ).astype('float32')
    
    return df


def run_data_processing(raw_data_path: str, processed_dir: str, raw_features: List[str], model_features: List[str], config: dict) -> str:
    """
    Ejecuta el pipeline de procesamiento de datos en crudo y lo guarda en procesado.
    """
    ruta_raw = Path(raw_data_path)
    ruta_proc = Path(processed_dir)
    ruta_proc.mkdir(parents=True, exist_ok=True)
    
    if not ruta_raw.exists():
        logger.error(f"El archivo {ruta_raw.resolve()} no existe.")
        raise FileNotFoundError(f"Input file not found at {ruta_raw}")
        
    logger.info(f"Cargando dataset raw desde: {ruta_raw}")
    df = pd.read_parquet(ruta_raw)
    
    # VALIDACIÓN DE VARIABLES CRUDAS (Garantiza robustez ante cambios en sensores)
    missing_raw = [c for c in raw_features if c not in df.columns]
    if missing_raw:
        logger.error(f"Faltan variables crudas esenciales en el input: {missing_raw}")
        raise ValueError(f"Faltan variables crudas: {missing_raw}")

    # Aseguramos el cruce de orden alfabético
    df = df.reindex(sorted(df.columns), axis=1)
        
    # Castings a string requeridos (leído de config)
    cols_str = config.get('string_columns', [])
    for col in cols_str:
        if col in df.columns:
            df[col] = df[col].astype(str)

    logger.info("Generando features temporales...")
    date_column = config.get('date_column', 'Fecha')
    df = apply_feature_engineering(df, date_column=date_column, strict=True)
        
    # Comprobación de Features finales necesarias para el modelo
    missing_model = [c for c in model_features if c not in df.columns]
    if missing_model:
        raise ValueError(f"No se pudieron generar todas las model_features esperadas: {missing_model}")
        
    out_filename = config['output_files']['processed_data']
    out_path = ruta_proc / out_filename
    df.to_parquet(out_path)
    logger.info(f"Conjunto de datos procesado guardado en: {out_path.resolve()}")
    
    return str(out_path)
