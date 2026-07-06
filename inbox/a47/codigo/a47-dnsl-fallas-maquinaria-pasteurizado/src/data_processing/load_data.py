import os
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_txt_and_convert_to_csv(ruta_datos: str, output_csv: str) -> pd.DataFrame:
    """
    Lee los ficheros .txt originales, los fusiona y transforma a formato 'Long',
    y luego guarda el resultado en un archivo .csv.
    """
    nombres_sensores = [
        'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 
        'EPS1',                                   
        'FS1', 'FS2',                             
        'TS1', 'TS2', 'TS3', 'TS4',               
        'VS1',                                    
        'CE', 'CP', 'SE'                          
    ]
    
    # Cargar etiquetas
    profile_path = os.path.join(ruta_datos, 'profile.txt')
    if not os.path.exists(profile_path):
        logger.error(f"No se encontró el archivo de etiquetas: {profile_path}")
        return pd.DataFrame()

    logger.info("Cargando archivo profile.txt...")
    df_completo = pd.read_csv(profile_path, sep='\t', header=None)
    df_completo.columns = ['Cooler_Condition', 'Valve_Condition', 'Pump_Leakage', 'Hydraulic_Accumulator', 'Stable_Flag']
    
    # Cargar sensores
    logger.info("Iniciando importación de sensores desde .txt...")
    for sensor in nombres_sensores:
        archivo = os.path.join(ruta_datos, f'{sensor}.txt')
        if not os.path.exists(archivo):
            logger.warning(f"No se encontró el archivo del sensor: {archivo}, saltando...")
            continue
        df_temporal = pd.read_csv(archivo, sep='\t', header=None)
        df_temporal.columns = [f'{sensor}_{i}' for i in range(df_temporal.shape[1])]
        df_completo = pd.concat([df_completo, df_temporal], axis=1)

    logger.info(f"Importación completada. Dimensiones: {df_completo.shape}")

    # Transformación a long
    n_ciclos = len(df_completo)
    n_puntos_por_ciclo = 6000 # asumiendo 60s a 100Hz max
    logger.info(f"Transformando a formato largo para {n_ciclos} ciclos...")

    cycle_ids = np.repeat(np.arange(n_ciclos), n_puntos_por_ciclo)
    time_axis = np.tile(np.arange(0, 60, 0.01), n_ciclos)

    data_dict = {
        'Cycle_ID': cycle_ids,
        'Time': time_axis
    }

    sensores_config = {
        'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100, 'EPS1': 100,
        'FS1': 10, 'FS2': 10,
        'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1, 'VS1': 1, 'CE': 1, 'CP': 1, 'SE': 1
    }

    for sensor, freq in sensores_config.items():
        cols = [c for c in df_completo.columns if c.startswith(f"{sensor}_")]
        if not cols:
            continue
        matriz_valores = df_completo[cols].to_numpy()
        
        if freq == 100:
            data_dict[sensor] = matriz_valores.flatten()
        else:
            factor_repeticion = 6000 // matriz_valores.shape[1]
            matriz_expandida = matriz_valores.repeat(factor_repeticion, axis=1)
            data_dict[sensor] = matriz_expandida.flatten()

    targets = ['Cooler_Condition', 'Valve_Condition', 'Pump_Leakage', 'Hydraulic_Accumulator', 'Stable_Flag']
    for target in targets:
        if target in df_completo.columns:
            vals = df_completo[target].to_numpy()
            data_dict[target] = vals.repeat(n_puntos_por_ciclo)

    df_long = pd.DataFrame(data_dict)

    # Optimización de memoria
    for col in df_long.select_dtypes(include=['float64']).columns:
        df_long[col] = df_long[col].astype('float32')

    logger.info(f"Guardando datos crudos largos en {output_csv}...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_long.to_csv(output_csv, index=False)
    
    return df_long

def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """
    Carga el csv si existe, ahorrando el tiempo de parseo.
    """
    if not os.path.exists(csv_path):
        logger.error(f"El archivo {csv_path} no existe.")
        raise FileNotFoundError(f"{csv_path} no encontrado")
        
    logger.info(f"Cargando dataset desde {csv_path}...")
    return pd.read_csv(csv_path)
