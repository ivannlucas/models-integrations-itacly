import pandas as pd
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)

def run_stats(config: dict) -> dict:
    """
    Devuelve un diccionario con las estadísticas e información descriptiva
    de cada columna que usa el modelo de Machine Learning.
    
    Args:
        config: Diccionario de configuración global (config.yaml).
    
    Returns:
        dict: Diccionario JSON / Hash map de {columna: descripción}
    """
    logger.info("Generando estadísticas e información descriptiva de las variables del modelo...")
    
    # b) Pequeña descripción del modelo y el objetivo del mismo
    print("\n" + "="*60)
    print(" MODELO DE DETECCIÓN DE ENFERMEDADES Y PLAGAS EN LA VID ")
    print("="*60)
    print("Objetivo: Clasificar y detectar con anticipación la presencia de patologías")
    print("y plagas en cultivos de vid utilizando un modelo de Deep Ensemble Learning.")
    print("El sistema emplea redes LSTM, CNN-1D y BiGRU para analizar secuencias")
    print("temporales de sensores meteorológicos, de suelo y térmicos.")
    print("="*60 + "\n")
    
    column_descriptions = {
        "Temp_Amb_C": "Temperatura ambiente medida por telemetría del sensor metereológico en Grados Celsius.",
        "Hum_Rel_Pct": "Humedad relativa ambiental (Porcentaje %); clave para inferir punto de rocío y formación de microgotas.",
        "Lluvia_mm": "Cantidad de pluviosidad recogida (milímetros o l/m2) en la ventana de muestreo.",
        "Viento_kmh": "Velocidad media del viento; esparce esporas de Oídio, Mildiu y Botrytis.",
        "Horas_Humedad_Foliar": "Horas continuas que la hoja permanece con humedad condensada, precursor vital para proliferación de hongos.",
        "GDD_Acumulado": "Grados Día de Desarrollo acumulado (Growing Degree Days). Predice eventos biológicos como brotación o eclosión de plagas.",
        "Hum_Suelo_Pct": "Humedad volumétrica de la tierra en porcentaje (%), afecta absorción de agua y susceptibilidad a podredumbres de la raíz.",
        "pH_Suelo": "Acidez (pH) del terreno. Condiciona la solubilidad de macro y micronutrientes como el hierro.",
        "CO2_ppm": "Partículas de Dióxido de Carbono por millón. Relacionado indirectamente a la tasa de estomas abiertos y transpiración foliar.",
        "VOC_ppb": "Partes por billón de Compuestos Orgánicos Volátiles emitidos por la planta al inicio de la infección (biomarcador temprano).",
        "Hora_Sin": "Componente sinusal de la hora (Ciclo Trigonométrico) para dar continuidad horaria de media-noche 23:59h -> 00:00h.",
        "Hora_Cos": "Componente cosinusal de la hora (acompaña al Hora_Sin para ubicar el cuadrante del día temporalmente).",
        "ID_Serie": "Identificador único (sensor, parcela, o serie temporal finita). Esencial evitar mezclar ventanas entre IDs (Leakage).",
        "Etiqueta_Clase": "Label (Target/Y) de la clase patológica en texto (ej: LOBESIA, HEALTHY, MildiU, etc)",
        "Grado_Infeccion": "Feature cuantitativa entre 0.0 y 1.0 (Output Y_Reg). Denota severidad o masa fungítica detectada en el cultivo",
        "Clase_Entrenamiento": "Transformación a string del campo patológico utilizado para agrupar o estratificar por SkLearn.",
        "Etiqueta_Num": "Vector entero 0..N del Label codificado a base del LabelEncoder."
    }
    
    logger.info("Listado de features generado correctamente.")
    for key, val in column_descriptions.items():
        print(f"- **{key}**: {val}")
        
    # Export descriptions to CSV (rutas desde config)
    proc_dir = Path(config['paths']['processed_dir'])
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    desc_df = pd.DataFrame(list(column_descriptions.items()), columns=["Feature", "Descripcion"])
    desc_df.to_csv(proc_dir / config['output_files']['feature_descriptions'], index=False, encoding="utf-8-sig")
    logger.info("Descripciones exportadas a feature_descriptions.csv")
    
    # Cargar dataset PROCESADO para calcular estadísticas (incluye Delta_T, Hora_Sin, Hora_Cos)
    processed_filename = config['output_files']['processed_data']
    processed_path = proc_dir / processed_filename
    if processed_path.exists():
        df_proc = pd.read_parquet(processed_path)
        stats = df_proc.describe().transpose()
        stats.to_csv(proc_dir / config['output_files']['estadisticas'], encoding="utf-8-sig")
        logger.info("Estadísticas del dataset procesado exportadas a estadisticas.csv")
    else:
        logger.warning(f"No se encontró el dataset procesado en {processed_path}. No se generan estadísticas numéricas.")
    
    # c) Metricas del modelo entrenado (en caso de que lo esté)
    metrics_dir = Path(config['paths']['metrics_dir'])
    metrics_path = metrics_dir / config['output_files']['metrics_ensemble_dir'] / config['output_files']['metrics_ensemble_file']
    if metrics_path.exists():
        print("\n" + "="*60)
        print(" METRICAS DEL MODELO ENTRENADO (Deep Ensemble Learning) ")
        print("="*60)
        df_metrics = pd.read_csv(metrics_path, index_col=0)
        # Mostrar f1-score y precision de las clases y accuracy global
        if 'f1-score' in df_metrics.columns:
            print(df_metrics[['precision', 'recall', 'f1-score']].head(10).to_string())
            if 'accuracy' in df_metrics.index:
                acc = df_metrics.loc['accuracy', 'f1-score']
                print(f"\nExactitud Global (Accuracy): {acc*100:.2f}%")
        print("="*60 + "\n")
    else:
        logger.warning("No se encontraron métricas de evaluación previas en models/metrics/DEL/")
        
    return column_descriptions

