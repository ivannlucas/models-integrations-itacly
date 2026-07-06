import pandas as pd
import os
from src.utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# DICCIONARIO DE DESCRIPCIONES DE LAS FEATURES DEL DATASET
# =============================================================================
FEATURE_DESCRIPTIONS = {
    "Cycle_ID":           "Identificador único de cada ciclo.",
    "Time_Segundos":      "Marca temporal de la serie para establecer un orden.",
    "PS1":                "Presión de entrada.",
    "PS3":                "Presión de salida.",
    "EPS1":               "Potencia Motor.",
    "FS1":                "Caudal / Flujo salida.",
    "TS1":                "Temperatura entrada. Transformada con desplazamiento térmico al baseline de 65ºC.",
    "TS2":                "Temperatura salida. Transformada con desplazamiento térmico al baseline de 65ºC.",
    "VS1":                "Vibración del tubo de entrada.",
    "Target_Fouling":     "Estado del Enfriador (eficiencia térmica). [0=Sano, 1=Warning, 2=Crítico]",
    "Target_Valvula":     "Estado de la Válvula (tiempos de conmutación). [0=Sano, 1=Warning, 2=Crítico]",
    "Target_Bomba":       "Estado de la Bomba (fugas internas). [0=Sano, 1=Warning, 2=Crítico]",
    "Target_Acumulador":  "Estado del Acumulador (presión de gas). [0=Sano, 1=Warning, 2=Crítico]",
}

# =============================================================================
# DESCRIPCIÓN DEL MODELO Y SU PROPÓSITO
# =============================================================================
MODEL_DESCRIPTION = (
    "Modelo utilizado: Deep Neurosymbolic Learning (DNSL). Es una arquitectura híbrida que fusiona "
    "una Red Neuronal Convolucional Unidimensional (1D-CNN), encargada de extraer patrones de las "
    "señales temporales de los sensores, con una función de pérdida guiada por la física (lógica "
    "neurosimbólica). Esta función evalúa matemáticamente que las predicciones de la red respeten "
    "las leyes inmutables de la termodinámica y la conservación de la energía, penalizando cualquier "
    "resultado que sea físicamente imposible."
)

MODEL_PURPOSE = (
    "Propósito: Detectar, anticipar y clasificar fallos críticos (como incrustaciones, fugas, "
    "cavitación o problemas en válvulas) en cuatro componentes clave de la maquinaria de "
    "pasteurización (enfriador, válvula, bomba y acumulador) de forma simultánea y en tiempo real. "
    "Su objetivo es evitar paradas de producción no planificadas, reducir los costes de mantenimiento "
    "y garantizar la seguridad del producto lácteo."
)


def generate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un listado de las features del dataset con su descripción estadística.
    """
    logger.info("Generando estadísticas descriptivas...")
    stats = df.describe().T
    
    # Agregamos correlación cruzada de las primeras 10 features como extra
    cols = list(df.select_dtypes(include=['number']).columns[:10])
    if cols:
        corr = df[cols].corr()
        logger.info("\nCorrelaciones principales (Top 10 features):")
        logger.info(corr)
        
    return stats


def generate_feature_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un DataFrame con el listado de features presentes en el dataset
    y su descripción textual.
    """
    logger.info("Generando listado de features con descripción...")
    
    rows = []
    for col in df.columns:
        desc = FEATURE_DESCRIPTIONS.get(col, "Sin descripción disponible.")
        rows.append({"Feature": col, "Descripción": desc})
    
    return pd.DataFrame(rows)


def print_model_info():
    """
    Imprime por consola la descripción del modelo y su propósito.
    """
    logger.info("=" * 80)
    logger.info("INFORMACIÓN DEL MODELO")
    logger.info("=" * 80)
    logger.info(MODEL_DESCRIPTION)
    logger.info("")
    logger.info(MODEL_PURPOSE)
    logger.info("=" * 80)


def load_trained_metrics(metrics_dir: str) -> dict:
    """
    Carga todas las métricas del modelo entrenado que se encuentren como .csv
    en la carpeta indicada. Devuelve un diccionario {nombre_archivo: DataFrame}.
    Devuelve un diccionario vacío si la carpeta no existe o no contiene CSVs.
    """
    metrics = {}
    
    if not os.path.isdir(metrics_dir):
        logger.warning(f"La carpeta de métricas '{metrics_dir}' no existe. "
                       "Ejecuta el entrenamiento con --metrics para generarlas.")
        return metrics
    
    csv_files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.warning(f"No se encontraron archivos .csv en '{metrics_dir}'. "
                       "Ejecuta el entrenamiento con --metrics para generarlas.")
        return metrics
    
    for csv_file in csv_files:
        filepath = os.path.join(metrics_dir, csv_file)
        name = os.path.splitext(csv_file)[0]
        try:
            metrics[name] = pd.read_csv(filepath)
            logger.info(f"Métricas cargadas: {csv_file}")
        except Exception as e:
            logger.warning(f"No se pudo leer '{csv_file}': {e}")
    
    return metrics
