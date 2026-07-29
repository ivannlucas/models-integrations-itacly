import joblib
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from src.utils.logging import get_logger
from src.utils.monitor import log_to_monitorization

logger = get_logger("PREDICTOR")

def load_config():
    """Carga el archivo de configuración global."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def monitor_model_health(current_conf):
    cfg = load_config()
    system = cfg["selected_system"]
    window = cfg[system].get("window_degradation", 50)
    
    # 1. Leer el historial directamente del archivo unificado
    log_path = Path(cfg["paths"]["logs"]) / f"monitorization_{system}.csv"
    
    history = []
    if log_path.exists():
        try:
            df_log = pd.read_csv(log_path)
            # Filtramos solo registros de tipo 'PREDICT' que tengan columna 'confidence'
            # y tomamos las últimas 'window' muestras
            if 'confidence' in df_log.columns:
                history = df_log['confidence'].dropna().tail(window).tolist()
        except Exception as e:
            logger.warning(f"No se pudo leer el histórico de monitorización: {e}")

    # 2. Calcular media y definir estado
    # Incluimos la predicción actual en el cálculo
    temp_history = history + [current_conf]
    rolling_avg = np.mean(temp_history)
    threshold = cfg[system].get("health_threshold", 75.0)
    
    status = "DEGRADADO" if rolling_avg < threshold else "ESTABLE"
    
    # 3. Guardar el nuevo registro en el archivo unificado
    log_to_monitorization(system, {
        'status': status,
        'confidence': current_conf
    })
    
    if status == "DEGRADADO":
        logger.warning(f"⚠️ DEGRADACIÓN DETECTADA: Salud (media): {rolling_avg:.2f}%")
    else:
        logger.info(f"Modelo ESTABLE. Salud actual: {current_conf:.2f}%, Promedio ({len(temp_history)}): {rolling_avg:.2f}%")
        
    return status
    

def load_model(system_type: str):
    """
    Busca modelos disponibles. Si solo existe el base, lo carga automáticamente.
    Si hay calibraciones, despliega un menú interactivo en terminal para elegir.
    
    Returns:
        tuple: (objeto_modelo, selected_suffix)
    """
    cfg = load_config()
    artifacts_path = Path(cfg["paths"]["artifacts"])
    
    # Buscamos todos los modelos calibrados disponibles para este sistema específico
    calibrated_models = sorted(list(artifacts_path.glob(f"{system_type}_model_calibrated_*.pkl")))
    
    selected_suffix = None  # None representa usar el modelo Base / [INICIAL]
    model_path = artifacts_path / f"{system_type}_model.pkl"
    
    if calibrated_models:
        print(f"\nSe han detectado varias versiones para el sistema [{system_type.upper()}].")
        print("Escoge la fecha del modelo con el que quieres hacer inferencia:")
        print("0) [INICIAL] Modelo base original de fábrica")
        
        for i, path in enumerate(calibrated_models, start=1):
            fecha_str = path.stem.replace(f"{system_type}_model_calibrated_", "")
            print(f"{i}) [CALIBRADO] Versión del {fecha_str}")
            
        while True:
            try:
                opcion = int(input(f"Seleccione una opción (0-{len(calibrated_models)}): "))
                if 0 <= opcion <= len(calibrated_models):
                    break
                print("⚠️ Opción inválida. Intente de nuevo.")
            except ValueError:
                print("⚠️ Por favor, introduzca un número válido.")
                
        if opcion > 0:
            chosen_model_path = calibrated_models[opcion - 1]
            selected_suffix = chosen_model_path.stem.replace(f"{system_type}_model_calibrated_", "")
            model_path = chosen_model_path
            logger.info(f"-> Operador seleccionó modelo calibrado: {model_path.name}")
        else:
            logger.info("-> Operador seleccionó modelo [INICIAL] original.")
    else:
        # Si no hay calibraciones previas, pasa directo de forma automática sin preguntar
        logger.info(f"No se detectaron calibraciones previas para {system_type}. Usando modelo INICIAL automáticamente.")

    if not model_path.exists():
        logger.error(f"Archivo de modelo no encontrado: {model_path}")
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    
    return joblib.load(model_path), selected_suffix

def run_inference(model, X: pd.DataFrame, system_type: str, selected_suffix=None):
    """Ejecuta la inferencia asegurando limpieza, alineación y escalado alineado."""
    try:
        cfg = load_config()
        
        # 1. Identificar columnas a eliminar (Metadatos y target)
        drop_cols_cfg = cfg.get(system_type, {}).get('drop_cols', [])
        metadatos = ['run_id', 'time_min', 'fault_id', 'fault', 'fault_numeric', 'prediction', 'confidence']
        to_drop = list(set(drop_cols_cfg + metadatos))
        
        X_cleaned = X.drop(columns=[c for c in to_drop if c in X.columns], errors='ignore')
        
        # 2. Alineación estricta con el entrenamiento
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            X_cleaned = X_cleaned[expected_features]

        # 3. Escalado dinámico y sincronizado
        if system_type == "refrigeracion":
            artifacts_path = Path(cfg["paths"]["artifacts"])
            scaler_path = artifacts_path / f"{system_type}_scaler.pkl"
            
            # Buscamos el archivo de estadísticas del modelo que se haya seleccionado
            if selected_suffix:
                stats_path = artifacts_path / f"{system_type}_stats_calibrated_{selected_suffix}.yaml"
            else:
                stats_path = artifacts_path / f"{system_type}_stats.yaml"
            
            if scaler_path.exists():
                logger.info(f"Cargando escalador empleando referencias de: {stats_path.name}")
                scaler = joblib.load(scaler_path)
                
                if hasattr(scaler, 'feature_names_in_'):
                    X_cleaned.loc[:, scaler.feature_names_in_] = scaler.transform(X_cleaned[scaler.feature_names_in_])
                else:
                    binary_cols = ['defrost_on', 'door_open']
                    num_cols = [c for c in X_cleaned.columns if c not in binary_cols]
                    X_cleaned.loc[:, num_cols] = scaler.transform(X_cleaned[num_cols])
            else:
                logger.warning("No se encontró scaler. Continuando sin escalado.")

        # 4. Predicción
        y_pred = model.predict(X_cleaned)
        y_probs = model.predict_proba(X_cleaned)
        
        return y_pred, y_probs

    except Exception as e:
        logger.error(f"Error durante la inferencia: {str(e)}")
        raise

def save_predictions(df_pred: pd.DataFrame, system_type: str) -> None:
    """
    Guarda las predicciones colapsando los datos por run_id.
    Si existen etiquetas reales (evaluación), las mantiene.
    Si no existen (inferencia real), exporta solo predicción y confianza.
    """
    cfg = load_config()
    
    # 1. Definir columnas de metadatos/referencia y resultados
    # 'run_id' es nuestra ancla para agrupar
    reference_cols = ['fault_id', 'fault'] 
    result_cols = ['prediction', 'confidence']
    
    # Identificar qué columnas de referencia están presentes en el DF
    present_ref = [c for c in reference_cols if c in df_pred.columns]
    
    # Columnas finales que procesaremos
    target_cols = ['run_id'] + present_ref + result_cols
    existing_cols = [c for c in target_cols if c in df_pred.columns]
    
    # 2. Construir reglas de agregación dinámicas
    agg_rules = {}
    for col in existing_cols:
        if col == 'run_id':
            continue
        if col == 'confidence':
            agg_rules[col] = 'mean' # Promedio de confianza en el ciclo
        else:
            agg_rules[col] = 'first' # Primera predicción (asumida estable por el voto)

    # 3. Agrupar y colapsar
    logger.info(f"Colapsando datos por ciclo (run_id) para {system_type}...")
    
    try:
        df_runs = df_pred[existing_cols].groupby('run_id').agg(agg_rules).reset_index()
        
        # 4. Obtener ruta desde el YAML
        out_dir = Path(cfg["paths"]["predictions_data"]) 
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"predictions_{system_type}.csv"

        # 5. Guardar CSV final
        df_runs.to_csv(out_file, index=False)
        
        logger.info(f"✅ Exportación completada: {len(df_runs)} ciclos procesados.")
        if not present_ref:
            logger.info("ℹ️ Modo Inferencia Real: No se detectaron etiquetas de referencia (fault_id).")
        
        logger.info(f"📍 Archivo generado: {out_file}")

    except Exception as e:
        logger.error(f"Error al colapsar o guardar predicciones: {str(e)}")
        raise