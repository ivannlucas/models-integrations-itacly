import joblib
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger("PREDICTOR")

def load_config():
    """Carga el archivo de configuración global."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def monitor_model_health(current_conf):
    cfg = load_config()
    system = cfg["selected_system"]
    window = cfg[system].get("window_degradation", 50) 
    
    logs_dir = Path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True) 
    history_path = logs_dir / f"health_history_{system}.csv"
    
    # 1. Cargar historial existente o crear uno nuevo
    if history_path.exists():
        try:
            # CORRECCIÓN: Leemos con pandas porque el archivo se guarda como CSV
            df_hist = pd.read_csv(history_path)
            history = df_hist['confidence'].tolist()
        except Exception:
            # Si el archivo está corrupto (por el error anterior), empezamos de cero
            logger.warning(f"Historial de salud corrupto o ilegible. Reiniciando historial.")
            history = []
    else:
        history = []

    # 2. Actualizar
    history.append(current_conf)
    if len(history) > window:
        history.pop(0)

    # 3. Guardar para la próxima ejecución
    # Mantenemos tu método de guardado que es correcto (CSV)
    pd.DataFrame({'confidence': history}).to_csv(history_path, index=False)

    # 4. Calcular media y alertar
    rolling_avg = np.mean(history)
    threshold = cfg[system].get("health_threshold", 75.0)
    
    if rolling_avg < threshold:
        logger.warning(f"⚠️ DEGRADACIÓN DETECTADA: Salud {rolling_avg:.2f}%")
        status = "DEGRADADO"
    else:
        logger.info(f"Salud actual: {current_conf:.2f}%, Promedio rolling ({window}): {rolling_avg:.2f}%")
        logger.info(f"Modelo ESTABLE. Historial actualizado en: {history_path}")
        status = "ESTABLE"

    return status
    

def load_model(system_type: str):
    """Carga el modelo desde la ruta definida en config.yaml."""
    cfg = load_config()
    artifacts_path = Path(cfg["paths"]["artifacts"])
    model_path = artifacts_path / f"{system_type}_model.pkl"
    
    if not model_path.exists():
        logger.error(f"Archivo de modelo no encontrado: {model_path}")
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    
    logger.info(f"Modelo {system_type} cargado correctamente.")
    return joblib.load(model_path)

def run_inference(model, X: pd.DataFrame, system_type: str):
    """Ejecuta la inferencia asegurando limpieza, alineación y escalado."""
    try:
        cfg = load_config()
        
        # 1. Identificar columnas a eliminar (Metadatos y target)
        drop_cols_cfg = cfg.get(system_type, {}).get('drop_cols', [])
        metadatos = ['run_id', 'time_min', 'fault_id', 'fault', 'fault_numeric', 'prediction', 'confidence']
        to_drop = list(set(drop_cols_cfg + metadatos))
        
        # X_cleaned solo contiene las FEATURES para el modelo
        X_cleaned = X.drop(columns=[c for c in to_drop if c in X.columns], errors='ignore')
        
        # 2. Alineación estricta con el entrenamiento
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            X_cleaned = X_cleaned[expected_features]

        # 3. Escalado (Usa path de config.yaml)
        if system_type == "refrigeracion":
            artifacts_path = Path(cfg["paths"]["artifacts"])
            scaler_path = artifacts_path / f"{system_type}_scaler.pkl"
            
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                if hasattr(scaler, 'feature_names_in_'):
                    X_cleaned.loc[:, scaler.feature_names_in_] = scaler.transform(X_cleaned[scaler.feature_names_in_])
                else:
                    # Fallback para scalers sin nombres de columnas
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