import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from tqdm import tqdm

from src.utils.logging import get_logger
from src.data_processing.preprocess import apply_feature_engineering

logger = get_logger(__name__)


def predecir_ensemble(X_input, modelos_ensemble, label_encoder):
    preds_clases_prob, preds_regresion = [], []
    for mod in modelos_ensemble:
        p_class, p_reg = mod.predict(X_input, verbose=0)
        preds_clases_prob.append(p_class)
        preds_regresion.append(p_reg)
        
    media_probs = np.mean(preds_clases_prob, axis=0)
    clases_finales_idx = np.argmax(media_probs, axis=1)
    confianza_final = np.max(media_probs, axis=1)
    nombres_clases = label_encoder.inverse_transform(clases_finales_idx)
    grado_final = np.mean(preds_regresion, axis=0).flatten()
    
    return nombres_clases, grado_final, confianza_final

def run_inference(input_csv: str, config: dict, id_serie: int = None):
    logger.info(f"Cargando dataset para inferencia: {input_csv}")
    df_raw = pd.read_csv(input_csv) if str(input_csv).endswith('.csv') else pd.read_parquet(input_csv)
    df = df_raw.copy()
    
    series_col = config['targets']['series_column']
    window_size = config['params']['window_size']
    date_column = config.get('date_column', 'Fecha')
    
    if id_serie is not None:
        if series_col in df.columns:
            logger.info(f"Filtrando dataset de pruebas por {series_col}={id_serie}")
            df = df[df[series_col] == id_serie].copy()
            if df.empty:
                logger.error(f"No se encontraron datos para {series_col}={id_serie}")
                return
        else:
            logger.warning(f"Argumento --id_serie provisto, pero el dataframe no contiene columna '{series_col}'. Omitiendo filtro.")

    # Guardar si hay múltiples series o no para saber cómo procesar el final
    has_multiple_series = series_col in df.columns and df[series_col].nunique() > 1

    series_dataframes = []
    if has_multiple_series:
        logger.info(f"Múltiples series detectadas ({df[series_col].nunique()}). Agrupando...")
        for name, group in df.groupby(series_col):
            series_dataframes.append((name, group.copy()))
    else:
        # Una sola serie (ya sea porque vino así o porque se filtró con --id_serie)
        serie_name = df[series_col].iloc[0] if series_col in df.columns else "Única"
        series_dataframes.append((serie_name, df))
    
    # Preparamos las estructuras para la iteración final (o única)
    ruta_arts = Path(config['paths']['artifacts_dir'])
    scaler = joblib.load(ruta_arts / config['model_names']['scaler'])
    le = joblib.load(ruta_arts / config['model_names']['label_encoder'])
    
    modelos = [
        load_model(ruta_arts / config['model_names']['m1_lstm']),
        load_model(ruta_arts / config['model_names']['m2_cnn']),
        load_model(ruta_arts / config['model_names']['m3_bigru'])
    ]

    cols_features = config.get('cols_features', [])
    resultados_totales = []

    for serie_name, serie_df in tqdm(series_dataframes, desc="Prediciendo series"):
        n_filas = len(serie_df)
        if n_filas > window_size:
            serie_df = serie_df.tail(window_size).copy()
        elif n_filas < window_size:
            pad = window_size - n_filas
            df_pad = pd.concat([serie_df.iloc[[0]]] * pad, ignore_index=True)
            serie_df = pd.concat([df_pad, serie_df], ignore_index=True)
        serie_df.reset_index(drop=True, inplace=True)
    
        # 2. INGENIERÍA (Función compartida con preprocess para evitar train-serving skew)
        serie_df = apply_feature_engineering(serie_df, date_column=date_column, strict=False)
            
        X_input = serie_df[cols_features].copy()
        
        X_scaled = scaler.transform(X_input)
        X_tensor = X_scaled.reshape(1, window_size, len(cols_features))
        
        pat_pred, grado_inf, conf = predecir_ensemble(X_tensor, modelos, le)
        patologia, grado, confianza = pat_pred[0], grado_inf[0], conf[0]
        
        BASE_CONOCIMIENTO = config.get('base_conocimiento_tratamientos', {})
        trats_dict = BASE_CONOCIMIENTO.get(patologia, {"Aviso": "N/A"})
        tratamientos = " | ".join([f"{k}: {v}" for k, v in trats_dict.items()])
        
        fecha_val = serie_df[date_column].iloc[-1] if date_column in serie_df.columns else "Desconocida"
        
        resultados_totales.append({
            series_col: serie_name,
            "Fecha_Evaluacion": fecha_val,
            "Diagnostico_IA": patologia,
            "Confianza_Clasificacion": f"{confianza * 100:.2f}%",
            "Grado_Severidad": round(float(grado), 4),
            "Tratamiento_Recomendado": tratamientos
        })
        
    resultado_df = pd.DataFrame(resultados_totales)
    
    ruta_preds = Path(config['paths']['predictions_dir'])
    ruta_preds.mkdir(parents=True, exist_ok=True)
    predictions_filename = config['output_files']['predictions']
    out_file = ruta_preds / predictions_filename
    
    resultado_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    logger.info(f"Predicción guardada con éxito en: {out_file.resolve()}")
    
    return str(out_file)
