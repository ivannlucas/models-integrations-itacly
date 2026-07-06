import torch
import joblib
import os
import pandas as pd
import numpy as np
from src.training.model import CNN_Pasteurizer
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_artifacts(models_dir: str, config: dict):
    """
    Carga el scaler, las columnas, la media de temperatura y el modelo.
    """
    logger.info(f"Cargando artefactos desde {models_dir}...")
    scaler = joblib.load(os.path.join(models_dir, "scaler_cnn_dns.pkl"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
    ts1_mean_train = joblib.load(os.path.join(models_dir, "ts1_mean_train.pkl"))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Entorno GPU detectado correctamente. Inicializando inferencia en CUDA...")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Entorno Apple Silicon GPU detectado. Inicializando inferencia en MPS...")
    else:
        device = torch.device("cpu")
        logger.warning("No se ha detectado GPU. Inferencia ejecutándose en CPU. Revisa tu instalación de PyTorch si dispones de hardware acelerador.")
        
    n_canales = len(feature_cols)
    n_classes = config['training'].get('n_classes', 3)
    dropout_prob = config['training'].get('dropout_rate', 0.5)
    model = CNN_Pasteurizer(n_sensors=n_canales, n_classes=n_classes, dropout_prob=dropout_prob).to(device)
    model.load_state_dict(torch.load(os.path.join(models_dir, "neurosymbolic_cnn.pth"), map_location=device, weights_only=True))
    model.eval()
    
    logger.info("Modelo y scaler cargados correctamente.")
    return model, scaler, feature_cols, ts1_mean_train, device

def apply_digital_twin_inference(df_raw: pd.DataFrame, ts1_mean_train: float, config: dict, apply_digital_twin: bool = False, verbose: bool = True):
    """
    Aplica el resampleo a 10Hz y, opcionalmente, el desplazamiento térmico (Gemelo Digital)
    idéntico al utilizado durante el entrenamiento.
    Por defecto NO se aplica el offset térmico, asumiendo que las temperaturas de entrada
    son las operativas reales de la planta (ej. 70-74 ºC).
    Si apply_digital_twin es True, se aplica el desplazamiento al baseline de 65 ºC
    (necesario cuando los datos provienen del dataset de laboratorio UCI).
    """
    cols_sensores_base = config['features']['cols_sensores_base']
    df_temp = df_raw.copy()
    
    if 'Time_Segundos' not in df_temp.columns:
        if 'Time' in df_temp.columns:
            df_temp['Time_Segundos'] = df_temp['Time'].round(1)
        else:
            raise KeyError("No se encontró la columna 'Time' ni 'Time_Segundos'.")

    # Agrupar por Cycle_ID si existe, de lo contrario solo por Time_Segundos
    if 'Cycle_ID' in df_temp.columns:
        df_10hz = df_temp.groupby(['Cycle_ID', 'Time_Segundos'])[cols_sensores_base].mean().reset_index()
    else:
        df_10hz = df_temp.groupby('Time_Segundos')[cols_sensores_base].mean().reset_index()
        
    df_thermo = df_10hz.copy()
    
    if apply_digital_twin:
        # Aplicar el desplazamiento térmico del Gemelo Digital al baseline de 65 ºC
        offset_temperatura = 65.0 - ts1_mean_train
        df_thermo['TS1'] = df_thermo['TS1'] + offset_temperatura
        if 'TS2' in df_thermo.columns:
            df_thermo['TS2'] = df_thermo['TS2'] + offset_temperatura
        if verbose: logger.info("Gemelo Digital aplicado: temperaturas desplazadas al baseline de 65 ºC.")
    else:
        if verbose: logger.info("Usando temperaturas reales (sin desplazamiento térmico).")
        
    return df_thermo

def run_inference(df_leche: pd.DataFrame, model, scaler, feature_cols, config: dict, device, verbose: bool = True):
    """
    Recibe el dataframe termodinámicamente corregido, genera features, 
    escala, convierte a tensor y predice.
    """
    if verbose: logger.info("Realizando feature engineering para inferencia...")
    cols_sensores_base = config['features']['cols_sensores_base']
    df_thermo = df_leche.sort_values('Time_Segundos').copy()
    
    # Feature engineering
    X_sensores = df_thermo[cols_sensores_base]
    rmean = X_sensores.rolling(5, min_periods=1).mean().add_suffix('_rmean')
    rstd = X_sensores.rolling(5, min_periods=1).std().fillna(0).add_suffix('_rstd')
    lag = X_sensores.shift(1).bfill().add_suffix('_lag1') 
    
    df_processed = pd.concat([df_thermo, rmean, rstd, lag], axis=1)

    try:
        df_to_scale = df_processed[feature_cols] 
    except KeyError as e:
        logger.error(f"Faltan características en inferencia: {e}")
        return None

    # Scaler
    X_scaled = scaler.transform(df_to_scale)

    # Tensor
    target_len = config['data']['window_size']
    if X_scaled.shape[0] < target_len:
        pad = np.zeros((target_len - X_scaled.shape[0], X_scaled.shape[1]))
        X_ready = np.vstack([X_scaled, pad])
    elif X_scaled.shape[0] > target_len:
        X_ready = X_scaled[:target_len, :]
    else:
        X_ready = X_scaled
        
    input_tensor = torch.tensor(X_ready, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

    # Inferencia
    with torch.no_grad():
        p_foul, p_valv, p_bomb, p_acum = model(input_tensor)
        probs = [torch.softmax(p, dim=1) for p in [p_foul, p_valv, p_bomb, p_acum]]
        preds = [p.argmax(1).item() for p in probs]
        confidences = [p.max(1).values.item() for p in probs]
        
    return {"predicciones": preds, "confianzas": confidences}
