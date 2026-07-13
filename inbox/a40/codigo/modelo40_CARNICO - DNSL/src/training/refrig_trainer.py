import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from src.utils.noise import apply_stress_test

def load_config():
    """Carga el archivo de configuración global."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_sample_weights(df, target_col='fault_numeric'):
    """
    Calcula pesos de muestra balanceados y aplica multiplicadores basados
    en heurísticas de presión para penalizar errores en condiciones críticas.
    """
    classes = np.unique(df[target_col])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=df[target_col]
    )
    class_weight_dict = dict(zip(classes, class_weights))
    
    # .copy() asegura que el array sea mutable
    weights = df[target_col].map(class_weight_dict).values.copy()

    # Umbrales basados en percentiles para identificar condiciones de alta carga/estrés
    p_dis_threshold = df["mean_P_dis_bar"].quantile(0.90)
    early_error_threshold = df["early_P_dis_error"].quantile(0.90)

    # Aplicar multiplicadores expertos: 
    # Priorizamos el aprendizaje en situaciones de error temprano (x1.5) y presiones altas (x1.2)
    weights *= np.where(df["early_P_dis_error"] > early_error_threshold, 1.5, 1.0)
    weights *= np.where(df["mean_P_dis_bar"] > p_dis_threshold, 1.2, 1.0)
    
    return weights

def train_refrigeration(df, config):
    """
    Entrena el modelo final de Refrigeración usando fault_id directamente.
    """
    full_cfg = load_config()
    # Usamos la llave 'refrigeracion' para lógica y 'paths' para rutas
    cfg = full_cfg["refrigeracion"]
    paths = full_cfg["paths"]
    stress_cfg = cfg.get("stress_test", {})
    # 1. CARGA DE ETIQUETAS (Usando directamente los números de fault_id)
    target_raw = config.get("target_column", "fault_id")
    print(f"DEBUG: Cargando etiquetas numéricas desde: {target_raw}")
    
    # Convertimos a numérico por si vienen como strings ("0", "1"...)
    df['fault_numeric'] = pd.to_numeric(df[target_raw], errors='coerce')
    
    # Limpieza de nulos (filas vacías al final del CSV o errores de carga)
    nans = df['fault_numeric'].isna().sum()
    if nans > 0:
        print(f"Aviso: Se encontraron {nans} filas nulas. Eliminándolas...")
        df = df.dropna(subset=['fault_numeric']).copy()
    
    df['fault_numeric'] = df['fault_numeric'].astype(int)

    # Verificación de seguridad
    if len(df) == 0:
        raise ValueError(f"ERROR: No hay datos para entrenar en la columna {target_raw}")

    # 2. STRATIFIED SPLIT BY RUN
    # Agrupamos por run para que el modelo no vea datos de la misma nevera en train y test
    run_labels = df.groupby('run_id')['fault_numeric'].first()
    unique_runs = run_labels.index

    train_runs, test_runs = train_test_split(
        unique_runs, 
        test_size=0.2, 
        random_state=42,
        stratify=run_labels 
    )

    train_df = df[df['run_id'].isin(train_runs)].copy()
    test_df  = df[df['run_id'].isin(test_runs)].copy()

    # PREPARACIÓN DE VARIABLES
    target_column = 'fault_numeric'
    # Quitamos todas las columnas que no son sensores/features
    drop_cols = [target_column, 'fault', 'run_id', 'fault_id', 'time_min', 
                 'T_cond_sat', 'T_cab_meas', 'P_suc_bar', 'fault_numeric']
    
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_test = test_df[target_column]

    if stress_cfg.get("active", False):
        print(f"⚠️ Aplicando Stress Test en Refrigeración (Ruido: {stress_cfg.get('noise_level')*100}%)")
        X_test_processed = apply_stress_test(X_test, stress_cfg)
        # Actualizamos la estructura nativa del dataframe que guardará el split
        features_names = X_test.columns
        test_df[features_names] = X_test_processed
    else:
        X_test_processed = X_test.copy()

    # CÁLCULO DE PESOS (Para clases desbalanceadas)
    sample_weights = compute_sample_weights(train_df)

    # ESCALADO
    binary_cols = ['defrost_on', 'door_open']
    numerical_cols = [c for c in X_train.columns if c not in binary_cols]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test_processed.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # ENTRENAMIENTO 
    print(f"Entrenando modelo final con {X_train_scaled.shape[0]} muestras...")

    params = cfg.get("model_params", {})
    model = RandomForestClassifier(**params)
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)    

    print(f"Entrenamiento terminado")

    #  GUARDADO (Rutas del YAML)
    art_path = Path(paths['artifacts'])
    art_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, art_path / "refrigeracion_model.pkl")
    joblib.dump(scaler, art_path / "refrigeracion_scaler.pkl")

    split_path = Path(paths['splits_data'])
    split_path.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(split_path / "refrigeracion_test.csv", index=False)
    train_df.to_csv(split_path / "refrigeracion_train.csv", index=False)
    
    print("✅ Entrenamiento de Refrigeración completado con éxito.")

    stats = {
        'mean': X_train.mean().to_dict(),
        'std': X_train.std().to_dict()
    }
    with open(art_path / "refrigeracion_stats.yaml", "w") as f:
        yaml.dump(stats, f)
    
    print("✅ Estadísticas de Refrigeración guardadas para detección de drift.")
    return model, scaler

    return model, scaler