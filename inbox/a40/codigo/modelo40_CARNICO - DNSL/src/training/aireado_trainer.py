import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.utils.noise import apply_stress_test

def load_config():
    """Carga el archivo de configuración global."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_aireado(df):
    """
    Entrena el modelo final de Aireado usando los mejores parámetros del tuning.
    Sin escalador, tal como se definió en el notebook original.
    """
    full_cfg = load_config()
    # Usamos la llave 'refrigeracion' para lógica y 'paths' para rutas
    cfg = full_cfg["aireado"]
    paths = full_cfg["paths"]
    stress_cfg = cfg.get("stress_test", {})

    # 1. Preparación de columnas
    target = 'fault_id'
    # Excluimos metadatos y la columna de texto 'fault'
    features = [col for col in df.columns if col not in ['run_id', 'time_min', 'fault_id', 'fault']]
    
    # 2. STRATIFIED SPLIT BY RUN
    run_info = df.groupby('run_id')[target].first()
    unique_runs = run_info.index
    
    train_runs, test_runs = train_test_split(
        unique_runs, 
        test_size=0.2, 
        random_state=42,
        stratify=run_info
    )

    train_df = df[df['run_id'].isin(train_runs)].copy()
    test_df  = df[df['run_id'].isin(test_runs)].copy()

    X_train = train_df[features]
    y_train = train_df[target]
    X_test  = test_df[features]
    y_test  = test_df[target]

    if stress_cfg.get("active", False):
        print(f"⚠️ Aplicando Stress Test en Aireado (Ruido: {stress_cfg.get('noise_level')*100}%)")
        X_test_processed = apply_stress_test(X_test, stress_cfg)
        # Actualizamos las columnas alteradas en el dataframe que se exportará
        test_df[features] = X_test_processed
    else:
        X_test_processed = X_test.copy()

    # ENTRENAMIENTO
    print("Entrenando modelo de Aireado...")

    params = cfg.get("model_params", {})
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    #  GUARDADO (Rutas del YAML)
    art_path = Path(paths['artifacts'])
    art_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, art_path / "aireado_model.pkl")
   

    split_path = Path(paths['splits_data'])
    split_path.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(split_path / "aireado_test.csv", index=False)
    train_df.to_csv(split_path / "aireado_train.csv", index=False)

    stats = {
        'mean': X_train.mean().to_dict(),
        'std': X_train.std().to_dict()
    }
    with open(art_path / "aireado_stats.yaml", "w") as f:
        yaml.dump(stats, f)
    
    print("✅ Entrenamiento de aireado y guardado de estadísticas completado.")

    print(f"Modelo guardado y set de test listo.")

    # Devolvemos None en el lugar del scaler para mantener la firma (model, scaler)
    return model, None