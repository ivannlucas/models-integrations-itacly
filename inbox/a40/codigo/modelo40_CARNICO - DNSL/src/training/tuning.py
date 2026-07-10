import pandas as pd
import joblib
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, StratifiedGroupKFold
from src.utils.logging import get_logger
import gc

logger = get_logger("TUNING")

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)
    
def reduce_mem_usage(df):
    """ 
    Reduce la memoria convirtiendo float64 a float32 e int64 a int32.
    Fundamental para sistemas con RAM limitada (< 64GB).
    """
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df

def tuning_refrigeracion():
    """Tuning siguiendo la estrategia de StratifiedGroupKFold."""
    full_cfg = load_config()
    cfg = full_cfg["refrigeracion"]
    paths = full_cfg["paths"]

    system = "refrigeracion"
    data_path = Path(paths["splits_data"]) / f"{system}_train.csv"
    
    df = pd.read_csv(data_path)
    df = reduce_mem_usage(df)
    target = cfg["target_column"]
    drop_cols = cfg.get("drop_cols", [])
    
    X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns], errors='ignore')
    y = df[target]
    groups = df['run_id']

    # Liberamos el DataFrame original para ahorrar RAM
    del df
    gc.collect()

    # Espacio de búsqueda específico
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20, 25],
        'min_samples_leaf': [10, 20, 50],
        'max_features': ['sqrt']
    }

    cv_strategy = StratifiedGroupKFold(n_splits=5)
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    logger.info(f"Iniciando RandomizedSearchCV para {system} (StratifiedGroupKFold)...")
    
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_grid,
        n_iter=5,
        cv=cv_strategy,
        scoring='f1_macro',
        n_jobs=4,
        verbose=2,
        random_state=42
    )

    rf_search.fit(X, y, groups=groups)
    
    _save_best_params(system, rf_search.best_params_, rf_search.best_score_)
    return rf_search.best_params_

def tuning_aireado():
    """Tuning siguiendo la estrategia de GroupKFold."""
    full_cfg = load_config()
    cfg = full_cfg["aireado"]
    paths = full_cfg["paths"]

    system = "aireado"
    data_path = Path(paths["splits_data"]) / f"{system}_train.csv"
    
    df = pd.read_csv(data_path)
    target = cfg["target_column"]
    drop_cols = cfg.get("drop_cols", [])
    
    X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns], errors='ignore')
    y = df[target]
    groups = df['run_id']

    # Espacio de búsqueda específico
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    gkf = GroupKFold(n_splits=5)
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    logger.info(f"Iniciando RandomizedSearchCV para {system} (GroupKFold)...")

    rf_random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=5, 
        cv=gkf,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='f1_weighted'
    )

    rf_random_search.fit(X, y, groups=groups)
    
    _save_best_params(system, rf_random_search.best_params_, rf_random_search.best_score_)
    return rf_random_search.best_params_

def _save_best_params(system, params, score):
    """Guarda los resultados en artifacts."""
    logger.info(f"Mejor Score para {system}: {score:.4f}")
    logger.info(f"Mejores parámetros: {params}")

    full_cfg = load_config()
    paths = full_cfg["paths"]

    # Uso de la ruta global 'artifacts'
    out_dir = Path(paths["artifacts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"{system}_best_params.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(params, out_path)
  

    readable_path = out_dir / f"{system}_best_params.yaml"
    with open(readable_path, "w") as f:
        yaml.dump({
            "system": system,
            "best_f1_score": float(score),
            "optimized_params": params
        }, f, default_flow_style=False)

    logger.info(f"Parámetros guardados en {out_path} y en {readable_path}")