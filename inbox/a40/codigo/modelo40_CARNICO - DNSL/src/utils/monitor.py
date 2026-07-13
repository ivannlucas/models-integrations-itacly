import pandas as pd
import datetime
from pathlib import Path
import yaml

def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def log_to_monitorization(system, data_dict):
    """
    Escribe en logs/monitorization_{system}.csv. 
    'data_dict' contiene solo las claves que quieras guardar (ej: {'drift': 0.02} o {'status': 'ESTABLE', 'confidence': 80})
    """
    cfg = load_config()
    log_path = Path(cfg["paths"]["logs"]) / f"monitorization_{system}.csv"
    
    # Añadimos timestamp a los datos
    data_dict['timestamp'] = datetime.datetime.now().isoformat()
    
    df_new = pd.DataFrame([data_dict])
    
    # Si el archivo existe, leemos y concatenamos; si no, creamos nuevo
    if log_path.exists():
        df_old = pd.read_csv(log_path)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
        
    df_final.to_csv(log_path, index=False)