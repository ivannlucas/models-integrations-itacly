import logging
import sys
import yaml
from pathlib import Path

def _load_log_path():
    """Carga la ruta de logs desde el config.yaml de forma segura."""
    config_path = Path("config/config.yaml")
    default_path = Path("logs")
    
    if not config_path.exists():
        return default_path
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            # Accedemos a paths -> logs en el YAML
            log_dir_str = config.get("paths", {}).get("logs", "logs")
            return Path(log_dir_str)
    except Exception:
        return default_path

def get_logger(name: str):
    """
    Configura y retorna un logger que obtiene su ruta desde el config.yaml.
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Formato profesional: [Fecha] [Nivel] [Módulo] -> Mensaje
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s -> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Handler para Consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Handler para Archivo (Ruta desde Config)
    log_dir = _load_log_path()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(
        filename=log_dir / "pipeline_mantenimiento.log", 
        mode='a', 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger