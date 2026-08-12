import sys
import argparse
import yaml
from pathlib import Path

# Añadimos la raíz del proyecto al sys.path para permitir ejecuciones directas
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils.logging import get_logger

logger = get_logger("main")

def load_config(config_path: str = "config/config.yaml") -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error cargando config.yaml: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Predicción de Patologías en la Vid")
    parser.add_argument("step", choices=['train', 'predict', 'data_processing', 'get_stats'], 
                        help="Paso del pipeline a ejecutar")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Ruta al archivo config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Dataset de entrada (Opcional, sobrescribe yaml)")
    parser.add_argument("--id_serie", type=int, default=None, help="Filtra una ID específica para inferencia (testing local)")
    parser.add_argument("--metrics", action="store_true", help="Generar métricas de evaluación sobre el conjunto Test tras el entrenamiento")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 1. DATA PROCESSING
    if args.step == 'data_processing':
        from src.data_processing.preprocess import run_data_processing
        raw = args.input if args.input else config['paths']['raw_data']
        procesado_dir = config['paths']['processed_dir']
        raw_feats = config.get('raw_features', [])
        model_feats = config.get('model_features', [])
        
        logger.info("=== INICIANDO DATA PROCESSING ===")
        run_data_processing(raw, procesado_dir, raw_feats, model_feats, config)
        logger.info("=== DATA PROCESSING FINALIZADO ===")

    # 2. TRAIN
    elif args.step == 'train':
        from src.training.train import run_training
        # Necesitamos la data procesada
        processed_filename = config['output_files']['processed_data']
        procesado_ruta = args.input if args.input else str(Path(config['paths']['processed_dir']) / processed_filename)
        # Aseguramos que el resto del pipeline use model_features
        config['cols_features'] = config.get('model_features', [])
        
        logger.info("=== INICIANDO ENTRENAMIENTO (TRAIN) ===")
        run_training(procesado_ruta, config, save_metrics=args.metrics)
        logger.info("=== ENTRENAMIENTO FINALIZADO ===")
        
    # 3. PREDICT
    elif args.step == 'predict':
        from src.predict.predictor import run_inference
        # Inference recibe un CSV/Parquet de un sensor o batch nuevo
        # Default: datos crudos (disponibles sin entrenamiento previo)
        inference_data = args.input if args.input else config['paths']['raw_data']
        config['cols_features'] = config.get('model_features', [])
        
        logger.info(f"=== INICIANDO INFERENCIA (PREDICT) ===")
        run_inference(inference_data, config, id_serie=args.id_serie)
        logger.info("=== INFERENCIA FINALIZADA ===")
        
    # 4. GET STATS
    elif args.step == 'get_stats':
        from src.get_stats.column_info import run_stats
        logger.info("=== OBTENIENDO ESTADÍSTICAS Y DICCIONARIO DE VARIABLES ===")
        stats = run_stats(config)
        logger.info("=== GET STATS FINALIZADO ===")

if __name__ == "__main__":
    main()
