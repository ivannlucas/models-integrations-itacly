import yaml
import os
import sys
import argparse

# Asegurar que el directorio raíz está en el path para permitir ejecuciones directas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logging import get_logger

logger = get_logger("Main")

def load_config(config_path="config/config.yaml"):
    logger.info(f"Cargando configuración desde {config_path}...")
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_data_processing():
    from src.data_processing.load_data import load_txt_and_convert_to_csv, load_raw_csv
    from src.data_processing.preprocess import clean_and_resample, split_data, apply_digital_twin_and_augment, feature_engineering
    
    config = load_config()
    raw_dir = config['paths']['raw_data']
    raw_csv = config['paths']['raw_csv']
    processed_csv = config['paths']['processed_data10hz']

    # 1. Cargar
    if not os.path.exists(raw_csv):
        _ = load_txt_and_convert_to_csv(raw_dir, raw_csv)
        import gc
        gc.collect()
        
    df_raw = load_raw_csv(raw_csv)
        
    if df_raw.empty:
        logger.error("No se pudo cargar la data cruda.")
        return

    # 2. Resamplear a 10Hz
    df_10hz = clean_and_resample(df_raw, config)
    os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
    df_10hz.to_csv(processed_csv, index=False)
    logger.info(f"Datos base a 10Hz guardados en {processed_csv}.")
    
    # Nota: Los pasos de Split, Gemelo Digital y Augmentation 
    # se ejecutan al momento de entrenar para evitar data leakage en disco.
    logger.info("Fase de procesamiento estructural completada de forma exitosa.")


def run_train(metrics: bool = False):
    from src.data_processing.load_data import load_raw_csv
    from src.data_processing.preprocess import split_data, apply_digital_twin_and_augment, feature_engineering, prepare_tensors
    from src.training.trainer import train_model
    from src.utils.reproducibility import seed_everything
    import pandas as pd
    import os
    import joblib
    
    config = load_config()
    processed_csv = config['paths']['processed_data10hz']
    splits_dir = config['paths'].get('splits_dir', 'data/splits/')
    
    # Reproducibilidad
    seed = config.get('data', {}).get('random_state', 42)
    seed_everything(seed)

    logger.info("Iniciando pipeline de entrenamiento...")
    
    # LÓGICA DE FALLBACK: Si no existe el processed_csv, buscar en splits_dir
    if not os.path.exists(processed_csv):
        train_csv = os.path.join(splits_dir, "train_split.csv")
        val_csv = os.path.join(splits_dir, "val_split.csv")
        test_csv = os.path.join(splits_dir, "test_split.csv")
        
        if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
            logger.warning(f"No se encontró {processed_csv}. Activando modo FALLBACK usando particiones en {splits_dir}...")
            df_train = load_raw_csv(train_csv)
            df_val = load_raw_csv(val_csv)
            df_test = load_raw_csv(test_csv)
            
            # Recuperar ts1_mean_train guardado previamente (necesario para el gemelo digital en inferencia)
            # Si no existe, aproximamos un valor típico de 45.0 para no romper el pipeline.
            ts1_path = os.path.join(config['paths']['models_dir'], "ts1_mean_train.pkl")
            if os.path.exists(ts1_path):
                ts1_mean_train = joblib.load(ts1_path)
            else:
                logger.warning(f"No se encontró {ts1_path}. Usando valor por defecto 45.0 para ts1_mean_train.")
                ts1_mean_train = 45.0
                
            df_final = pd.concat([df_train, df_val, df_test], ignore_index=True)
            train_cycles_aug = df_train['Cycle_ID'].unique()
            val_cycles = df_val['Cycle_ID'].unique()
            test_cycles = df_test['Cycle_ID'].unique()
            
            logger.info("Particiones cargadas correctamente. Saltando el pipeline de preprocesamiento inicial...")
        else:
            logger.error(f"No se encontró {processed_csv} ni los CSVs de partición en {splits_dir}. Imposible entrenar.")
            return
    else:
        df_10hz = load_raw_csv(processed_csv)
        
        # Pipeline completo
        train_cycles, val_cycles, test_cycles = split_data(df_10hz, config)
        df_aug, train_cycles_aug, ts1_mean_train = apply_digital_twin_and_augment(df_10hz, train_cycles, config)
        df_final = feature_engineering(df_aug, config)

        from src.data_processing.preprocess import save_processed_splits
        save_processed_splits(df_final, train_cycles_aug, val_cycles, test_cycles, config)
    
    # Preparar tensores
    train_loader, val_loader, test_loader, scaler, feature_cols = prepare_tensors(
        df_final, train_cycles_aug, val_cycles, test_cycles, config
    )
    
    # Entrenar
    train_model(train_loader, val_loader, test_loader, scaler, feature_cols, ts1_mean_train, config, save_metrics=metrics)
    logger.info("Entrenamiento finalizado y artefactos exportados correctamente.")


def run_predict(input_csv_path: str = None, output_csv_path: str = None, target_cycle=None, apply_digital_twin=False):
    from src.data_processing.load_data import load_raw_csv
    from src.predict.predictor import load_artifacts, apply_digital_twin_inference, run_inference
    from src.predict.postprocess import format_output, save_predictions
    import pandas as pd
    
    config = load_config()
    models_dir = config['paths']['models_dir']
    
    input_csv_path = input_csv_path or config['paths'].get('prediction_input', "data/raw/hydraulic_raw.csv")
    output_csv_path = output_csv_path or config['paths'].get('prediction_output', "data/predictions/prediction_output.csv")
    
    logger.info(f"Iniciando predicción usando datos de {input_csv_path}")
    df_leche = load_raw_csv(input_csv_path)
    
    # Filtrar por ciclo si se solicita
    if target_cycle is not None:
        if 'Cycle_ID' in df_leche.columns:
            logger.info(f"Filtrando por el Cycle_ID: {target_cycle}...")
            df_leche = df_leche[df_leche['Cycle_ID'] == int(target_cycle)]
            if df_leche.empty:
                logger.error(f"El Cycle_ID {target_cycle} no existe en los datos cargados.")
                return
        else:
            logger.warning("No se puede filtrar por ciclo porque la columna 'Cycle_ID' no existe.")

    # Extraer Cycle_IDs
    if 'Cycle_ID' in df_leche.columns:
        cycle_ids = df_leche['Cycle_ID'].unique()
    else:
        cycle_ids = ["Desconocido"]

    # Cargar y preprocesar modelo
    model, scaler, feature_cols, ts1_mean_train, device = load_artifacts(models_dir, config)
    
    all_preds = []
    logger.info(f"Procesando predicciones para {len(cycle_ids)} ciclo(s)...")

    verbose_mode = True if len(cycle_ids) == 1 else False

    if not verbose_mode:
        from tqdm import tqdm
        loop_iterable = tqdm(cycle_ids, desc="Analizando ciclos", unit="ciclo")
    else:
        loop_iterable = cycle_ids

    for cid in loop_iterable:
        if cid != "Desconocido":
            df_cycle = df_leche[df_leche['Cycle_ID'] == cid].copy()
        else:
            df_cycle = df_leche.copy()

        # Extraer "Ground Truth" si viene en el dataframe original
        real_targets = None
        config_targets = config['features']['cols_targets']
        raw_targets = ['Cooler_Condition', 'Valve_Condition', 'Pump_Leakage', 'Hydraulic_Accumulator']
        
        if all(t in df_cycle.columns for t in raw_targets):
            from src.utils.target_mapping import map_cooler, map_valve, map_pump, map_acc
            
            row = df_cycle.iloc[0]
            real_targets = [
                map_cooler(row['Cooler_Condition']),
                map_valve(row['Valve_Condition']),
                map_pump(row['Pump_Leakage']),
                map_acc(row['Hydraulic_Accumulator'])
            ]
        elif all(t in df_cycle.columns for t in config_targets):
            real_targets = df_cycle[config_targets].iloc[0].values.astype(int).tolist()

        # Gemelo Digital en inferencia
        df_thermo = apply_digital_twin_inference(df_cycle, ts1_mean_train, config, apply_digital_twin=apply_digital_twin, verbose=verbose_mode)
        
        # Inferencia
        resultados = run_inference(df_thermo, model, scaler, feature_cols, config, device, verbose=verbose_mode)
        
        if resultados is not None:
            # Formateo
            df_pred = format_output(resultados, cycle_id=cid, real_targets=real_targets, verbose=verbose_mode)
            all_preds.append(df_pred)

    if all_preds:
        df_pred_final = pd.concat(all_preds, ignore_index=True)
        save_predictions(df_pred_final, output_csv_path)
    else:
        logger.error("No se pudieron generar predicciones.")


def run_fine_tuning(
    train_csv: str,
    val_csv: str,
    fluid_density_kg_l: float = None,
    fluid_cp_kj_kgK: float = None,
    ft_epochs: int = None,
    ft_patience: int = None,
):
    """Calibra el modelo DNSL preentrenado a datos reales de planta.

    Congela el backbone CNN y re-entrena únicamente las cabezas de
    clasificación usando ciclos reales del fluido de producción.
    El modelo base no se sobreescribe; se genera un artefacto nuevo:
    models/artifacts/neurosymbolic_cnn_finetuned.pth
    """
    from src.fine_tuning.fine_tuner import run_fine_tuning as _run_ft

    config = load_config()
    _run_ft(
        train_csv=train_csv,
        val_csv=val_csv,
        config=config,
        fluid_density_kg_l=fluid_density_kg_l,
        fluid_cp_kj_kgK=fluid_cp_kj_kgK,
        ft_epochs=ft_epochs,
        ft_patience=ft_patience,
    )
    logger.info("Fine-tuning/calibración finalizado correctamente.")


def run_get_stats():
    from src.data_processing.load_data import load_raw_csv
    from src.get_stats.column_info import (
        generate_stats, generate_feature_descriptions, 
        print_model_info, load_trained_metrics
    )
    
    config = load_config()
    processed_csv = config['paths']['processed_data10hz']
    
    logger.info("Iniciando generación de estadísticas completas...")
    df_10hz = load_raw_csv(processed_csv)
    
    # 1. Descripción del modelo y su propósito
    print_model_info()
    
    # 2. Listado de features con descripción
    features_df = generate_feature_descriptions(df_10hz)
    features_path = config['paths'].get('features_csv', "data/processed/feature_descriptions.csv")
    features_df.to_csv(features_path, index=False)
    logger.info(f"Listado de features exportado a {features_path}.")
    
    # 3. Estadísticas descriptivas del dataset
    stats = generate_stats(df_10hz)
    stats_path = config['paths'].get('stats_csv', "data/processed/estadisticas.csv")
    stats.to_csv(stats_path)
    logger.info(f"Estadísticas descriptivas exportadas a {stats_path}.")
    
    # 4. Métricas del modelo entrenado (si existen)
    metrics_dir = config['paths'].get('metrics_dir', 'models/metrics')
    metrics = load_trained_metrics(metrics_dir)
    
    if metrics:
        logger.info("=" * 80)
        logger.info("MÉTRICAS DEL MODELO ENTRENADO")
        logger.info("=" * 80)
        for name, df_metric in metrics.items():
            logger.info(f"\n--- {name} ---")
            logger.info(f"\n{df_metric.to_string(index=False)}")
        logger.info("=" * 80)
    
    logger.info("Generación de estadísticas completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline del Modelo DNSL.")
    parser.add_argument("mode", choices=["data_processing", "train", "predict", "get_stats", "fine_tune"], help="Modo de ejecución.")
    parser.add_argument("--input", type=str, help="Ruta de entrada para predicción (opcional).", default=None)
    parser.add_argument("--output", type=str, help="Ruta de salida para predicción (opcional).", default=None)
    parser.add_argument("--metrics", action="store_true", help="Si se incluye, se calculan y guardan las métricas en Test durante el entrenamiento.")
    parser.add_argument("--cycle", type=int, default=None, help="Si se incluye al predecir o extraer, aislará ese Cycle_ID en específico.")
    parser.add_argument("--apply_digital_twin", action="store_true", help="Si está activo en modo predict, aplica el desplazamiento térmico del Gemelo Digital al baseline de 65ºC. Útil para datos de laboratorio (UCI). Por defecto se usan las temperaturas reales.")
    parser.add_argument("--train_input", type=str, default=None, help="(fine_tune) Ruta al CSV de ciclos de planta para entrenamiento del fine-tuning.")
    parser.add_argument("--val_input", type=str, default=None, help="(fine_tune) Ruta al CSV de ciclos de planta para validación del fine-tuning.")
    parser.add_argument("--fluid_density", type=float, default=None, help="(fine_tune) Densidad del fluido real en kg/L. Si no se indica, se usa el valor del config.yaml (por defecto 1.03, leche).")
    parser.add_argument("--fluid_cp", type=float, default=None, help="(fine_tune) Calor específico del fluido real en kJ/(kg·K). Si no se indica, se usa el valor del config.yaml (por defecto 3.93, leche).")
    parser.add_argument("--ft_epochs", type=int, default=None, help="(fine_tune) Épocas máximas de calibración. Si no se indica, se usa el valor del config.yaml (por defecto 50).")
    parser.add_argument("--ft_patience", type=int, default=None, help="(fine_tune) Paciencia del Early Stopping del fine-tuning. Si no se indica, se usa el valor del config.yaml (por defecto 7).")

    args = parser.parse_args()
    
    if args.mode == "data_processing":
        run_data_processing()
    elif args.mode == "train":
        run_train(metrics=args.metrics)
    elif args.mode == "predict":
        run_predict(args.input, args.output, target_cycle=args.cycle, apply_digital_twin=args.apply_digital_twin)
    elif args.mode == "get_stats":
        run_get_stats()
    elif args.mode == "fine_tune":
        if not args.train_input or not args.val_input:
            parser.error("El modo 'fine_tune' requiere --train_input y --val_input.")
        run_fine_tuning(
            train_csv=args.train_input,
            val_csv=args.val_input,
            fluid_density_kg_l=args.fluid_density,
            fluid_cp_kj_kgK=args.fluid_cp,
            ft_epochs=args.ft_epochs,
            ft_patience=args.ft_patience,
        )
