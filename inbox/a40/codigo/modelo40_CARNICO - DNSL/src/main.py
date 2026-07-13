import sys
import yaml
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import datetime

# 1. Utilidades y Logging
from src.utils.logging import get_logger
from src.utils.monitor import log_to_monitorization
from src.get_stats.column_info import get_column_summary, check_data_leakage, get_feature_groups

# 2. Procesamiento de datos
from src.data_processing.load_data import load_csv_data, save_processed_data
from src.data_processing.preprocess import extract_refrigeration_features, create_refrigeration_lags, extract_aireado_features, create_aireado_lags, physics_indicators_refrigeration

# 3. Entrenamiento
from src.training.refrig_trainer import train_refrigeration
from src.training.aireado_trainer import train_aireado

from src.training.tuning import tuning_refrigeracion, tuning_aireado

# 4. Inferencia y Post-proceso
from src.predict.predictor import load_model, run_inference, save_predictions, monitor_model_health
from src.predict.postprocess import apply_neurosymbolic_rules, apply_run_voting


# Inicializar Logger
logger = get_logger("MAIN")

def load_config(config_path="config/config.yaml"):
    """Carga la configuración central del proyecto."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo de configuración en {config_path}")
        sys.exit(1)

def apply_feature_engineering(df, system):
    """Aplica la ingeniería de variables según el sistema seleccionado."""
    if system == "refrigeracion":
        df = extract_refrigeration_features(df)
        df = create_refrigeration_lags(df)
        df = physics_indicators_refrigeration(df)
    elif system == "aireado":
        df = extract_aireado_features(df)
        df = create_aireado_lags(df)
    return df

def data_processing():
    """Pipeline de preprocesamiento y feature engineering."""
    cfg = load_config()
    system = cfg["selected_system"]
    logger.info(f"--- Iniciando Procesamiento: {system.upper()} ---")
    
    # Cargar datos usando el nuevo módulo de carga
    raw_path = cfg[system]["raw_data"]
    df = load_csv_data(raw_path)
    
    df = apply_feature_engineering(df, system)

    # Guardar usando el nuevo módulo
    out_path = Path(cfg["paths"]["processed_data"]) / f"{system}_final.csv"
    save_processed_data(df, out_path)
    logger.info(f"Preprocesamiento finalizado para {system}")

def train():
    """Pipeline de entrenamiento y optimización."""
    cfg = load_config()
    system = cfg["selected_system"]
    logger.info(f"--- Iniciando Entrenamiento: {system.upper()} ---")
    
    data_path = Path(cfg["paths"]["processed_data"]) / f"{system}_final.csv"
    if not data_path.exists():
        logger.error("No existen datos procesados. Ejecute 'data_processing' primero.")
        return

    df = pd.read_csv(data_path)
    
    if system == "refrigeracion":
        model, scaler = train_refrigeration(df, cfg["refrigeracion"])
        joblib.dump(scaler, f"models/artifacts/{system}_scaler.pkl")
        logger.info(f"Escalador de {system} guardado.")
    elif system == "aireado":
        model, _ = train_aireado(df)
    
    # Guardar modelo
    model_path = f"models/artifacts/{system}_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Modelo de {system} guardado exitosamente.")

def tuning():
    # Cargamos el config para saber qué sistema tunear
        cfg = load_config()
        system = cfg.get("selected_system", "refrigeracion")
        
        logger.info(f"--- Iniciando Optimización de Hiperparámetros: {system.upper()} ---")
        
        if system == "refrigeracion":
            best_params = tuning_refrigeracion()
        elif system == "aireado":
            best_params = tuning_aireado()
        else:
            logger.error(f"Sistema '{system}' no reconocido para tuning.")
            return

def predict():
    """Pipeline de inferencia con carga atómica y acoplada de artefactos (Modelo, Escalador y Reglas)."""
    cfg = load_config()
    system = cfg["selected_system"]
    paths = cfg["paths"]
    art_path = Path(paths["artifacts"])
    
    # ==========================================
    # 1. CARGA DE DATOS Y PROCESAMIENTO
    # ==========================================
    input_cliente = Path(paths["to_predict"]) / f"input_{system}.csv"
    if input_cliente.exists():
        logger.info(f"Procesando datos crudos del cliente para {system}...")
        df_raw = pd.read_csv(input_cliente)
        df_test = apply_feature_engineering(df_raw, system)
    else:
        logger.info("Usando set de TEST por defecto...")
        data_path = Path(paths["splits_data"]) / f"{system}_test.csv"
        df_test = pd.read_csv(data_path)
    
    # ==========================================
    # 2. SELECCIÓN INTERACTIVA Y CARGA DE MODELO
    # ==========================================
    # Delegamos el menú y la lógica de carpetas a predictor.load_model
    model, selected_suffix = load_model(system)

    # ==========================================
    # 3. INFERENCIA TÉCNICA (ML puro o Calibrado + Escalado alineado)
    # ==========================================
    # 'run_inference' se encarga internamente del escalado usando el suffix para buscar las stats correctas
    y_ml, y_probs = run_inference(model, df_test, system, selected_suffix=selected_suffix)
    
    # ==========================================
    # 4. POST-PROCESO EXPERTO (Umbrales alineados por Fecha)
    # ==========================================
    mapping = cfg[system]["mapping"]
    
    # Redirigimos la ruta de umbrales en base a lo que se seleccionó de forma interactiva
    if selected_suffix:
        thresh_to_load = art_path / f"{system}_thresholds_calibrated_{selected_suffix}.yaml"
    else:
        thresh_to_load = art_path / f"{system}_thresholds.yaml"
        
    logger.info(f"Aplicando reglas físicas con umbrales desde: {thresh_to_load.name}")
    y_ns = apply_neurosymbolic_rules(df_test, y_ml, mapping, system, thresh_path=thresh_to_load)
    y_final = apply_run_voting(df_test, y_ns)
    
    # ==========================================
    # 5. PREPARAR Y GUARDAR RESULTADOS
    # ==========================================
    df_test['prediction'] = y_final
    confidences = (y_probs.max(axis=1) * 100).round(2)
    df_test['confidence'] = confidences

    avg_conf = confidences.mean()
    # monitor_model_health se encarga internamente de loguear a monitorization_*.csv
    health_status = monitor_model_health(avg_conf) 
    df_test['model_health'] = health_status
    
    # Salva y colapsa los datos por run_id
    save_predictions(df_test, system)
    logger.info("Evaluation finalizada con éxito.")

    

def evaluate():
    """
    Evalúa el rendimiento del modelo en tres niveles: 
    ML Puro, Post-proceso Neurosimbólico y Voto Temporal.
    """
    cfg = load_config()
    system = cfg["selected_system"]
    paths = cfg["paths"]
    system = cfg["selected_system"]
    logger.info(f"--- Iniciando Evaluación Comparativa: {system.upper()} ---")
    
    # 1. Cargar el set de test
    data_path = Path(paths["splits_data"]) / f"{system}_test.csv"
    if not data_path.exists():
        logger.error(f"Archivo de test no encontrado en {data_path}. Ejecuta 'train' primero.")
        return

    df_test = pd.read_csv(data_path)
    target_col = cfg[system].get("target_column", "fault_id")

    # VALIDACIÓN CRÍTICA: ¿Tenemos etiquetas reales?
    if target_col not in df_test.columns:
        logger.error(f"ERROR: La columna de target '{target_col}' no existe en el set de test.")
        logger.error("La evaluación solo es posible si los datos están etiquetados.")
        return

    y_true = df_test[target_col]
    mapping = cfg[system]["mapping"]
    present_classes = sorted(y_true.unique())
    target_names = [mapping[i] for i in present_classes]

    # 2. Obtener Predicciones
    model,_ = load_model(system)
    
    # Nivel 1: ML Puro
    y_ml, _ = run_inference(model, df_test, system)
    
    # Nivel 2: Neurosimbólico (Física corrigiendo a la IA)
    y_ns = apply_neurosymbolic_rules(df_test, y_ml, mapping, system)
    
    # Nivel 3: NS + Voto por Ciclo (Estabilidad temporal)
    y_final = apply_run_voting(df_test, y_ns)

    # 3. Generar Reportes
    print(f"\n" + "="*60)
    print(f"REPORTES DE RENDIMIENTO - SISTEMA: {system.upper()}")
    print("="*60)

    print("\nMODELO BASE (MACHINE LEARNING PURO)")
    print("-" * 60)
    print(classification_report(y_true, y_ml, labels = present_classes, target_names=target_names))

    print("\nMODELO CON POST-PROCESO NEUROSIMBÓLICO")
    print("-" * 60)
    print(classification_report(y_true, y_ns, labels = present_classes, target_names=target_names))

    print("\nMODELO FINAL (NS + VOTO POR CICLO)")
    print("-" * 60)
    print(classification_report(y_true, y_final, labels = present_classes, target_names=target_names))

    # 3. EXPORTACIÓN DE MÉTRICAS
    metrics_path = Path(paths["metrics"])
    metrics_path.mkdir(parents=True, exist_ok=True)

    # --- A. Guardar Reporte de Clasificación (Texto) ---
    report_file = metrics_path / f"classification_report_{system}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"REPORTE DE RENDIMIENTO - SISTEMA: {system.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write("MODELO FINAL (NS + VOTO POR CICLO)\n")
        f.write("-" * 60 + "\n")
        f.write(classification_report(y_true, y_final, labels=present_classes, target_names=target_names))
    
    logger.info(f"Reporte de clasificación guardado en: {report_file}")

    # --- B. Generar y Guardar Matriz de Confusión (Imagen con Números Absolutos Grandes) ---
    plt.figure(figsize=(13, 11)) # Aumentamos ligeramente el lienzo para dar espacio al texto grande
    cm = confusion_matrix(y_true, y_final, labels=present_classes)
    
    # annot_kws={'size': 14} hace que los números de dentro de las celdas sean grandes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={'size': 13}) # Números internos grandes y en negrita
    
    # Ajustamos el tamaño de las etiquetas de los ejes (nombres de las clases)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Ajustamos los títulos principales del gráfico
    plt.title(f'Matriz de Confusión - {system.upper()} (Modelo Final)', fontsize=16, weight='bold', pad=20)
    plt.ylabel('Realidad', fontsize=14, weight='bold', labelpad=10)
    plt.xlabel('Predicción', fontsize=14, weight='bold', labelpad=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0) # Mantiene los nombres de la izquierda en horizontal para leerlos mejor
    plt.tight_layout()
    
    plot_file = metrics_path / f"confusion_matrix_{system}.png"
    plt.savefig(plot_file, dpi=300) 
    plt.close()
    
    logger.info(f"Matriz de confusión guardada en: {plot_file}")
    
    logger.info("Evaluación comparativa completada con éxito.")


def calibrate():
    """
    Realiza calibración técnica de probabilidades, registra deriva y actualiza umbrales dinámicos
    sobre datos nuevos, manteniendo el modelo base intacto y versionado con marcas de tiempo
    para cumplir estrictamente con los requerimientos de la auditoría.
    """
    import datetime
    import yaml
    import joblib
    import pandas as pd
    from pathlib import Path
    from sklearn.calibration import CalibratedClassifierCV

    cfg = load_config()
    system = cfg["selected_system"]
    
    # 1. Definición de Rutas desde config.yaml
    art_path = Path(cfg["paths"]["artifacts"])
    processed_path = Path(cfg["paths"]["processed_data"]) / f"real_data_processed_{system}.csv"
    real_data_path = Path(f"data/raw/real_data_{system}.csv")
    
    thresh_base_path = art_path / f"{system}_thresholds.yaml"
    stats_base_path = art_path / f"{system}_stats.yaml"
    scaler_path = art_path / f"{system}_scaler.pkl" 
    
    if not real_data_path.exists():
        logger.error(f"No se encontró {real_data_path} para calibrar.")
        return

    # 2. Procesamiento de datos (Feature Engineering esencial)
    logger.info(f"--- Iniciando Calibración para {system} ---")
    df_raw = pd.read_csv(real_data_path)
    df_real = apply_feature_engineering(df_raw, system)
    save_processed_data(df_real, processed_path)
    
    # 3. Registro de Deriva
    if not stats_base_path.exists():
        logger.error(f"No se encontró el archivo de estadísticas base {stats_base_path} para calcular la deriva.")
        return
        
    stats_base = yaml.safe_load(open(stats_base_path))
    features_entrenamiento = list(stats_base['mean'].keys())
    available_features = [f for f in features_entrenamiento if f in df_real.columns]
    
    drift = (df_real[available_features].mean() - pd.Series(stats_base['mean'])[available_features]).abs().mean()
    log_to_monitorization(system, {'drift': drift}) 
    logger.info(f"Nivel de deriva registrado en monitorization_{system}.csv: {drift:.4f}")

    # ========================================================================
    # 4. GESTIÓN CONDICIONAL DEL ESCALADOR (Solo Refrigeración)
    # ========================================================================
    if system == "refrigeracion" and scaler_path.exists():
        logger.info("Cargando y actualizando el escalador original para REFRIGERACIÓN...")
        scaler = joblib.load(scaler_path)
        
        # Identificamos las columnas binarias que NO deben entrar al escalador
        binary_cols = ['defrost_on', 'door_open']
        
        # Filtramos 'available_features' para quedarnos SOLO con las numéricas que el escalador conoce
        features_numericas = [f for f in available_features if f not in binary_cols]
        
        # Nos aseguramos de que mantengan el orden exacto que espera scikit-learn
        if hasattr(scaler, 'feature_names_in_'):
            features_finales = [f for f in scaler.feature_names_in_ if f in df_real.columns]
        else:
            features_finales = features_numericas

        df_para_escalar = df_real[features_finales]
        
        # Ejecutamos el fit parcial con las columnas numéricas limpias
        if hasattr(scaler, 'partial_fit'):
            scaler.partial_fit(df_para_escalar)
        else:
            scaler.fit(df_para_escalar)
            
        # Generamos el DataFrame transformado manteniendo solo el espacio numérico
        X_calibracion = pd.DataFrame(scaler.transform(df_para_escalar), columns=features_finales)
        
        # Reinyectamos el resto de columnas de df_real (incluyendo defrost_on, door_open y fault_id) sin tocar sus valores
        for col in df_real.columns:
            if col not in X_calibracion.columns:
                X_calibracion[col] = df_real[col].values
                
        joblib.dump(scaler, scaler_path)
        logger.info("Escalador de refrigeración actualizado y guardado.")
    else:
        if system == "refrigeracion":
            logger.warning(f"No se detectó el archivo del escalador en {scaler_path} a pesar de ser refrigeración.")
        X_calibracion = df_real[available_features]

    # 5. ESTRATEGIA DE GUARDADO SEGÚN VOLUMEN DE DATOS
    min_samples = cfg[system].get("min_samples_for_calibration", 100)
    fecha_hoy = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Diccionario base para actualizar umbrales (Heredamos del base físico)
    current_thresh = yaml.safe_load(open(thresh_base_path)) if thresh_base_path.exists() else {}
    
    # Estructura de las nuevas estadísticas calculadas
    new_stats = {
        'mean': df_real[available_features].mean().to_dict(),
        'std': df_real[available_features].std().to_dict()
    }

    # Recalculamos los umbrales basados en la física de los nuevos datos
    if system == "aireado":
        if 'Encostramiento_Risk' in df_real.columns:
            current_thresh['encostramiento_risk'] = float(df_real['Encostramiento_Risk'].quantile(0.90))
        if 'N_fan_Hz' in df_real.columns:
            current_thresh['fan_fail_hz'] = float(df_real['N_fan_Hz'].quantile(0.05))
            
    elif system == "refrigeracion":
        if 'early_P_dis_error' in df_real.columns:
            current_thresh['nc_low'] = float(df_real['early_P_dis_error'].quantile(0.1))
        if 'T_cond_approach' in df_real.columns:
            current_thresh['cf_high'] = float(df_real['T_cond_approach'].quantile(0.9))
        if 'P_suc_bar' in df_real.columns:
            current_thresh['uc_p_gate'] = float(df_real['P_suc_bar'].quantile(0.1))
        if 'T_cab_meas_diff' in df_real.columns:
            current_thresh['drift_limit'] = float(df_real['T_cab_meas_diff'].std() * 3)
        if 'Eff_vol' in df_real.columns:
            current_thresh['eff_vol_limit'] = float(df_real['Eff_vol'].quantile(0.05))

    # ---- bifurcación de guardado ----
    if len(df_real) < min_samples:
        logger.warning(f"⚠️ Muestras insuficientes ({len(df_real)} < {min_samples}). NO se alterará el modelo de ML para evitar Overfitting.")
        logger.info("Actualizando exclusivamente los artefactos base genéricos (Post-proceso físico).")
        
        # Guardamos directamente en las rutas base (sin fecha)
        with open(stats_base_path, 'w') as f:
            yaml.dump(new_stats, f)
        with open(thresh_base_path, 'w') as f:
            yaml.dump(current_thresh, f)
            
        logger.info(f"✅ Artefactos base de {system} actualizados.")
    else:
        logger.info(f"Volumen de datos suficiente ({len(df_real)} muestras). Ajustando calibrador de salida...")
        model_base, _ = load_model(system) 

        # [BLINDAJE CRÍTICO]: Asegurar alineación estricta de columnas con el modelo matemático
        if hasattr(model_base, 'feature_names_in_'):
            expected_features = list(model_base.feature_names_in_)
            X_calibracion_ready = X_calibracion[expected_features]
        else:
            drop_meta = ['fault', 'run_id', 'fault_id', 'time_min', 'fault_numeric', 'prediction', 'confidence']
            X_calibracion_ready = X_calibracion.drop(columns=[c for c in drop_meta if c in X_calibracion.columns], errors='ignore')

        # [APLICACIÓN DE LA DOCUMENTACIÓN OFICIAL]
        # Envolvemos el modelo base en un FrozenEstimator para indicarle a sklearn que ya está entrenado
        from sklearn.frozen import FrozenEstimator
        
        model_congelado = FrozenEstimator(model_base)
        
        # Instanciamos el calibrador nativo. Al usar FrozenEstimator, cv y ensemble se gestionan solos correctamente
        model_calibrado = CalibratedClassifierCV(
            estimator=model_congelado, 
            method='isotonic'
        )
        
        # Ajustamos las funciones de calibración isotónica sobre los datos limpios
        model_calibrado.fit(X_calibracion_ready, df_real['fault_id'])
        
        # Definimos las rutas fechadas para este bloque atómico calibrado
        model_calib_path = art_path / f"{system}_model_calibrated_{fecha_hoy}.pkl"
        stats_calib_path = art_path / f"{system}_stats_calibrated_{fecha_hoy}.yaml"
        thresh_calib_path = art_path / f"{system}_thresholds_calibrated_{fecha_hoy}.yaml"
        
        # Guardamos todo el paquete con la misma marca de tiempo
        joblib.dump(model_calibrado, model_calib_path)
        with open(stats_calib_path, 'w') as f:
            yaml.dump(new_stats, f)
        with open(thresh_calib_path, 'w') as f:
            yaml.dump(current_thresh, f)
            
        logger.info(f"✅ Nueva calibración guardada con fecha de [{fecha_hoy}].")


def get_stats():
    """Imprime estadísticas detalladas y salud del dataset."""
    cfg = load_config()
    system = cfg["selected_system"]
    path = Path(cfg["paths"]["processed_data"]) / f"{system}_final.csv"
    
    if path.exists():
        df = pd.read_csv(path)
        logger.info(f"Generando reporte de salud para {system}...")
        
        # 1. Resumen de columnas
        summary = get_column_summary(df)
        print("\n--- RESUMEN DE CALIDAD DE DATOS ---")
        print(summary[['dtype', 'null_pct', 'unique_values']].head(30)) # Limitamos para ver lo importante

        print("\n--- DIVIDIENDO LAS FEATURES EN CATEGORÍAS ---")
        get_feature_groups(df)
        
        # 2. Check de Leakage inteligente
        target = cfg[system].get("target_column", "fault_id")
        
        # Definimos qué columnas ignorar en el check de correlación porque NO son sensores
        # Estas coinciden con tus drop_cols del trainer
        ignore_cols = [
            'run_id', 'time_min', 'fault', 'fault_id', 
            'fault_numeric', 'T_cond_sat', 'T_cab_meas', 'P_suc_bar'
        ]
        
        # Filtrar el DF para el check de leakage: solo sensores vs target
        df_for_leakage = df.drop(columns=[c for c in ignore_cols if c in df.columns and c != target])
        
        print("\n--- ANALIZANDO POSIBLE DATA LEAKAGE (QUITADO EL METADATA)---")
        check_data_leakage(df_for_leakage, target)

        
        # Ver distribución de fallos
        if target in df.columns:
            print("\n--- DISTRIBUCIÓN DE FALLOS EN EL DATASET ---")
            dist = df[target].value_counts(normalize=True) * 100
            mapping = cfg[system].get("mapping", {})
            for fid, pct in dist.items():
                name = mapping.get(int(fid), f"ID {fid}")
                print(f"  {name:<25}: {pct:>6.2f}%")
        
        print(f"\n✅ Ciclos detectados (Runs): {df['run_id'].nunique()}")
        print(f"✅ Total de muestras: {len(df)}")
        
    else:
        logger.error(f"No se han encontrado datos procesados en {path}")

def get_info():
    """Genera un glosario técnico dividido por origen de datos y diagnósticos."""
    cfg = load_config()
    system = cfg["selected_system"]
    
    
    if system == "refrigeracion":
        # ---  FEATURES ORIGINALES (REQUERIDAS) ---
        raw_features = {
        'T_amb': "Temperatura ambiente exterior. Base para el cálculo de eficiencia de condensación.",
        'T_set': "Punto de consigna. Temperatura objetivo configurada en el termostato.",
        'T_cab': "Temperatura de cabina lógica (valor usado por el algoritmo de control).",
        'T_evap_sat': "Temperatura de saturación en evaporador. Indica el estado del gas en baja presión.",
        'T_cond_sat': "Temperatura de saturación en condensador. Indica el estado del gas en alta presión.",
        'P_suc_bar': "Presión de succión. Fundamental para detectar fugas de refrigerante.",
        'P_dis_bar': "Presión de descarga. Sensor crítico para detectar obstrucciones y sobrecargas.",
        'N_comp_Hz': "Frecuencia del compresor (Inverter). Indica la demanda de potencia.",
        'SH_K': "Sobrecalentamiento (Superheat). Diferencia entre temp. real y de saturación en succión.",
        'P_comp_W': "Consumo eléctrico real del compresor en vatios.",
        'Q_evap_W': "Capacidad frigorífica. Calor total extraído del interior de la nevera.",
        'COP': "Ratio de eficiencia. Relación entre frío generado y energía consumida.",
        'frost_level': "Estimación de escarcha acumulada en las aletas del evaporador.",
        'T_cab_meas': "Valor leído directamente por el sensor físico de temperatura de cabina.",
        'valve_open': "Porcentaje de apertura de la válvula de expansión electrónica.",
        'door_open': "Sensor de contacto de puerta (0: Cerrada, 1: Abierta).",
        'defrost_on': "Estado de la resistencia de desescarche (0: Apagada, 1: Activa).",
        'time_min': "Tiempo de operación transcurrido en el ciclo actual.",
        'run_id': "Identificador único de cada ciclo de encendido/apagado."
    }

        # --- INGENIERÍA DE CARACTERÍSTICAS (PROCESADAS) ---
        eng_features = {
        'T_error': "Desviación (T_cab - T_set). Mide el error de seguimiento de temperatura.",
        'T_lift': "Salto térmico (T_cond - T_evap). El trabajo térmico total que realiza el sistema.",
        'T_cond_approach': "Diferencial (T_cond - T_amb). Indicador directo de suciedad en condensador.",
        'P_ratio': "Relación de presiones. Mide el estrés mecánico y eficiencia del compresor.",
        'Q_est': "Eficiencia de transferencia. Vatios consumidos por cada grado de enfriamiento real.",
        'Sensor_error': "Diferencia entre sensor lógico y físico. Detecta desviaciones (Drift).",
        'T_cab_grad': "Tendencia o velocidad de cambio de la temperatura en la cabina.",
        'Eff_vol': "Eficiencia volumétrica estimada. Relación térmica entre baja y alta presión.",
        'P_dis_error': "Error de presión de descarga. Clave para detectar gases incondensables (Aire).",
        'EEI': "Energy Efficiency Index. Relación entre potencia y el diferencial de temperatura.",
        'Thermal_Load_Index': "Índice de carga térmica. Combina el esfuerzo de presión con el salto térmico.",
        'P_dis_volatility': "Estabilidad de la presión de alta. Detecta inestabilidades en la expansión."
    }

        # ---  CLASES DE FALLO ---
        fault_desc = {
        0:  "NORMAL: Operación correcta.",
        1:  "COND_FOUL_MILD: Suciedad leve en condensador.",
        2:  "COND_FOUL_SEVERE: Obstrucción crítica en condensador.",
        3:  "EVAP_FAN_DEG: Degradación del ventilador del evaporador.",
        4:  "EVAP_FAN_FAIL: Fallo total del ventilador del evaporador.",
        5:  "UNDERCHARGE_MILD: Fuga de refrigerante leve.",
        6:  "UNDERCHARGE_SEVERE: Falta crítica de refrigerante.",
        7:  "OVERCHARGE: Exceso de carga de refrigerante.",
        8:  "SENSOR_DRIFT_PLUS: Sensor de cabina mide por encima del valor real.",
        9:  "SENSOR_DRIFT_MINUS: Sensor de cabina mide por debajo del valor real.",
        10: "COMP_INEFFICIENCY: Compresor desgastado o ineficiente.",
        11: "NON_CONDENSABLES: Aire o gases no condensables en el circuito.",
        12: "UNDERCHARGE_AND_COND_FOUL: Fallo combinado (Fuga + Suciedad)."
    }
    elif system == "aireado":
        # --- 1. AIREADO: FEATURES ORIGINALES ---
        raw_features = {
            'run_id': "Identificador del lote o ciclo de secado.",
            'time_min': "Tiempo transcurrido desde el inicio del proceso de aireado.",
            'T_amb': "Temperatura ambiente exterior a la cámara de secado.",
            'T_set': "Temperatura consigna para el proceso de curado.",
            'Kg_embutido': "Carga total de producto fresco introducida en la cámara.",
            'N_fan_Hz': "Frecuencia de los ventiladores de aireación. Controla la velocidad del aire.",
            'T_cab': "Temperatura del aire interior de la cámara de aireado.",
            'RH_cab': "Humedad Relativa (%) dentro de la cámara. Crítica para el secado.",
            'T_evap_sat': "Temperatura de saturación del evaporador (punto de rocío/hielo).",
            'P_comp_W': "Potencia consumida por el sistema de frío asociado al aireado."
        }

        # --- 2. AIREADO: INGENIERÍA DE CARACTERÍSTICAS ---
        eng_features = {
            'VPD': "Vapor Pressure Deficit. Fuerza impulsora que extrae el agua del embutido hacia el aire.",
            'RH_error': "Desviación respecto al 75% HR. Mide el alejamiento del punto óptimo de curado.",
            'Air_Flow_Ratio': "Relación Aire/Carga. Evalúa si el flujo es excesivo o pobre para la masa de carne.",
            'Evap_Eff_Index': "Eficiencia de Evaporación. Equilibrio entre enfriar el aire y quitarle humedad.",
            'Specific_Power_Load': "Energía por Kg. Mide la eficiencia energética relativa al volumen de producto.",
            'Encostramiento_Risk': "Índice de riesgo de 'Case Hardening'. Detecta si el aire rápido seca la piel muy pronto."
        }

        # --- 3. AIREADO: CLASES ---
        fault_desc = {
            0: "NORMAL: Proceso de secado correcto y uniforme.",
            1: "ENCOSTRAMIENTO: El exterior se seca muy rápido y sella el interior (Peligro de putrefacción interna).",
            2: "SATURACIÓN/HIELO: Exceso de humedad o evaporador congelado. El producto no pierde agua.",
            3: "FALLO VENTILADOR: Falta de flujo de aire. Crecimiento de moho indeseado y falta de uniformidad."
        }

    # --- EXPLICACIÓN GENERAL DE TRANSFORMACIONES TEMPORALES ---
    temporal_desc = {
        'Lags (t-n)': "Valores pasados de un sensor. Permiten al modelo ver la inercia (ej. qué pasó hace 15 min).",
        'Deltas (Δ)': "Diferencia neta entre el valor actual y uno pasado. Detecta caídas o subidas bruscas.",
        'Rolling Mean': "Media móvil. Suaviza el ruido de los sensores para ver la tendencia real del ciclo.",
        'Rolling Std': "Volatilidad. Mide si un sensor está inestable o 'vibrando', lo que indica fallos mecánicos."
    }

    # --- 3. MODELOS UTILIZADOS (GLOBAL) ---
    model_info = {
        'Arquitectura Final': "Random Forest (RF) + Capa Neurosimbólica (NS) + Voto por Run.",
        'Capa Neurosimbólica': "Inyección de reglas físicas (Axiomas) para corregir clasificaciones erróneas.",
        'Voto por Run': "Consolidación temporal que estabiliza el diagnóstico en ciclos largos."
    }

    # Sección Modelos
    print(f"\n[ESTRATEGIA DE MODELADO]")
    for k, v in model_info.items():
        print(f" {k:<25} | {v}")

    # --- IMPRESIÓN DEL REPORTE ---
    print(f"\nSISTEMA DE {system.upper()}")
    print(f"\nFEATURES ORIGINALES DEL DATASET")
    print(f"{'-'*95}")
    for feat, desc in raw_features.items():
        print(f" {feat:<20} | {desc}")

    print(f"\nINGENIERÍA DE CARACTERÍSTICAS (BASADA EN LITERATURA TÉCNICA)")
    print(f"{'-'*95}")
    for feat, desc in eng_features.items():
        print(f" {feat:<20} | {desc}")

    print(f"\nTRANSFORMACIONES TEMPORALES (Métricas Dinámicas)")
    print(f"{'-'*95}")
    for feat, desc in temporal_desc.items():
        print(f" {feat:<20} | {desc}")

    print(f"\nCLASES DE FALLO")
    print(f"{'-'*95}")
    for fid, desc in fault_desc.items():
        print(f" ID {fid:<2} | {desc}")

    print(f"{'='*95}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "data_processing": data_processing()
        elif cmd == "train": train()
        elif cmd == "tuning": tuning()
        elif cmd == "predict": predict()
        elif cmd == "calibrate": calibrate()
        elif cmd == "evaluate": evaluate()
        elif cmd == "get_stats": get_stats()
        elif cmd == "get_info": get_info()
        else:
            logger.error(f"Comando no reconocido: {cmd}")
    else:
        print("Uso: python src/main.py [data_processing|train|tuning|predict|evaluate|calibrate|get_stats|get_info]")