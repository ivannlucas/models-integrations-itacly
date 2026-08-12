import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import gc
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D

from src.utils.logging import get_logger

logger = get_logger(__name__)

def crear_secuencias(df_part, window_size, features, t_class, t_reg, series_col):
    """Convierte dataset tabular en tensores 3D."""
    X, y_c, y_r = [], [], []
    grupos = df_part.groupby(series_col)
    
    for _, grupo in tqdm(grupos, desc="Generando tensores", leave=False):
        data_features = grupo[features].values
        data_class = grupo[t_class].values
        data_reg = grupo[t_reg].values
        
        for i in range(len(data_features) - window_size):
            X.append(data_features[i : i + window_size])
            y_c.append(data_class[i + window_size])
            y_r.append(data_reg[i + window_size])
            
    return np.array(X, dtype=np.float32), np.array(y_c, dtype=np.int8), np.array(y_r, dtype=np.float32)

def build_models(input_shape: tuple, num_classes: int, learning_rates: dict, arch_params: dict) -> list:
    """Construye las arquitecturas de red."""
    def compilar_modelo(modelo, lr):
        modelo.compile(
            optimizer=Adam(learning_rate=lr),
            loss={"out_class": "sparse_categorical_crossentropy", "out_reg": "mse"},
            loss_weights={"out_class": 1.0, "out_reg": 1.0},
            metrics={"out_class": ["accuracy"], "out_reg": ["mae"]}
        )
        return modelo

    # M1 LSTM
    inp = Input(shape=input_shape, name="Input_M1")
    x = LSTM(arch_params['lstm']['lstm_1'], return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(arch_params['lstm']['dropout'])(x)
    x = LSTM(arch_params['lstm']['lstm_2'])(x)
    x = BatchNormalization()(x)
    out_class1 = Dense(num_classes, activation='softmax', name="out_class")(x)
    out_reg1 = Dense(1, activation='sigmoid', name="out_reg")(x)
    m1_lstm = compilar_modelo(Model(inputs=inp, outputs=[out_class1, out_reg1], name="M1_LSTM"), learning_rates.get('lstm', 0.001))

    # M2 CNN
    inp = Input(shape=input_shape, name="Input_M2")
    x = Conv1D(filters=arch_params['cnn']['filters_1'], kernel_size=arch_params['cnn']['kernel_1'], activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Conv1D(filters=arch_params['cnn']['filters_2'], kernel_size=arch_params['cnn']['kernel_2'], activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(arch_params['cnn']['dropout'])(x)
    out_class2 = Dense(num_classes, activation='softmax', name="out_class")(x)
    out_reg2 = Dense(1, activation='sigmoid', name="out_reg")(x)
    m2_cnn = compilar_modelo(Model(inputs=inp, outputs=[out_class2, out_reg2], name="M2_CNN"), learning_rates.get('cnn', 0.001))

    # M3 BiGRU
    inp = Input(shape=input_shape, name="Input_M3")
    x = Bidirectional(GRU(arch_params['bigru']['gru_units'], return_sequences=False))(inp)
    x = BatchNormalization()(x)
    x = Dropout(arch_params['bigru']['dropout'])(x)
    x = Dense(arch_params['bigru']['dense_units'], activation='relu')(x)
    out_class3 = Dense(num_classes, activation='softmax', name="out_class")(x)
    out_reg3 = Dense(1, activation='sigmoid', name="out_reg")(x)
    m3_bigru = compilar_modelo(Model(inputs=inp, outputs=[out_class3, out_reg3], name="M3_BiGRU"), learning_rates.get('bigru', 0.001))

    return [m1_lstm, m2_cnn, m3_bigru]


def run_training(processed_data_path: str, config: dict, save_metrics: bool = False):
    """Ejecuta toda la lógica de validación, escalado, transformación temporal y red neuronal."""
    
    # 0. VERIFICAR ACELERACIÓN GPU / CUDA
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("⚠️ CUDA no detectado: TensorFlow entrenará en CPU. Esto incrementará significativamente el tiempo de entrenamiento (x3-5).")
    else:
        logger.info(f"✅ GPU detectada para aceleración: {len(gpus)} dispositivo(s) encontrado(s).")
        
    # La semilla se fija más abajo, justo antes de construir los modelos,
    # tras haber leído config['split']['random_seed'].
    
    ruta_proc = Path(processed_data_path)
    ruta_splits = Path(config['paths']['splits_dir'])
    
    # 2. SELECCIÓN DE VARIABLES
    cols_features = config.get('cols_features', [])
    target_class = config['targets']['classification_label']
    target_reg = config['targets']['regression_column']
    class_col = config['targets']['classification_column']
    stratify_col = config['targets']['stratify_column']
    series_col = config['targets']['series_column']
    seed = config['split']['random_seed']
    
    # 1. CODIFICACIÓN (Ajustamos encoder)
    le = LabelEncoder()
    
    if not ruta_proc.exists():
        logger.warning(f"Archivo procesado no encontrado: {ruta_proc}. Intentando cargar splits pre-existentes (Fallback Mode)...")
        train_path = ruta_splits / config['output_files']['train_split']
        val_path = ruta_splits / config['output_files']['val_split']
        test_path = ruta_splits / config['output_files']['test_split']
        
        if train_path.exists() and val_path.exists() and test_path.exists():
            logger.info("Cargando splits directamente de data/splits/")
            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)
            
            # Reconstruir el LabelEncoder (solo sobre train para evitar fuga de datos)
            le.fit(train_df[class_col])
            logger.info(f"Clases a entrenar (reconstruidas): {le.classes_}")
            
            # Generar columna target numérica (necesaria para crear_secuencias)
            for split_df in [train_df, val_df, test_df]:
                split_df[target_class] = le.transform(split_df[class_col])
        else:
            logger.error("No se encontraron splits pre-existentes en data/splits/. Abortando entrenamiento.")
            return
    else:
        df = pd.read_parquet(ruta_proc)
        
        # ASEGURAR ORDEN DE COLUMNAS (Igual que en el Notebook)
        df = df.reindex(sorted(df.columns), axis=1)
        
        df[target_class] = le.fit_transform(df[class_col])
        logger.info(f"Clases a entrenar: {le.classes_}")

        # 3. SPLIT ESTRATÉGICO
        logger.info("Partiendo el dataset por IDs de series para evitar Leakage...")
        df_unicos = df.drop_duplicates(subset=[series_col])[[series_col, stratify_col]]
        test_size = config['split']['test_size']
        val_test_ratio = config['split']['val_test_ratio']
        train_ids, temp_ids, _, temp_etiquetas = train_test_split(
            df_unicos[series_col], df_unicos[stratify_col], 
            test_size=test_size, random_state=seed, stratify=df_unicos[stratify_col]
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=val_test_ratio, random_state=seed, stratify=temp_etiquetas
        )
        
        train_df = df[df[series_col].isin(train_ids)].copy()
        val_df = df[df[series_col].isin(val_ids)].copy()
        test_df = df[df[series_col].isin(test_ids)].copy()
        
        logger.info(f"Series repartidas: Train {len(train_ids)}, Val {len(val_ids)}, Test {len(test_ids)}")

        # 4. EXPORTAR SPLITS
        ruta_splits.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(ruta_splits / config['output_files']['train_split'])
        val_df.to_parquet(ruta_splits / config['output_files']['val_split'])
        test_df.to_parquet(ruta_splits / config['output_files']['test_split'])
        logger.info("Splits generados y guardados en data/splits/")

    # 5. ESCALADO FIT A TRAIN
    scaler = StandardScaler()
    scaler.fit(train_df[cols_features])
    train_df[cols_features] = scaler.transform(train_df[cols_features])
    val_df[cols_features] = scaler.transform(val_df[cols_features])
    test_df[cols_features] = scaler.transform(test_df[cols_features])
    
    # 6. PREPARAR DIRECTORIO DE ARTEFACTOS
    ruta_artifacts = Path(config['paths']['artifacts_dir'])
    ruta_artifacts.mkdir(parents=True, exist_ok=True)

    # 7. GENERACIÓN TENSORES
    window_size = config['params']['window_size']
    logger.info("Empaquetando secuencias en tensores 3D...")
    X_train, y_train_c, y_train_r = crear_secuencias(train_df, window_size, cols_features, target_class, target_reg, series_col)
    X_val, y_val_c, y_val_r = crear_secuencias(val_df, window_size, cols_features, target_class, target_reg, series_col)

    if save_metrics:
        logger.info("Generando tensores de test para métricas...")
        X_test, y_test_c, y_test_r = crear_secuencias(test_df, window_size, cols_features, target_class, target_reg, series_col)
    else:
        X_test, y_test_c, y_test_r = None, None, None

    del train_df, val_df, test_df
    gc.collect()

    # 8. CONSTRUIR REDES
    input_shape = (window_size, len(cols_features))
    num_classes = len(le.classes_)
    
    # Fijar semilla global para reproducibilidad (determinismo completo)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        logger.warning(f"⚠️ No se pudo activar determinismo completo de operaciones (puede ocurrir en GPU): {e}")
    
    modelos = build_models(
        input_shape, num_classes, 
        learning_rates=config['params']['learning_rates'],
        arch_params=config['params']['architectures']
    )
    
    epochs = config['params']['epochs']
    batch = config['params']['batch_size']
    patience = config['params']['early_stopping_patience']
    
    ruta_metrics = Path(config['paths']['metrics_dir'])
    ruta_metrics.mkdir(parents=True, exist_ok=True)

    logger.info(f"Comenzando Entrenamiento Secuencial de {len(modelos)} Redes...")
    start_time_total = time.time()
    
    for modelo in modelos:
        logger.info(f"Entrenando: {modelo.name}")
        start_time_model = time.time()
        
        ruta_modelo = ruta_artifacts / f"{modelo.name}.keras"
        
        c_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        c_check = ModelCheckpoint(filepath=ruta_modelo, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        
        hist = modelo.fit(
            X_train, {"out_class": y_train_c, "out_reg": y_train_r},
            validation_data=(X_val, {"out_class": y_val_c, "out_reg": y_val_r}),
            epochs=epochs, batch_size=batch,
            callbacks=[c_stop, c_check], verbose=1
        )
        
        ruta_historial = ruta_metrics / f"history_{modelo.name}.csv"
        pd.DataFrame(hist.history).to_csv(ruta_historial, index=False)
            
        end_time_model = time.time()
        elapsed_model_mins = (end_time_model - start_time_model) / 60.0
        logger.info(f"⏱️ Tiempo de entrenamiento para {modelo.name}: {elapsed_model_mins:.2f} minutos.")
            
    end_time_total = time.time()
    elapsed_total_mins = (end_time_total - start_time_total) / 60.0
    logger.info(f"⏱️ TIEMPO TOTAL DE ENTRENAMIENTO DEL ENSEMBLE: {elapsed_total_mins:.2f} minutos.")
    
    # GUARDAR ARTEFACTOS POST-ENTRENAMIENTO (SCALER/LABEL)
    # Se guardan DESPUÉS de completar todos los modelos para evitar mismatch
    # entre scaler.pkl y los .keras en caso de interrupción.
    joblib.dump(scaler, ruta_artifacts / config['model_names']['scaler'])
    joblib.dump(le, ruta_artifacts / config['model_names']['label_encoder'])
    logger.info("Scaler y LabelEncoder guardados correctamente tras completar entrenamiento.")
    
    if save_metrics and X_test is not None:
        logger.info("Calculando métricas sobre conjunto de Test...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
        
        preds_clases_prob = []
        preds_regresion = []
        
        for modelo in modelos:
            logger.info(f"Evaluando: {modelo.name}")
            p_class, p_reg = modelo.predict(X_test, verbose=0)
            preds_clases_prob.append(p_class)
            preds_regresion.append(p_reg)
            
            y_pred_c = np.argmax(p_class, axis=1)
            y_pred_r = p_reg.flatten()
            
            # Nombre de modelo (LSTM, CNN, BiGRU)
            model_short_name = modelo.name.split('_')[-1]
            model_metrics_dir = ruta_metrics / model_short_name
            model_metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Matriz de confusión
            cm = confusion_matrix(y_test_c, y_pred_c)
            plt.figure(figsize=(10,8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title(f'Matriz de Confusión - {modelo.name}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(model_metrics_dir / f"confusion_matrix_{model_short_name}_test.png")
            plt.close()
            
            # Reporte de métricas
            report_dict = classification_report(y_test_c, y_pred_c, target_names=le.classes_, output_dict=True)
            metrics_df = pd.DataFrame(report_dict).transpose()
            metrics_df['mae'] = mean_absolute_error(y_test_r, y_pred_r)
            metrics_df['mse'] = mean_squared_error(y_test_r, y_pred_r)
            metrics_df['r2'] = r2_score(y_test_r, y_pred_r)
            metrics_df.to_csv(model_metrics_dir / f"metrics_{model_short_name}_test.csv")
            
        # Evaluación DEL (Ensemble)
        logger.info("Evaluando Ensamble (DEL)...")
        media_probs = np.mean(preds_clases_prob, axis=0)
        y_pred_c_del = np.argmax(media_probs, axis=1)
        y_pred_r_del = np.mean(preds_regresion, axis=0).flatten()
        
        del_metrics_dir = ruta_metrics / config['output_files']['metrics_ensemble_dir']
        del_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        cm_del = confusion_matrix(y_test_c, y_pred_c_del)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm_del, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Matriz de Confusión - Deep Ensemble Learning (DEL)')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(del_metrics_dir / "confusion_matrix_DEL_test.png")
        plt.close()
        
        report_dict_del = classification_report(y_test_c, y_pred_c_del, target_names=le.classes_, output_dict=True)
        metrics_df_del = pd.DataFrame(report_dict_del).transpose()
        metrics_df_del['mae'] = mean_absolute_error(y_test_r, y_pred_r_del)
        metrics_df_del['mse'] = mean_squared_error(y_test_r, y_pred_r_del)
        metrics_df_del['r2'] = r2_score(y_test_r, y_pred_r_del)
        metrics_df_del.to_csv(del_metrics_dir / config['output_files']['metrics_ensemble_file'])
        
        logger.info("Artefactos de evaluación guardados en models/metrics/")
        
    logger.info("Proceso de ENTRENAMIENTO completado integralmente.")
