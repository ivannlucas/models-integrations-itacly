import numpy as np
import pandas as pd
import torch
import random
import pygad
import torch.nn as nn
import torch.optim as optim
import joblib
import yaml
import sys
import os
from sklearn.preprocessing import StandardScaler



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processing.load_data import load_split_data
from src.data_processing.generate_data import generate_data

from src.training.model import PasteurizationANN
from src.training.training_functions import train_function, split_and_save_splits

from src.predict.predictor import load_model, run_inference, save_predictions, compare_optimization_baseline
from src.predict.metrics import get_metrics

def data_generating():
    """Pipeline ETL"""

    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    raw_dataset_path = config["raw_dataset_path"]
    splits_path = config["splits_path"]


    df = generate_data(data_saving_path=raw_dataset_path)

    X_train, X_test, X_val, y_train, y_test, y_val = split_and_save_splits(df, splits_path)


def data_processing():
    """
    Procesamiento de datos siguiendo estrictamente las fórmulas del generador de Datagia.
    Sirve para limpiar el dataset de entrenamiento o para procesar una entrada de operario.
    """
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    df_raw = pd.read_csv(config['raw_dataset_path'])
    df = df_raw.copy()

    # 1. VARIABLES DE ENTRADA / CONTROL (Lo que nos da el operario o sensor)
    # Suponemos que ya vienen en el df: temp_entrada_leche, temp_ambiente, 
    # temp_setpoint_leche, flujo_leche_lh, horas_desde_limpieza, presion_diferencial_bar

    # 2. CÁLCULOS FÍSICOS REPLICADOS (Exactamente como en tu generador)

    # A. Temperatura de Proceso (En operación real es el sensor, en simulación es el setpoint)
    # Si la columna no existe (porque es una prueba del GA), la creamos
    if 'temp_proceso_leche' not in df.columns:
        df['temp_proceso_leche'] = df['temp_setpoint_leche']

    # B. Temperatura de Agua de Servicio (Si no viene dada, se calcula por el Delta T)
    # En tu generador: temp_agua_servicio = temp_proceso_leche + delta_t_servicio (6-14)
    if 'temp_agua_servicio' not in df.columns:
        # Para un operario, asumimos un delta medio de 10°C si no se conoce
        df['temp_agua_servicio'] = df['temp_proceso_leche'] + 10.0

    # C. Seguridad Microbiológica (PU) - Fórmulas exactas de tu código
    volumen_retencion = 16
    Z = 7
    Tref = 72

    # tiempo_residencia = (16 / flujo) * 3600
    df['tiempo_residencia'] = (volumen_retencion / df['flujo_leche_lh']) * 3600

    # pu = tiempo_residencia * 10^((temp_proceso - 72) / 7)
    df['valor_pu_microbiologico'] = (
        df['tiempo_residencia'] * 10**((df['temp_proceso_leche'] - Tref) / Z)
    )

    # D. Índice de Seguridad
    df['indice_seguridad'] = (df['valor_pu_microbiologico'] >= 13).astype(int)

    # 3. LIMPIEZA Y ORDEN (Mismo orden que X_cols para la ANN)
    cols_para_ann = [
        'temp_entrada_leche', 'temp_ambiente', 'temp_setpoint_leche', 
        'temp_proceso_leche', 'temp_agua_servicio', 'flujo_leche_lh', 
        'horas_desde_limpieza', 'presion_diferencial_bar'
    ]

    if 'consumo_agua_l' in df.columns:
        df_final = df[cols_para_ann + ['consumo_agua_l', 'valor_pu_microbiologico', 'indice_seguridad']]
    else:
        df_final = df[cols_para_ann + ['valor_pu_microbiologico', 'indice_seguridad']]


    X_train, X_test, X_val, y_train, y_test, y_val = split_and_save_splits(df_final, config['splits_path'])
    os.makedirs(os.path.dirname(config['dataset_path']), exist_ok=True)
    df_final.to_csv(config['dataset_path'], index=False)
    print(f"Datos generados con éxito en {config['dataset_path']}")

    return df_final


def train():

    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    #### CONFIG!! ####
    model_artifacts_path = config['model_artifacts_path']
    model_metrics_path = config['model_metrics_path']
    predictions_path = config['predictions_path']
    splits_path = config['splits_path']
    random_seed = config["random_seed"]

    os.makedirs(model_artifacts_path, exist_ok=True)
    os.makedirs(model_metrics_path, exist_ok=True)
    os.makedirs(predictions_path, exist_ok=True)
    os.makedirs(splits_path, exist_ok=True)

    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)



    features = ['temp_entrada_leche', 'temp_ambiente', 'temp_setpoint_leche', 'temp_proceso_leche',
            'temp_agua_servicio', 'flujo_leche_lh', 'horas_desde_limpieza', 'presion_diferencial_bar']
    target = 'consumo_agua_l'

    X_train, X_test, X_val, y_train, y_test, y_val = load_split_data(config)

    # Escalado (Crucial para Redes Neuronales y para el GA posterior)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    model = PasteurizationANN(len(features))
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    epochs = 300


    best_state, best_val_loss = train_function(model, optimizer, criterion, scheduler, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, epochs, scaler_y=scaler_y)
    model.load_state_dict(best_state)

    # 4. GUARDADO DE ARTEFACTOS 

    torch.save(model.state_dict(), os.path.join(model_artifacts_path, 'model_ann.pt'))
    joblib.dump(scaler_X, os.path.join(model_artifacts_path, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(model_artifacts_path, 'scaler_y.pkl'))

# Convertir arrays de NumPy a DataFrames de Pandas para poder usar .to_csv()
    pd.DataFrame(X_train_scaled, columns=features).to_csv(os.path.join(splits_path, 'X_train_scaled.csv'), index=False)
    pd.DataFrame(y_train_scaled, columns=[target]).to_csv(os.path.join(splits_path, 'y_train_scaled.csv'), index=False)
    pd.DataFrame(X_val_scaled, columns=features).to_csv(os.path.join(splits_path, 'X_val_scaled.csv'), index=False)
    pd.DataFrame(y_val_scaled, columns=[target]).to_csv(os.path.join(splits_path, 'y_val_scaled.csv'), index=False)


    train_r2 = predict(X_train, y_train, "train_predictions.csv", "train_metrics.txt")
    val_r2 = predict(X_val, y_val,  "val_predictions.csv", "val_metrics.txt")

    print(f"Entrenamiento completado. R2 train: {train_r2:.4f}. R2 val: {val_r2:.4f}. Artefactos, predicciones y métricas de entrenamiento guardados.")


def predict(X, y, predictions_file_name=None, metrics_file_name=None):
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    predictions_path = config['predictions_path']
    model_metrics_path = config['model_metrics_path']
    splits_path = config['splits_path']

    model, scaler_X, scaler_y = load_model(config)    

    features = ['temp_entrada_leche', 'temp_ambiente', 'temp_setpoint_leche', 'temp_proceso_leche',
            'temp_agua_servicio', 'flujo_leche_lh', 'horas_desde_limpieza', 'presion_diferencial_bar']

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)
    y_pred = run_inference(model, X_scaled, scaler_y)


    # Creamos un DataFrame para comparar Real vs Predicho en el entrenamiento
    df_train_preds = pd.DataFrame(X, columns=features)
    df_train_preds['consumo_real'] = y.values
    df_train_preds['consumo_predicho'] = y_pred

    # Calcular PU 
    t_res = (16 / df_train_preds['flujo_leche_lh']) * 3600
    df_train_preds['pu_logrado'] = t_res * 10**((df_train_preds['temp_proceso_leche'] - 72) / 7)

    if predictions_file_name:
        save_predictions(df_train_preds, os.path.join(predictions_path + predictions_file_name))
    if metrics_file_name:
        r2, mae, rmse, mae_relativo, rmse_mae_ratio = get_metrics(y, y_pred, model_metrics_path + metrics_file_name)

    return r2


def evaluate_test():

    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    X_train, X_test, X_val, y_train, y_test, y_val = load_split_data(config)

    predict(X_test, y_test ,predictions_file_name="test_predictions.csv", metrics_file_name="test_metrics.txt")
    print("Evaluación completada. Métricas de test almacenadas.")


def predict_external(input_path, output_path=None):
    """
    Ejecuta inferencia sobre un CSV externo.
    El CSV debe incluir las variables de entrada del modelo. Si no incluye
    temp_proceso_leche o temp_agua_servicio, se calculan de forma determinista
    igual que en data_processing().
    """
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if not input_path:
        raise ValueError("Debe indicarse una ruta de entrada con --input_path")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    features = [
        'temp_entrada_leche', 'temp_ambiente', 'temp_setpoint_leche',
        'temp_proceso_leche', 'temp_agua_servicio', 'flujo_leche_lh',
        'horas_desde_limpieza', 'presion_diferencial_bar'
    ]

    df = pd.read_csv(input_path)

    if 'temp_proceso_leche' not in df.columns and 'temp_setpoint_leche' in df.columns:
        df['temp_proceso_leche'] = df['temp_setpoint_leche']
    if 'temp_agua_servicio' not in df.columns and 'temp_proceso_leche' in df.columns:
        df['temp_agua_servicio'] = df['temp_proceso_leche'] + 10.0

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(
            "El CSV de inferencia no contiene todas las variables requeridas: "
            + ", ".join(missing)
        )

    model, scaler_X, scaler_y = load_model(config)
    y_pred = run_inference(model, df[features], scaler_y, scaler_X, X_is_scaled=False)

    df_result = df.copy()
    df_result['consumo_agua_predicho_l'] = y_pred.reshape(-1)
    t_res = (16 / df_result['flujo_leche_lh']) * 3600
    df_result['pu_logrado'] = t_res * 10**((df_result['temp_proceso_leche'] - 72) / 7)

    if output_path is None:
        output_path = os.path.join(config['predictions_path'], 'external_predictions.csv')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_result.to_csv(output_path, index=False)
    print(f"Inferencia externa completada. Filas procesadas: {len(df_result)}")
    print(f"Entrada usada: {input_path}")
    print(f"Resultados guardados en: {output_path}")

    return df_result


def fine_tune(input_path, output_model_path=None, metrics_path=None, epochs=100, learning_rate=0.001):
    """
    Calibra el modelo preentrenado con datos etiquetados del cliente.
    El CSV debe contener las variables de entrada y consumo_agua_l.
    """
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if not input_path:
        raise ValueError("Debe indicarse una ruta de entrada con --input_path")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    features = [
        'temp_entrada_leche', 'temp_ambiente', 'temp_setpoint_leche',
        'temp_proceso_leche', 'temp_agua_servicio', 'flujo_leche_lh',
        'horas_desde_limpieza', 'presion_diferencial_bar'
    ]
    target = 'consumo_agua_l'

    df = pd.read_csv(input_path)
    if 'temp_proceso_leche' not in df.columns and 'temp_setpoint_leche' in df.columns:
        df['temp_proceso_leche'] = df['temp_setpoint_leche']
    if 'temp_agua_servicio' not in df.columns and 'temp_proceso_leche' in df.columns:
        df['temp_agua_servicio'] = df['temp_proceso_leche'] + 10.0

    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        raise ValueError(
            "El CSV de fine-tuning no contiene todas las columnas requeridas: "
            + ", ".join(missing)
        )

    model, scaler_X, scaler_y = load_model(config)
    X_scaled = scaler_X.transform(df[features])
    y_scaled = scaler_y.transform(df[[target]])

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)

    for epoch in range(int(epochs)):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
    y_pred = scaler_y.inverse_transform(pred_scaled)
    r2, mae, rmse, mae_relativo, rmse_mae_ratio = get_metrics(df[[target]], y_pred)

    if output_model_path is None:
        output_model_path = os.path.join(config['model_artifacts_path'], 'model_ann_finetuned.pt')
    if metrics_path is None:
        metrics_path = os.path.join(config['model_metrics_path'], 'fine_tune_metrics.txt')

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("INFORME DE FINE-TUNING - PROYECTO DATAGIA\n")
        f.write("------------------------------------------------\n")
        f.write(f"Filas usadas:   {len(df)}\n")
        f.write(f"Epochs:         {int(epochs)}\n")
        f.write(f"Learning rate:  {learning_rate}\n")
        f.write(f"R2 Score:       {r2:.4f}\n")
        f.write(f"MAE:            {mae:.2f} L\n")
        f.write(f"RMSE:           {rmse:.2f} L\n")
        f.write(f"MAE Relativo:   {mae_relativo:.2f}%\n")
        f.write(f"RMSE/MAE:       {rmse_mae_ratio:.2f}\n")

    print(f"Fine-tuning completado con {len(df)} filas.")
    print(f"Modelo calibrado guardado en: {output_model_path}")
    print(f"Métricas guardadas en: {metrics_path}")

    return output_model_path


def optimize(mode="single_mode", batch_input_path=None):
    """
    Orquestador de optimización. 
    Si batch_input_path es None, usa el contexto_actual de config.yaml.
    Si tiene una ruta, procesa todas las filas del CSV.
    """
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # 1. Cargar Artefactos
    model, scaler_X, scaler_y = load_model(config)
    model.eval()

    if mode == "massive_mode":
        escenarios = pd.DataFrame(config["optimization_scenarios"]) 
        print(f"--- Modo Massive: Procesando {len(escenarios)} escenarios del YAML ---")

    elif mode == "csv_mode":
        if not batch_input_path:
            raise ValueError("Debe indicarse --input_path cuando mode=csv_mode")

        if not os.path.exists(batch_input_path):
            raise FileNotFoundError(f"No existe el archivo: {batch_input_path}")

        escenarios = pd.read_csv(batch_input_path)

        print(f"--- Modo CSV: Procesando {len(escenarios)} escenarios ---")

    else:
        escenarios = pd.DataFrame([config["contexto_actual"]])
        print("--- Modo Single: Usando contexto_actual ---")

    print("Variables de contexto usadas:")
    print(", ".join(escenarios.columns))

    lista_resultados = []

    # 3. Bucle de Optimización
    for i, row in escenarios.iterrows():
        contexto = row.to_dict()

        def fitness_func(ga_instance, solution, solution_idx):
            # Lógica de predicción con la ANN
            features = ['temp_entrada_leche', 'temp_ambiente', 'temp_setpoint_leche', 
                        'temp_proceso_leche', 'temp_agua_servicio', 'flujo_leche_lh', 
                        'horas_desde_limpieza', 'presion_diferencial_bar']

            input_dict = {
                'temp_entrada_leche': contexto['temp_entrada_leche'],
                'temp_ambiente': contexto['temp_ambiente'],
                'temp_setpoint_leche': solution[0],
                'temp_proceso_leche': solution[0],
                'temp_agua_servicio': solution[1],
                'flujo_leche_lh': solution[2],
                'horas_desde_limpieza': contexto['horas_desde_limpieza'],
                'presion_diferencial_bar': contexto['presion_diferencial_bar']
            }

            input_df = pd.DataFrame([input_dict], columns=features)
            input_scaled = scaler_X.transform(input_df)

            with torch.no_grad():
                consumo_s = model(torch.FloatTensor(input_scaled)).item()
                # scaler_y espera 2D array
                consumo_real = scaler_y.inverse_transform([[consumo_s]])[0][0]

            # Restricción PU (Seguridad Alimentaria)
            t_res = (16 / solution[2]) * 3600
            pu = t_res * 10**((solution[0] - 72) / 7)

            if  solution[0] < 72 or pu < 13.0:
                return 1e-8

            return 1.0 / (consumo_real + 1e-6)

        # Configurar y ejecutar GA para este escenario
        ga_instance = pygad.GA(
            num_generations=50, 
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=20,
            num_genes=3,
            gene_space=[{'low': 72, 'high': 82}, {'low': 75, 'high': 92}, {'low': 2500, 'high': 4500}],
            suppress_warnings=True,
            random_seed=42
        )

        ga_instance.run()
        best_sol, _, _ = ga_instance.best_solution()
        t_res = (16/ best_sol[2]) * 3600
        pu_logrado = t_res * 10**((best_sol[0] - 72) / 7)

        # Comparar con baseline (Función que ya tienes definida)
        consumo_estandar, consumo_optimizado, ahorro_l, ahorro_pct = compare_optimization_baseline(contexto, best_sol, model, scaler_X, scaler_y)
        # Creamos el diccionario de resultados manualmente para evitar el error de mapping
        res_dict = {
            "opt_temp_leche": best_sol[0],
            "opt_temp_agua": best_sol[1],
            "opt_flujo": best_sol[2],
            "consumo_estandar": consumo_estandar,
            "consumo_optimizado": consumo_optimizado,
            "ahorro_l": ahorro_l,
            "ahorro_pct": ahorro_pct,
            "pu_logrado": pu_logrado
        }

        # Añadir metadatos del contexto para trazabilidad
        informe_fila = {**contexto, **res_dict}
        lista_resultados.append(informe_fila)

    # 4. Guardar resultados consolidados
    df_final = pd.DataFrame(lista_resultados)

    if mode == "massive_mode":
        output_path = os.path.join(config['predictions_path'], 'optimization_results_massive_mode.csv')
    elif mode == "csv_mode":
        output_path = os.path.join(config['predictions_path'], 'optimization_results_csv_mode.csv')
    else:
        output_path = os.path.join(config['predictions_path'], 'optimization_results_single_mode.csv')
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Proceso finalizado. Filas generadas: {len(df_final)}")
    print(f"Resultados en: {output_path}")
