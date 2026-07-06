import joblib
import torch
import os
import numpy as np
import pandas as pd

from src.training.model import PasteurizationANN


def load_model(config):
    ### CONFIG
    model_artifacts_path = config['model_artifacts_path']
    features = [
        'temp_entrada_leche', 'temp_ambiente', 'temp_setpoint_leche', 'temp_proceso_leche',
        'temp_agua_servicio', 'flujo_leche_lh', 'horas_desde_limpieza', 'presion_diferencial_bar'
    ]
    model = PasteurizationANN(input_size=len(features)) 
    model.load_state_dict(torch.load(os.path.join(model_artifacts_path, 'model_ann.pt')))
    scaler_X = joblib.load(os.path.join(model_artifacts_path, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(model_artifacts_path, 'scaler_y.pkl'))

    return model, scaler_X, scaler_y

def run_inference(model, X_scaled, scaler_y, scaler_X=None, X_is_scaled = True):
    """
    Realiza la predicción técnica usando la ANN.
    """
    model.eval()

    if X_is_scaled == False:
        X_scaled = scaler_X.transform(X_scaled)

    with torch.no_grad():
        # Aseguramos que X_scaled sea un tensor
        X_tensor = torch.FloatTensor(X_scaled)
        predictions_scaled = model(X_tensor).numpy()
        # Desescalamos para obtener litros reales
    predictions = scaler_y.inverse_transform(predictions_scaled)
    return predictions

def save_predictions(df_predictions, out_path):
    """Guarda la predicción de consumo asociado"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_predictions.to_csv(out_path, index=False)
    print(f"Registro completo guardado en: {out_path}")

def compare_optimization_baseline(contexto_dispositivo, best_sol, model, scaler_X, scaler_y):

    setpoints_estandar = np.array([80.0, 88.0, 3500.0]) # [Temp_L, Temp_S, Flujo]
    standard_input_vector = pd.DataFrame([[
        contexto_dispositivo['temp_entrada_leche'],
        contexto_dispositivo['temp_ambiente'],
        setpoints_estandar[0],  # Temp setpoint Leche
        setpoints_estandar[0],  # Temp proceso
        setpoints_estandar[1],  # Temp Servicio
        setpoints_estandar[2],  # Flujo
        contexto_dispositivo['horas_desde_limpieza'],
        contexto_dispositivo['presion_diferencial_bar']
    ]], columns=scaler_X.feature_names_in_)
    consumo_estandar = run_inference(model, standard_input_vector, scaler_y, scaler_X, X_is_scaled=False).item()

    optimized_input_vector = pd.DataFrame([[
        contexto_dispositivo['temp_entrada_leche'],
        contexto_dispositivo['temp_ambiente'],
        best_sol[0], # Temp Leche
        best_sol[0], # Temp proceso
        best_sol[1], # Temp Servicio
        best_sol[2], # Flujo
        contexto_dispositivo['horas_desde_limpieza'],
        contexto_dispositivo['presion_diferencial_bar']
    ]], columns=scaler_X.feature_names_in_)
    consumo_optimizado = run_inference(model, optimized_input_vector, scaler_y, scaler_X, X_is_scaled=False).item()


    ahorro_l = consumo_estandar - consumo_optimizado
    ahorro_pct = (ahorro_l / consumo_estandar) * 100 if consumo_estandar != 0 else 0

    # ============================================================
    # VEREDICTO DE AUDITORÍA - PROYECTO DATAGIA
    # ============================================================
    print("\n" + "="*50)
    print("REPORTE DE OPTIMIZACIÓN HÍDRICA (ANN + GA)")
    print("="*50)
    print(
        "Escenario Contextual: "
        f"Temp Entrada: {contexto_dispositivo['temp_entrada_leche']}°C | "
        f"Temp Amb: {contexto_dispositivo['temp_ambiente']}°C | "
        f"Horas Uso: {contexto_dispositivo['horas_desde_limpieza']}h | "
        f"Presión: {contexto_dispositivo['presion_diferencial_bar']} bar"
    )
    print("Baseline estándar: Temp. leche 80.0°C | Temp. agua 88.0°C | Flujo 3500 L/h")
    print("-"*50)
    print(f"Consumo ESTÁNDAR:      {consumo_estandar:,.2f} Litros")
    print(f"Consumo OPTIMIZADO:    {consumo_optimizado:,.2f} Litros")
    print("-"*50)
    print(f"Ahorro Neto:           {ahorro_l:,.2f} Litros")
    print(f"Reducción Lograda:     {ahorro_pct:.2f}%")
    print("-"*50)

    # Verificación de KPI (15% según Guía Modelo 35)
    kpi_objetivo = 15.0
    if ahorro_pct >= kpi_objetivo:
        print(f"KPI SUPERADO ({ahorro_pct:.2f}% >= {kpi_objetivo}%)")
        print("Resultado válido para el escenario simulado; requiere validación con datos reales antes de despliegue.")
    else:
        print(f"KPI NO ALCANZADO ({ahorro_pct:.2f}% < {kpi_objetivo}%)")
        print("Se recomienda ajustar la tasa de mutación o el espacio de búsqueda del GA.")
    print("="*50)

    print(f"\nSetpoints Sugeridos por Datagia:")
    print(f"   - Temp. Leche: {best_sol[0]:.2f}°C (vs 80.0°C)")
    print(f"   - Flujo:       {best_sol[2]:.0f} L/h (vs 3500 L/h)")

    return consumo_estandar, consumo_optimizado, ahorro_l, ahorro_pct




