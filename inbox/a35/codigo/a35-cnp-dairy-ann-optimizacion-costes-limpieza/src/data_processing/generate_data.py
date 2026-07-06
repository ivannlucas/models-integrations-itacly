import numpy as np
import pandas as pd
import os

def generate_data(data_saving_path, n_samples=5000):
    np.random.seed(42)

    # ----------------------------
    # 1. CONDICIONES AMBIENTALES
    # ----------------------------
    temp_ambiente = np.random.uniform(12, 32, n_samples)
    temp_entrada_leche = (
        4 + 0.05*(temp_ambiente-20)
        + np.random.normal(0,0.4,n_samples)
    )

    # ----------------------------
    # 2. VARIABLES DE PRODUCCIÓN
    # ----------------------------
    flujo_leche_lh = np.random.uniform(2500,5000,n_samples)
    horas_desde_limpieza = np.random.uniform(0,20,n_samples)

    # ----------------------------
    # 3. CONTROL DE TEMPERATURA
    # ----------------------------
    temp_setpoint_leche = np.random.uniform(72,78,n_samples)
    temp_proceso_leche = (
        temp_setpoint_leche
        + np.random.normal(0,0.25,n_samples)
    )
    delta_t_servicio = np.random.uniform(6,14,n_samples)
    temp_agua_servicio = temp_proceso_leche + delta_t_servicio

    # ----------------------------
    # 4. FOULING
    # ----------------------------
    fouling = 1 - np.exp(-horas_desde_limpieza/6)
    # mayor temperatura acelera fouling
    fouling *= np.exp((temp_proceso_leche-72)/20)

    # ----------------------------
    # 5. PRESIÓN DIFERENCIAL
    # ----------------------------
    presion_diferencial_bar = (
        0.25
        + 0.8*fouling
        + flujo_leche_lh/25000
        + np.random.normal(0,0.03,n_samples)
    )

    # ----------------------------
    # 6. BALANCE TÉRMICO
    # ----------------------------
    cp_leche = 3.9
    calor_necesario = (
        flujo_leche_lh
        * cp_leche
        * (temp_proceso_leche-temp_entrada_leche)
    )

    eficiencia_termica = 0.9/(1+fouling*0.3)

    # ----------------------------
    # 7. CONSUMO DE AGUA
    # ----------------------------
    delta_t_agua = 15
    cp_agua = 4.18
    # Definimos la duración del ciclo de pasteurización 
    duracion_ciclo_h = 1.0
    consumo_agua_enfriamiento = (calor_necesario/ (cp_agua*delta_t_agua))*duracion_ciclo_h
    consumo_agua_enfriamiento /= eficiencia_termica
    consumo_agua_limpieza = 300 + 900*fouling
    consumo_agua_total = (
        consumo_agua_enfriamiento
        + consumo_agua_limpieza
    )
    consumo_agua_total *= np.random.uniform(0.98,1.02,n_samples)

    # ----------------------------
    # 8. SEGURIDAD MICROBIOLÓGICA
    # ----------------------------
    volumen_retencion = 16
    tiempo_residencia = (
        volumen_retencion/flujo_leche_lh
    )*3600
    Z = 7
    Tref = 72
    pu = (
        tiempo_residencia
        * 10**((temp_proceso_leche-Tref)/Z)
    )
    indice_seguridad = (pu>=13).astype(int)

    # ----------------------------
    # DATAFRAME FINAL
    # ----------------------------

    final_dataframe= pd.DataFrame({
        'temp_entrada_leche': temp_entrada_leche,
        'temp_ambiente': temp_ambiente,
        'temp_setpoint_leche': temp_setpoint_leche,
        'temp_proceso_leche': temp_proceso_leche,
        'temp_agua_servicio': temp_agua_servicio,
        'flujo_leche_lh': flujo_leche_lh,
        'horas_desde_limpieza': horas_desde_limpieza,
        'presion_diferencial_bar': presion_diferencial_bar,
        'consumo_agua_l': consumo_agua_total,
        'valor_pu_microbiologico': pu,
        'indice_seguridad': indice_seguridad
    })

    # GUARDAMOS
    os.makedirs(os.path.dirname(data_saving_path), exist_ok=True)
    final_dataframe.to_csv(data_saving_path, index=False)
    print(f"Datos generados con éxito en {data_saving_path}")

    return final_dataframe
