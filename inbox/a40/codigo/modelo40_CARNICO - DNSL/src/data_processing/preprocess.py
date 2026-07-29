import pandas as pd
import numpy as np

# --- LÓGICA DE REFRIGERACIÓN ---

def extract_refrigeration_features(df):
    """
    Aplica las fórmulas termodinámicas 
    """
    df = df.copy()
    df = df.sort_values(["run_id", "time_min"])
        # Diferenciales térmicos
    df['T_error'] = df['T_cab'] - df['T_set']
    df['T_lift'] = df['T_cond_sat'] - df['T_evap_sat']
    # Si el condensador está sucio, la temperatura de saturación (T_cond_sat) se aleja mucho de la ambiente
    df['T_cond_approach'] = df['T_cond_sat'] - df['T_amb'] 
    df['T_spread'] = df['T_cond_sat'] - df['T_evap_sat']
    df['T_cab_meas_diff'] = df['T_cab'] - df['T_cab_meas'] 

    # Presiones y potencia
    df['P_ratio'] = df['P_dis_bar'] / (df['P_suc_bar'] + 0.1)
    df['Power_per_diff'] = df['P_comp_W'] / (df['T_lift'] + 0.1)

    # Eficiencia de transferencia: Vatios consumidos por cada grado de enfriamiento
    df['Q_est'] = df['P_comp_W'] / (df['T_cab'] - df['T_evap_sat'] + 1e-5)

    # Error de sensor 
    df['Sensor_error'] = df['T_cab'] - df['T_cab_meas'] 
    
    # Derivadas (Tendencias instantáneas)
    df['T_cab_grad'] = df.groupby('run_id')['T_cab'].diff().fillna(0)
    df['P_suc_rate'] = df.groupby('run_id')['P_suc_bar'].diff().fillna(0)

    df['Eff_vol'] = (df['T_evap_sat'] + 273.15) / (df['T_cond_sat'] + 273.15)
    df['P_suc_norm'] = df['P_suc_bar'] / df['T_amb']

    # Desviación de saturación (Crucial para detectar Aire/Incondensables - Clase 11)
    df['P_dis_error'] = df['P_dis_bar'] - (df['T_cond_sat'] * 0.25)

    # Diferencial de Subenfriamiento aproximado
    df['T_subcooling_approx'] = df['T_cond_sat'] - df['T_amb'] 
    # Relación Potencia-Presión
    df['Power_to_Pratio'] = df['P_comp_W'] / (df['P_ratio'] + 0.1)

    # Estabilidad de presión
    df['P_dis_volatility'] = df.groupby('run_id')['P_dis_bar'].transform(lambda x: x.rolling(window=5).std()).fillna(0)

    df["pressure_ratio"] = df["P_dis_bar"] / (df["P_suc_bar"] + 0.1)
    df["specific_work"] = df["P_comp_W"] / (df["Q_evap_W"]+ 0.1)
    df["cop_degradation"] = df["COP"] / df.groupby("run_id")["COP"].transform("max")

    # --- NUEVAS VARIABLES CRÍTICAS (PREDICTIVE) ---
    # Índice de eficiencia energética: relación entre el esfuerzo eléctrico y la capacidad de absorber calor
    df['EEI'] = df['P_comp_W'] / (df['T_cab'] - df['T_evap_sat'] + 0.1)
    # Carga térmica relativa (basada en inercia de producto si aplica)
    df['Thermal_Load_Index'] = (df['T_cab'] - df['T_evap_sat']) * df['P_ratio']

    return df

def create_refrigeration_lags(df):
    """
    Crea las variables temporales (Lags, Deltas y Rolling) para refrigeración.
    """
    df = df.copy().sort_values(['run_id', 'time_min'])
    
    # 1. Configuración de variables
    lag_features = ["P_dis_bar", "T_cond_sat"]
    lags = [5, 15, 45]
    rolling_features = ["P_dis_bar", "T_cond_sat", "EEI"]
    windows = [15, 30]

    # 2. Generación de Lags y Deltas
    for feature in lag_features:
        for lag in lags:
            # Shift por grupo
            df[f"{feature}_lag_{lag}"] = df.groupby("run_id")[feature].shift(lag)
            # Delta respecto al valor actual
            df[f"{feature}_delta_{lag}"] = df[feature] - df[f"{feature}_lag_{lag}"]

    # Lag específico de error de presión (Largo plazo)
    if "P_dis_error" in df.columns:
        df['P_dis_error_lag_100'] = df.groupby("run_id")["P_dis_error"].shift(100)

    # 3. Generación de Rolling Features (Mean y Std)
    for feature in rolling_features:
        for w in windows:
            # Std
            df[f"{feature}_roll_std_{w}"] = df.groupby("run_id")[feature].transform(
                lambda x: x.rolling(w, min_periods=w//2).std()
            )
            # Mean
            roll_mean = df.groupby("run_id")[feature].rolling(w, min_periods=w//2).mean()
            df[f"{feature}_roll_mean_{w}"] = roll_mean.reset_index(level=0, drop=True)

    # 4. Estabilidad Refinada (Ventana 20)
    # Presión
    df["P_dis_bar_roll_std_20"] = (
        df.groupby("run_id")["P_dis_bar"]
          .rolling(20, min_periods=10)
          .std()
          .reset_index(level=0, drop=True)
    )
    df["P_dis_bar_roll_mean_20"] = (
        df.groupby("run_id")["P_dis_bar"]
          .rolling(20, min_periods=10)
          .mean()
          .reset_index(level=0, drop=True)
    )
    
    # Índice de Inestabilidad
    df["Pdis_instability_20"] = df["P_dis_bar_roll_std_20"] / (df["P_dis_bar_roll_mean_20"] + 1e-5)

    # Inestabilidad en transferencia (Approach)
    if "T_cond_approach" in df.columns:
        df["cond_approach_std_20"] = (
            df.groupby("run_id")["T_cond_approach"]
              .rolling(20, min_periods=10)
              .std()
              .reset_index(level=0, drop=True)
        )

    # Limpieza final de NaNs generados
    df = df.dropna(subset=["Pdis_instability_20", "P_dis_bar_delta_15"])

    
    return df

def physics_indicators_refrigeration(df):
    
    df = df.copy()
    
    df['early_P_dis_error'] = df.groupby('run_id')['P_dis_error'].transform(lambda x: x.iloc[:100].mean())
    df['mean_P_dis_bar'] = df.groupby('run_id')['P_dis_bar'].transform('mean')
    

    
    return df

# --- LÓGICA DE AIREADO ---

def extract_aireado_features(df):
    """
    Ingeniería de variables específica para el sistema de aireado.
    """
    df_ext = df.copy()

    # --- FUNCIÓN AUXILIAR PSICROMÉTRICA ---
    def calculate_vpd(T, RH):
        """
        Calcula el Déficit de Presión de Vapor (VPD) en kPa.
        Fórmula de Tetens para presión de saturación.
        """
        # Presión de saturación (es)
        es = 0.61078 * np.exp((17.27 * T) / (T + 237.3))
        # Presión real de vapor (ea)
        ea = es * (RH / 100.0)
        return es - ea

    # VPD (Vapor Pressure Deficit) - Fuerza impulsora del secado
    # Esencial para detectar Clases 1 (Encostramiento) y 2 (Saturación)
    df_ext['VPD'] = calculate_vpd(df_ext['T_cab'], df_ext['RH_cab'])
    
    # Delta Higroscópico (Referencia: Ruiz-Ramirez, 2005)
    # Diferencia respecto a la humedad de equilibrio teórica
    df_ext['RH_error'] = df_ext['RH_cab'] - 75.0 
    
    # Ratio Aire/Carga (Referencia: Imre, 1974)
    # Evalúa si el flujo de aire es suficiente para la masa de embutido
    df_ext['Air_Flow_Ratio'] = df_ext['N_fan_Hz'] / (df_ext['Kg_embutido'] + 1.0)
    
    # Eficiencia de Evaporación (Referencia: Andrés et al., 2007)
    # Relación entre enfriamiento y deshumidificación
    df_ext['Evap_Eff_Index'] = (df_ext['T_cab'] - df_ext['T_evap_sat']) / (df_ext['RH_cab'] + 0.1)
    
    # Potencia Específica por Carga (Referencia: Toldrá, 2006)
    # Energía consumida por cada Kg de producto fresco
    df_ext['Specific_Power_Load'] = df_ext['P_comp_W'] / (df_ext['Kg_embutido'] + 1.0)
    
    # Indicador de Encostramiento (Predictor de Falla 1)
    # Aire alto + RH baja = Riesgo de Case Hardening
    df_ext['Encostramiento_Risk'] = df_ext['N_fan_Hz'] / (df_ext['RH_cab'] + 1.0)

    return df_ext

def create_aireado_lags(df):
    """
    Variables temporales específicas para el sistema de aireado.
    """
    features_to_lag = ['RH_cab', 'T_cab', 'N_fan_Hz', 'Evap_Eff_Index']
    lags_aireado = [10, 30, 60]
    df_lagged = df.copy()
    for feat in features_to_lag:
        for lag in lags_aireado:
            # Valor previo del sensor
            df_lagged[f"{feat}_lag_{lag}"] = df_lagged.groupby("run_id")[feat].shift(lag)
            # TENDENCIA: Delta de cambio (Crucial para ver si el secado se estanca)
            df_lagged[f"{feat}_delta_{lag}"] = df_lagged[feat] - df_lagged[f"{feat}_lag_{lag}"]
    
    # Rolling mean para suavizar ciclos de ventilación periódica (Imre, 1974)
    df_lagged['RH_roll_mean_20'] = df_lagged.groupby('run_id')['RH_cab'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    
    return df_lagged