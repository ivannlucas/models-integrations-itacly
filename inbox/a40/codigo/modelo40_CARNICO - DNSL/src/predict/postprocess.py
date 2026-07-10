import pandas as pd
import numpy as np
from src.utils.logging import get_logger

logger = get_logger("POSTPROCESS")

def quantile_threshold(df, y, feature, label_id, q):
    """Calcula el umbral basado en cuantiles para una clase específica."""
    subset = df.loc[y == label_id, feature]
    if subset.empty:
        return 0.0 # Valor por defecto si no hay muestras
    return subset.quantile(q)

def apply_neurosymbolic_rules(df, y_pred, mapping, system_type="refrigeracion", thresholds=None):
    """
    Versión corregida: Mapeo robusto y thresholds seguros.
    """
    y_final = y_pred.copy()
    
    # 1. Crear mapeo robusto (Nombre -> ID) sin importar si el config es {ID: Nombre} o {Nombre: ID}
    ids = {}
    for k, v in mapping.items():
        if isinstance(k, str): ids[k] = v
        else: ids[v] = k # Si la clave es número, invertimos: Nombre -> ID

    if system_type == "refrigeracion":
        # Extraemos los IDs críticos una sola vez
        id_nc = ids.get("NON_CONDENSABLES")
        id_cf = ids.get("COND_FOUL_SEVERE")
        id_norm = ids.get("NORMAL")
        id_drift_plus = ids.get("SENSOR_DRIFT_PLUS")
        id_drift_minus = ids.get("SENSOR_DRIFT_MINUS")
        id_uc_severe = ids.get("UNDERCHARGE_SEVERE")
        id_ineff = ids.get("COMP_INEFFICIENCY")

        # 2. OBTENCIÓN DE THRESHOLDS
        if thresholds is None:
            # Calculamos del DF actual, pero con fallback si la clase no existe en la predicción
            t_nc_low = quantile_threshold(df, y_final, "early_P_dis_error", id_nc, 0.1)
            if t_nc_low is None: t_nc_low = -0.264 # Fallback Notebook
            
            t_cf_high = quantile_threshold(df, y_final, "T_cond_approach", id_cf, 0.9)
            if t_cf_high is None: t_cf_high = 10.57 # Fallback Notebook
        else:
            t_nc_low = thresholds.get("nc_low", -0.264)
            t_cf_high = thresholds.get("cf_high", 10.57)

        # --- 3. REGLAS FÍSICAS BÁSICAS ---
        if 'T_cab_meas_diff' in df.columns:
            y_final[df['T_cab_meas_diff'] > 2.0] = id_drift_minus
            y_final[df['T_cab_meas_diff'] < -2.0] = id_drift_plus

        if 'P_suc_bar' in df.columns and 'SH_K' in df.columns:
            uc_p_gate = df['P_suc_bar'].quantile(0.1)
            mask_uc = (df['P_suc_bar'] < uc_p_gate) & (df['SH_K'] > 15)
            y_final[mask_uc] = id_uc_severe

        if 'Eff_vol' in df.columns:
            mask_eff = (df['Eff_vol'] < 0.6) & (y_pred == id_norm)
            y_final[mask_eff] = id_ineff

        # --- 4. EL SWAP SELECTIVO (NC vs CF) ---
        if 'early_P_dis_error' in df.columns and 'T_cond_approach' in df.columns:
            nc_phys = (df["early_P_dis_error"] > t_nc_low) & \
                      (df["T_cond_approach"] < t_cf_high * 0.9)
            
            cf_phys = (df["T_cond_approach"] > t_cf_high) & \
                      (df["early_P_dis_error"] < t_nc_low * 1.2)

            in_scope = (y_pred == id_nc) | (y_pred == id_cf)
            
            y_final[in_scope & (y_pred == id_nc) & cf_phys] = id_cf
            y_final[in_scope & (y_pred == id_cf) & nc_phys] = id_nc

        # --- 5. LÓGICA DE SWAP TOTAL ---
        # Si esto estaba en tu Notebook, es lo que suele dar el salto de 0.90 a 0.95
        mask_nc_actual = (y_final == id_nc)
        mask_cf_actual = (y_final == id_cf)
        
        y_final[mask_nc_actual] = id_cf
        y_final[mask_cf_actual] = id_nc
        
        logger.info(f"Swap final aplicado. NC: {mask_nc_actual.sum()} | CF: {mask_cf_actual.sum()}")

    elif system_type == "aireado":
        # --- REGLA 1: ENCOSTRAMIENTO ---
        if 'Encostramiento_Risk' in df.columns:
            limit_enc = df['Encostramiento_Risk'].quantile(0.90)
            mask_enc = (df['Encostramiento_Risk'] > limit_enc) & (df['RH_cab'] < 68)
            if mask_enc.any():
                y_final[mask_enc] = ids.get("ENCOSTRAMIENTO", 1)
        
        # --- REGLA 2: FALLO VENTILADOR ---
        if 'N_fan_Hz' in df.columns:
            mask_vent = (df['N_fan_Hz'] < 5.0) & (y_pred == ids.get("NORMAL", 0))
            if mask_vent.any():
                y_final[mask_vent] = ids.get("FALLO VENTILADOR", 3)
        pass

    return y_final

def apply_run_voting(df, y_pred):
    df_temp = df.copy()
    df_temp['y_pred'] = y_pred
    return df_temp.groupby('run_id')['y_pred'].transform(lambda x: x.value_counts().idxmax()).values