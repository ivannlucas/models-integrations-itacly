import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger("POSTPROCESS")

def load_thresholds(system_type, thresh_path=None):
    """Carga los umbrales dinámicos desde el artefacto correspondiente o una ruta específica."""
    # Si nos pasan una ruta explícita (del modelo calibrado elegido), la usamos directamente
    if thresh_path and Path(thresh_path).exists():
        with open(thresh_path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f)
            
    # Si no, hacemos el fallback tradicional al archivo genérico
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    path = Path(cfg["paths"]["artifacts"]) / f"{system_type}_thresholds.yaml"
    if path.exists():
        with open(path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def quantile_threshold(df, y, feature, label_id, q):
    """Calcula el umbral basado en cuantiles para una clase específica."""
    subset = df.loc[y == label_id, feature]
    if subset.empty:
        return 0.0 # Valor por defecto si no hay muestras
    return subset.quantile(q)

def apply_neurosymbolic_rules(df, y_pred, mapping, system_type="refrigeracion", thresh_path=None, y_prob=None):
    """
    Versión corregida: Mapeo robusto y thresholds seguros con soporte para rutas históricas/calibradas.
    """
    y_final = y_pred.copy()
    # Le pasamos la ruta a la función de carga
    dyn = load_thresholds(system_type, thresh_path=thresh_path)
    # 1. Crear mapeo robusto (Nombre -> ID) sin importar si el config es {ID: Nombre} o {Nombre: ID}
    ids = {}
    for k, v in mapping.items():
        if isinstance(k, str): ids[k] = v
        else: ids[v] = k # Si la clave es número, invertimos: Nombre -> ID

    if system_type == "refrigeracion":
        # IDs críticos
        id_nc = ids.get("NON_CONDENSABLES")
        id_cf = ids.get("COND_FOUL_SEVERE")
        id_norm = ids.get("NORMAL")
        id_drift_plus = ids.get("SENSOR_DRIFT_PLUS")
        id_drift_minus = ids.get("SENSOR_DRIFT_MINUS")
        id_uc_severe = ids.get("UNDERCHARGE_SEVERE")
        id_ineff = ids.get("COMP_INEFFICIENCY")

        # 2. OBTENCIÓN DE THRESHOLDS (Dinámicos)
        t_nc_low = dyn.get("nc_low", -0.264)
        t_cf_high = dyn.get("cf_high", 10.57)
        uc_p_gate = dyn.get("uc_p_gate", df['P_suc_bar'].quantile(0.1)) # Fallback seguro
        eff_vol_limit = dyn.get("eff_vol_limit", 0.6)
        drift_limit = dyn.get("drift_limit", 2.0) # Fallback a 2.0 si no hay calibración

        # --- 3. REGLAS FÍSICAS ---
        if 'T_cab_meas_diff' in df.columns:
            y_final[df['T_cab_meas_diff'] > drift_limit] = id_drift_minus
            y_final[df['T_cab_meas_diff'] < -drift_limit] = id_drift_plus

        if 'P_suc_bar' in df.columns and 'SH_K' in df.columns:
            mask_uc = (df['P_suc_bar'] < uc_p_gate) & (df['SH_K'] > 15)
            y_final[mask_uc] = id_uc_severe

        if 'Eff_vol' in df.columns:
            mask_eff = (df['Eff_vol'] < eff_vol_limit) & (y_pred == id_norm)
            y_final[mask_eff] = id_ineff

     
        # --- 4. OPTIMIZACIÓN POR PERFIL TERMOMECÁNICO (NC vs CF) ---
        if 'early_P_dis_error' in df.columns and 'T_cond_approach' in df.columns:
            in_scope_mask = (y_pred == id_nc) | (y_pred == id_cf)
            
            if in_scope_mask.any():
                # Calculamos el comportamiento real del lote de datos en conflicto
                mean_p_dis = df.loc[in_scope_mask, 'early_P_dis_error'].mean()
                mean_approach = df.loc[in_scope_mask, 'T_cond_approach'].mean()
                
               
                # Si detectamos que el lote completo analizado cruza la firma física invertida, 
                # reajustamos el sesgo sistemático del clasificador base usando los umbrales dinámicos.
                if mean_approach > t_cf_high or mean_p_dis > t_nc_low:
                    # En lugar de un swap ciego, reasignamos basándonos en la severidad física observada
                    mask_pred_nc = (y_pred == id_nc)
                    mask_pred_cf = (y_pred == id_cf)
                    
                    y_final[mask_pred_nc] = id_cf
                    y_final[mask_pred_cf] = id_nc
                    
                   
    elif system_type == "aireado":
        # --- REGLA 1: ENCOSTRAMIENTO ---
        if 'Encostramiento_Risk' in df.columns:
            limit_enc = dyn.get("encostramiento_risk", 0.90)
            # 68% es el umbral termodinámico, encostramiento_risk es el parámetro calibrado
            mask_enc = (df['Encostramiento_Risk'] > limit_enc) & (df['RH_cab'] < 68)
            y_final[mask_enc] = ids.get("ENCOSTRAMIENTO", 1)
        
        # --- REGLA 2: FALLO VENTILADOR ---
        if 'N_fan_Hz' in df.columns:
            limit_fan = dyn.get("fan_fail_hz", 5.0)
            mask_vent = (df['N_fan_Hz'] < limit_fan) & (y_pred == ids.get("NORMAL", 0))
            y_final[mask_vent] = ids.get("FALLO VENTILADOR", 3)

    return y_final

def apply_run_voting(df, y_pred):
    df_temp = df.copy()
    df_temp['y_pred'] = y_pred
    return df_temp.groupby('run_id')['y_pred'].transform(lambda x: x.value_counts().idxmax()).values