import numpy as np
import pandas as pd

def apply_stress_test(X_df, stress_config):
    """
    Inyecta ruido gaussiano y picos de error (spikes) al conjunto de datos
    basándose en los parámetros del archivo de configuración global.
    """
    if not stress_config or not stress_config.get("active", False):
        return X_df.copy()

    X_noisy = X_df.copy()
    noise_level = stress_config.get("noise_level", 0.0)
    include_spikes = stress_config.get("include_spikes", False)
    
    # Asegurar reproducibilidad de la traza de ruido
    np.random.seed(42)

    for col in X_noisy.columns:
        # 1. Inyección de Ruido Gaussiano (Fluctuación continua)
        if noise_level > 0.0:
            scale = X_noisy[col].std() * noise_level
            if scale > 0:
                X_noisy[col] += np.random.normal(0, scale, X_noisy.shape[0])
                
        # 2. Inyección de Spikes (Anomalías puntuales físicas)
        if include_spikes:
            # 1 de cada 100 lecturas sufre una pérdida física o pico crítico
            mask = np.random.rand(len(X_noisy)) < 0.01
            if mask.any():
                X_noisy.loc[mask, col] *= np.random.choice([0.8, 1.2])
                
    return X_noisy