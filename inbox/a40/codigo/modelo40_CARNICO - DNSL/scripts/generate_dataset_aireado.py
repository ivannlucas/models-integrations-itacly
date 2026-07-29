import pandas as pd
import numpy as np
import os
from pathlib import Path
import yaml

def generate_expert_aireado_dataset(n_rows=30000):
    np.random.seed(42)
    base_path = Path(__file__).resolve().parent.parent
    config_path = base_path / 'config' / 'config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    file_path = base_path / Path(config['aireado']['raw_data'])
    os.makedirs(file_path.parent, exist_ok=True)
    
    run_length = 200 
    n_runs = n_rows // run_length
    
    # 1. Crear estructura e inicializar columnas
    df = pd.DataFrame({
        'run_id': np.repeat(np.arange(n_runs), run_length),
        'time_min': np.tile(np.arange(run_length), n_runs),
        'fault_id': 0,
        'Kg_embutido': np.repeat(np.random.uniform(200, 1500, n_runs), run_length),
        'T_amb': 22.0 + np.random.normal(0, 1, n_rows),
        'T_set': 14.0, 
        'N_fan_Hz': 0.0,
        'RH_cab': 75.0,
        'T_cab': 16.0,
        'T_evap_sat': 8.0
    })

    # 2. Rellenar variables base con variabilidad estocástica
    # Introducimos un factor aleatorio por RUN para desacoplar la correlación 0.98
    random_factor = np.repeat(np.random.uniform(0.8, 1.2, n_runs), run_length)
    
    base_vent = 40 + 20 * np.sin(df['time_min'] * (2 * np.pi / 25))
    df['N_fan_Hz'] = base_vent + np.random.normal(0, 1.5, n_rows)
    
    # T_cab ahora tiene variabilidad individual por lote (menor correlación directa)
    df['T_cab'] = df['T_set'] + ((df['Kg_embutido'] * random_factor) / 750) + np.random.normal(0, 0.3, n_rows)
    df['T_evap_sat'] = df['T_cab'] - 8

  # Lógica de fallas progresivas
    for r in range(n_runs):
        mask = (df['run_id'] == r)
        t_prog = np.linspace(0, 1, run_length)
        # Generamos ruido específico para este lote
        ruido_lote = np.random.normal(0, 0.8, run_length) 
        
        tipo_falla = r % 4 
        df.loc[mask, 'fault_id'] = tipo_falla
        
        if tipo_falla == 1: # Encostramiento: más ruido en ventilación
            df.loc[mask, 'N_fan_Hz'] += (t_prog * 15) + np.random.normal(0, 1.2, run_length)
            df.loc[mask, 'RH_cab'] = 75 - (t_prog * 20) + ruido_lote
        elif tipo_falla == 2: # Hielo: mayor inestabilidad en evaporación
            df.loc[mask, 'T_evap_sat'] = (df.loc[mask, 'T_cab'] - 8) - (t_prog * 10) + np.random.normal(0, 0.5, run_length)
            df.loc[mask, 'RH_cab'] = 75 + (t_prog * 20) + ruido_lote
        elif tipo_falla == 3: # Ventilador: caída inestable
            df.loc[mask, 'N_fan_Hz'] *= (1 - (t_prog * 0.8))
            df.loc[mask, 'N_fan_Hz'] += np.random.normal(0, 2.0, run_length) # Vibración del ventilador roto
            df.loc[mask, 'RH_cab'] = 75 + (t_prog * 20) + ruido_lote
        else: # Normal
            df.loc[mask, 'RH_cab'] = 75 + (t_prog * 2) + ruido_lote
    # 4. Consumo energético desacoplado de la carga lineal
    # Añadimos un factor de eficiencia que decae levemente con el uso
    efficiency_loss = np.random.uniform(0.9, 1.05, n_rows)
    df['P_comp_W'] = ((df['Kg_embutido'] * 0.8) + (df['N_fan_Hz'] * 10)) * efficiency_loss + np.random.normal(0, 30, n_rows)
    
    df.to_csv(file_path, index=False)
    print(f"Dataset generado correctamente en: {file_path}")
    return df

generate_expert_aireado_dataset()