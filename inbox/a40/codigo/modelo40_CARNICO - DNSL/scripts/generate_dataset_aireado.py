import pandas as pd
import numpy as np
import os
from pathlib import Path
import yaml

def generate_expert_aireado_dataset(n_rows=30000):
    """
    Generador de datos sintéticos para equipos de aireado de embutidos.
    Lógica basada en: Toldrá, F. (2010). Handbook of Meat Processing. Wiley-Blackwell.
    """
    np.random.seed(42)
    base_path = Path(__file__).resolve().parent.parent
    config_path = base_path / 'config' / 'config.yaml'
    

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    file_rel_path = Path(config['aireado']['raw_data'])
    full_file_path = base_path / file_rel_path
    output_path = full_file_path
    
        
    if not os.path.exists(output_path.parent): 
        os.makedirs(output_path.parent)
    file_name = os.path.join(output_path.parent, 'dataset_aireado.csv')
    
    run_length = 100
    n_runs = n_rows // run_length
    run_ids = np.repeat(np.arange(n_runs), run_length)
    time_min = np.tile(np.arange(run_length), n_runs)
    
    # --- LÓGICA DE CARGA Y DIFUSIVIDAD ---
    # REF: Gou, P., Comaposada, J., & Arnau, J. (2004). Moisture diffusivity in the lean tissue 
    # of dry-cured ham at different process times. Meat Science, 67(2), 203-209. DOI https://doi.org/10.1016/j.meatsci.2003.10.007
    # Justificación: La masa del producto determina la resistencia térmica y la inercia de humedad.
    carga_kg = np.random.uniform(200, 1500, n_runs)
    df_carga = pd.DataFrame({'run_id': np.arange(n_runs), 'Kg_embutido': carga_kg})
    
    df = pd.DataFrame({
        'run_id': run_ids,
        'time_min': time_min,
        'T_amb': 22 + np.random.normal(0, 1, n_rows),
        'T_set': 14.0,
        'fault_id': 0
    })
    df = df.merge(df_carga, on='run_id')

    # --- LÓGICA DE VENTILACIÓN PERIÓDICA ---
    # REF: Imre, L. (1974). Drying of salami sorts. In Handbook of Drying (in Hungarian). 
    # Budapest: Műszaki Kiadó.
    # Justificación: El sistema utiliza ciclos de "Rinse" (aire rápido) y "Saturación" (reposo)
    # para equilibrar la aw superficial y evitar el gradiente de humedad excesivo.
    vent_cycle = 40 + 20 * np.sin(time_min * (2 * np.pi / 25)) 
    df['N_fan_Hz'] = vent_cycle + np.random.normal(0, 1, n_rows)
    
    # --- LÓGICA DE PSICROMETRÍA (RH/T) ---
    # REF: Incze, K. (2004). Dry and semi-dry sausages. In Encyclopedia of Meat Sciences. 
    # Elsevier Academic Press.
    # Justificación: La RH_cab debe oscilar en oposición al ventilador para permitir 
    # la migración de agua desde el núcleo hacia la superficie.
    inercia = df['Kg_embutido'] / 1500
    df['T_cab'] = df['T_set'] + (2 * inercia) + np.random.normal(0, 0.2, n_rows)
    df['RH_cab'] = 75 + (10 * inercia) - (0.2 * df['N_fan_Hz']) + np.random.normal(0, 2, n_rows)
    df['T_evap_sat'] = df['T_cab'] - 8

    # --- FALLA 1: ENCOSTRAMIENTO (CASE HARDENING) ---
    # REF: Ruiz-Ramirez, J., Serra, X., Arnau, J., & Gou, P. (2005). Profiles of water content, 
    # water activity and texture in crusted dry-cured loin. Meat Science, 69(3), 519-525.
    # Justificación: Un flujo de aire excesivo con RH < 60% colapsa los poros superficiales 
    # creando una costra que detiene el secado.
    mask_enc = (df['run_id'] % 10 == 1)
    df.loc[mask_enc, 'fault_id'] = 1
    df.loc[mask_enc, 'N_fan_Hz'] += 20 
    df.loc[mask_enc, 'RH_cab'] -= 25  

    # --- FALLA 2: SATURACIÓN POR HIELO (BLOQUEO DE CONDENSACIÓN) ---
    # REF: Andrés, A., Barat, J. M., Grau, J., & Fito, P. (2007). Principles of drying and smoking. 
    # In Handbook of Fermented Meat and Poultry. Blackwell Publishing.
    # Justificación: Si el evaporador se congela, pierde capacidad de deshumidificación (absorción).
    mask_ice = (df['run_id'] % 10 == 2)
    df.loc[mask_ice, 'fault_id'] = 2
    df.loc[mask_ice, 'T_evap_sat'] -= 12
    df.loc[mask_ice, 'RH_cab'] = 98.0 

    # --- FALLA 3: FALLO MOTOR VENTILADOR ---
    # REF: Heinz, G., & Hautzinger, P. (2007). Meat processing technology for small- to 
    # medium-scale food enterprises. FAO.
    # Justificación: Sin flujo de aire masivo (convección forzada), el agua no se retira de la 
    # superficie del embutido, facilitando el crecimiento de moho indeseado.
    mask_fan = (df['run_id'] % 10 == 3)
    df.loc[mask_fan, 'fault_id'] = 3
    df.loc[mask_fan, 'N_fan_Hz'] = np.random.uniform(0, 2, mask_fan.sum())
    df.loc[mask_fan, 'RH_cab'] = 99.0

    # --- CÁLCULO DE CONSUMO ENERGÉTICO ---
    # REF: Toldrá, F. (2006). The role of muscle enzymes in dry-cured meat products with 
    # different drying conditions. Trends in Food Science and Technology, 17, 164-172.
    df['P_comp_W'] = (df['Kg_embutido'] * 0.8) + (df['N_fan_Hz'] * 10) + np.random.normal(0, 30, n_rows)
    
    df.to_csv(file_name, index=False)
    print(f"Dataset generado exitosamente en: {file_name}")
    return df

# Ejecución
generate_expert_aireado_dataset()