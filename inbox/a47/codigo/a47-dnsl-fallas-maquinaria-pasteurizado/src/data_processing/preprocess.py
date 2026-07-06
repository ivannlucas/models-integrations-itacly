import pandas as pd
import numpy as np
import os
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.logging import get_logger
from src.utils.target_mapping import map_cooler, map_valve, map_pump, map_acc

logger = get_logger(__name__)

def clean_and_resample(df_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Convierte el tiempo, corrige las etiquetas y resamplea a 10Hz.
    """
    logger.info("Convirtiendo tiempo y limpiando etiquetas...")
    df_raw = df_raw.reset_index(drop=True)
    df_raw['Time_Segundos'] = df_raw['Time'].round(1)

    # Mapeo de targets centralizado en src.utils.target_mapping

    df_raw['Target_Fouling'] = df_raw['Cooler_Condition'].apply(map_cooler)
    df_raw['Target_Valvula'] = df_raw['Valve_Condition'].apply(map_valve)
    df_raw['Target_Bomba'] = df_raw['Pump_Leakage'].apply(map_pump)
    df_raw['Target_Acumulador'] = df_raw['Hydraulic_Accumulator'].apply(map_acc)

    cols_targets = config['features']['cols_targets']
    cols_sensores = config['features']['cols_sensores_base']

    logger.info("Resampleando a 10Hz...")
    df_grouped = df_raw.groupby(['Cycle_ID', 'Time_Segundos'])[cols_sensores].mean().reset_index()
    df_targs = df_raw.groupby(['Cycle_ID'])[cols_targets].first().reset_index()   
    df_10hz = pd.merge(df_grouped, df_targs, on='Cycle_ID')

    return df_10hz

def split_data(df_10hz: pd.DataFrame, config: dict):
    """
    Realiza el split temprano de los ciclos (Train/Val/Test).
    """
    logger.info("Realizando split temprano (Train/Val/Test)...")
    unique_cycles = df_10hz['Cycle_ID'].unique()
    ts1 = config['data']['test_size_1']
    ts2 = config['data']['test_size_2']
    seed = config['data']['random_state']
    
    train_cycles, temp_cycles = train_test_split(unique_cycles, test_size=ts1, random_state=seed, shuffle=True)
    val_cycles, test_cycles = train_test_split(temp_cycles, test_size=ts2, random_state=seed, shuffle=True)
    
    logger.info(f"Split realizado. Train: {len(train_cycles)} | Val: {len(val_cycles)} | Test: {len(test_cycles)}")
    
    splits_dict = {
        "train_cycles": train_cycles.tolist(),
        "val_cycles": val_cycles.tolist(),
        "test_cycles": test_cycles.tolist()
    }
    
    ruta_splits = config['paths'].get('split_ids', 'data/splits/cycle_splits.json')
    
    os.makedirs(os.path.dirname(ruta_splits), exist_ok=True)
    with open(ruta_splits, "w") as f:
        json.dump(splits_dict, f, indent=4)
    
    logger.info(f"IDs de los splits guardados en: {ruta_splits}")
    return train_cycles, val_cycles, test_cycles

def save_processed_splits(df_final: pd.DataFrame, train_cycles: np.ndarray, val_cycles: np.ndarray, test_cycles: np.ndarray, config: dict):
    """
    Filtra df_final por los ciclos de cada split y los guarda en CSVs independientes.
    """
    logger.info("Guardando particiones (CSVs) por split...")
    
    splits_dir = config['paths'].get('splits_dir', 'data/splits/')
    os.makedirs(splits_dir, exist_ok=True)
    
    df_final[df_final['Cycle_ID'].isin(train_cycles)].to_csv(os.path.join(splits_dir, "train_split.csv"), index=False)
    df_final[df_final['Cycle_ID'].isin(val_cycles)].to_csv(os.path.join(splits_dir, "val_split.csv"), index=False)
    df_final[df_final['Cycle_ID'].isin(test_cycles)].to_csv(os.path.join(splits_dir, "test_split.csv"), index=False)
    
    logger.info(f"Datasets de splits guardados en: {splits_dir}")

def apply_digital_twin_and_augment(df_10hz: pd.DataFrame, train_cycles: np.ndarray, config: dict):
    """
    Aplica la lógica del gemelo digital (desplazamiento térmico a 65ºC) y
    Data Augmentation sólo al conjunto de entrenamiento.
    Retorna el DataFrame final y la media de TS1 en train.
    """
    # 1. Gemelo digital: Desplazar temperaturas
    mask_train = df_10hz['Cycle_ID'].isin(train_cycles)
    ts1_mean_train = df_10hz.loc[mask_train, 'TS1'].mean()
    
    logger.info("Desplazando temperaturas a la línea base de 65 ºC...")
    offset_temperatura = 65.0 - ts1_mean_train
    
    df_thermo = df_10hz.copy()
    df_thermo['TS1'] = df_thermo['TS1'] + offset_temperatura
    df_thermo['TS2'] = df_thermo['TS2'] + offset_temperatura

    # 2. Data Augmentation
    logger.info("Aplicando Data Augmentation (Ruido solo a Train)...")
    df_train_only = df_thermo[df_thermo['Cycle_ID'].isin(train_cycles)].copy()
    df_noisy = df_train_only.copy()
    df_noisy['Cycle_ID'] = df_noisy['Cycle_ID'] + 50000 

    noise_level = config['data']['noise_level']
    cols_sensores = config['features']['cols_sensores_base']
    for c_name in cols_sensores:
        std_dev = df_noisy[c_name].std()
        noise = np.random.normal(0, std_dev * noise_level, size=len(df_noisy))
        df_noisy[c_name] += noise

    df_aug = pd.concat([df_thermo, df_noisy], ignore_index=True)
    train_cycles_aug = np.concatenate([train_cycles, df_noisy['Cycle_ID'].unique()])
    
    return df_aug, train_cycles_aug, ts1_mean_train

def feature_engineering(df_aug: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calcula medias móviles, desviaciones, y lags para características temporales.
    """
    logger.info("Calculando características temporales (_rmean, _rstd, _lag)...")
    cols_sensores = config['features']['cols_sensores_base']
    cols_targets = config['features']['cols_targets']
    
    def engineer_features(group):
        group = group.sort_values('Time_Segundos') 
        X = group[cols_sensores]                    
        rmean = X.rolling(5, min_periods=1).mean().add_suffix('_rmean')
        rstd = X.rolling(5, min_periods=1).std().fillna(0).add_suffix('_rstd')
        lag = X.shift(1).bfill().add_suffix('_lag1') 
        return pd.concat([group, rmean, rstd, lag], axis=1)

    # Desactivamos el warning de pandas
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    df_final = df_aug.groupby('Cycle_ID', group_keys=False).apply(engineer_features).reset_index(drop=True)

    # Reordenamos columnas
    cols_nuevas = [c for c in df_final.columns if '_rmean' in c or '_rstd' in c or '_lag1' in c]
    cols_ordenadas = ['Cycle_ID', 'Time_Segundos'] + cols_sensores + cols_nuevas + cols_targets
    df_final = df_final[cols_ordenadas]
    
    return df_final

def prepare_tensors(df_final: pd.DataFrame, train_cycles_aug, val_cycles, test_cycles, config: dict):
    """
    Ajusta el scaler y crea los DataLoaders de PyTorch.
    Retorna los dataloaders, el scaler, y el feature_cols utilizado.
    """
    logger.info("Escalando variables y preparando tensores...")
    cols_targets = config['features']['cols_targets']
    feature_cols = [c for c in df_final.columns if c not in ['Cycle_ID', 'Time_Segundos', 'date'] + cols_targets]
    
    scaler = StandardScaler()
    mask_train_aug = df_final['Cycle_ID'].isin(train_cycles_aug)
    scaler.fit(df_final.loc[mask_train_aug, feature_cols])

    df_final[feature_cols] = scaler.transform(df_final[feature_cols])

    def create_tensors(df_in, cycle_ids):
        df_subset = df_in[df_in['Cycle_ID'].isin(cycle_ids)]
        target_subset = df_subset.groupby('Cycle_ID')[cols_targets].first()
        
        x_list, y_list = [], []
        for cid in df_subset['Cycle_ID'].unique():
            cycle_data = df_subset[df_subset['Cycle_ID'] == cid][feature_cols].values
            # Pad o Truncate según el config window size de 600
            window_size = config['data']['window_size']
            if len(cycle_data) > window_size: 
                cycle_data = cycle_data[:window_size]
            elif len(cycle_data) < window_size:
                pad = np.zeros((window_size - len(cycle_data), len(feature_cols)))
                cycle_data = np.vstack([cycle_data, pad])
                
            x_list.append(cycle_data)
            y_list.append(target_subset.loc[cid].values)
        
        X_t = torch.tensor(np.array(x_list), dtype=torch.float32).permute(0, 2, 1)
        y_t = torch.tensor(np.array(y_list), dtype=torch.long)
        return X_t, y_t

    X_train_pt, y_train_pt = create_tensors(df_final, train_cycles_aug)
    X_val_pt, y_val_pt = create_tensors(df_final, val_cycles)           
    X_test_pt, y_test_pt = create_tensors(df_final, test_cycles)        

    from torch.utils.data import DataLoader, TensorDataset
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(TensorDataset(X_train_pt, y_train_pt), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_pt, y_val_pt), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_pt, y_test_pt), batch_size=batch_size, shuffle=False)

    logger.info(f"Tensores generados. Train Shape: {X_train_pt.shape}")
    return train_loader, val_loader, test_loader, scaler, feature_cols
