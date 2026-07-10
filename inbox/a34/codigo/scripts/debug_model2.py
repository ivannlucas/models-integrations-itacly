import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Compare raw data generated
df_raw = pd.read_csv('data/raw/pasteurizacion_dataset_simulado.csv')
df_proc = pd.read_csv('data/processed/final_data_sim.csv')

print("Raw data stats:")
print(f"  Shape: {df_raw.shape}")
print(f"  T_servicio: min={df_raw['T_servicio'].min():.2f}, max={df_raw['T_servicio'].max():.2f}, mean={df_raw['T_servicio'].mean():.2f}")
print(f"  T_out_leche: min={df_raw['T_out_leche'].min():.2f}, max={df_raw['T_out_leche'].max():.2f}")
print(f"  E_consumo: min={df_raw['E_consumo'].min():.1f}, max={df_raw['E_consumo'].max():.1f}")
print(f"  F_flow: min={df_raw['F_flow'].min():.1f}, max={df_raw['F_flow'].max():.1f}")
print()

# Check what the old notebook data looked like (from old train_metrics context)
# The old model was trained with T_servicio range from PID control (80-86 range)
# But GA searches T_servicio in (76, 95)

# Compare with test split
df_test = pd.read_csv('data/splits/test.csv')
print("Test data stats:")
print(f"  T_servicio: min={df_test['T_servicio'].min():.2f}, max={df_test['T_servicio'].max():.2f}")
print(f"  T_out_leche: min={df_test['T_out_leche'].min():.2f}, max={df_test['T_out_leche'].max():.2f}")
print()

# Check: does the notebook ga_evaluation use test.csv? 
# Yes: DATA_PATH = "../../data/splits/test.csv"

# The issue: notebook results show T_serv_ia_medio=80.42
# But training data has T_servicio min=80.6
# So the old notebook model was trained with a DIFFERENT dataset (V2/V3 simulator)
# that had T_servicio in range [76, 95] or wider

# Let's check if old ga_evaluation results exist
df_old = pd.read_csv('data/predictions/evaluation_rt_hist_vs_ia.csv')
print("Current script results (from optimize.py):")
print(f"  IA_F_flow: mean={df_old['IA_F_flow'].mean():.1f}, unique={df_old['IA_F_flow'].nunique()}")
print(f"  IA_T_servicio: mean={df_old['IA_T_servicio'].mean():.1f}, unique={df_old['IA_T_servicio'].nunique()}")
print()

# The key problem: V3.2 simulator generates T_servicio via PID control
# which only produces values in narrow range [80.6, 86.3]
# But the GA searches [76, 95]
# Model extrapolates badly outside training range

print("=== DIAGNOSIS ===")
print("Training data T_servicio range: [80.6, 86.3]")
print("GA search range for T_servicio: [76.0, 95.0]")
print("Model was NEVER trained on T_servicio < 80.6 or > 86.3")
print("Predictions at T_serv=76 are pure extrapolation (Tanh outputs constant ~72.69)")
print()
print("This is why GA converges to (F=5500, T_serv=76) for all instances:")
print("  - Model predicts T_out=72.69 (just above 72.3 constraint) even at T_serv=76")
print("  - Model predicts lowest E_consumo at lowest T_servicio")  
print("  - These are ARTIFACTS of extrapolation, not real behavior")
