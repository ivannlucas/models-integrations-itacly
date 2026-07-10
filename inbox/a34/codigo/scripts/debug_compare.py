import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from src.training.artifacts import load_artifacts
from src.predict.inference import predict_with_model
import pandas as pd
import numpy as np

model, sX, sY, cfg = load_artifacts()
model.eval()

# Load old notebook results (from ga_evaluation.ipynb cached output)
# F_flow_ia_medio=5354, T_serv_ia_medio=80.42

# Load script results
df_script = pd.read_csv('data/predictions/evaluation_rt_hist_vs_ia.csv')
print("Script results summary:")
print(f"  IA_F_flow: mean={df_script['IA_F_flow'].mean():.1f}, std={df_script['IA_F_flow'].std():.1f}")
print(f"  IA_T_servicio: mean={df_script['IA_T_servicio'].mean():.1f}, std={df_script['IA_T_servicio'].std():.1f}")
print(f"  IA_E_consumo: mean={df_script['IA_E_consumo'].mean():.1f}")
print(f"  IA_T_out: mean={df_script['IA_T_out'].mean():.2f}, min={df_script['IA_T_out'].min():.2f}")
print()

# Check first 5 test rows and verify predictions
df_test = pd.read_csv('data/splits/test.csv')
print("First 5 test rows with model predictions for HISTORICAL setpoints:")
for i in range(5):
    row = df_test.iloc[i]
    T_in = float(row['T_in_leche'])
    F_hist = float(row['F_flow'])
    T_serv_hist = float(row['T_servicio'])
    t_ciclo = float(row['t_ciclo'])
    dp = float(row['Delta_P'])
    
    E_pred, T_pred = predict_with_model(F_hist, T_serv_hist, T_in, t_ciclo, dp, model, sX, sY)
    
    print(f"  Row {i}: T_in={T_in}, F_hist={F_hist:.0f}, T_serv_hist={T_serv_hist:.1f}")
    print(f"           E_hist={row['E_consumo']:.2f}, E_pred={E_pred:.2f}")
    print(f"           T_out_hist={row['T_out_leche']:.2f}, T_out_pred={T_pred:.2f}")
print()

# Now check: what does the model predict at the script's recommended setpoints?
print("First 5 script recommendations verified:")
for i in range(5):
    row_script = df_script.iloc[i]
    T_in = float(row_script['T_in_leche'])
    F_ia = float(row_script['IA_F_flow'])
    T_serv_ia = float(row_script['IA_T_servicio'])
    t_ciclo = float(row_script['t_ciclo'])
    dp = float(row_script['Delta_P'])
    
    E_check, T_check = predict_with_model(F_ia, T_serv_ia, T_in, t_ciclo, dp, model, sX, sY)
    
    print(f"  Row {i}: IA_F={F_ia}, IA_Tserv={T_serv_ia}")
    print(f"           IA_E={row_script['IA_E_consumo']:.4f}, recheck_E={E_check:.4f}")
    print(f"           IA_Tout={row_script['IA_T_out']:.2f}, recheck_Tout={T_check:.2f}")
print()

# Check if ALL rows are at boundaries
unique_F = df_script['IA_F_flow'].unique()
unique_T = df_script['IA_T_servicio'].unique()
print(f"Unique IA_F_flow values: {len(unique_F)} ({unique_F[:10]}...)")
print(f"Unique IA_T_servicio values: {len(unique_T)} ({unique_T[:10]}...)")
print()

# Check rows where T_in is very low (might need higher T_serv)
print("Rows with T_in < 2 (cold milk, needs more heating):")
cold = df_test[df_test['T_in_leche'] < 2.0].iloc[:5]
for i, row in cold.iterrows():
    T_in = float(row['T_in_leche'])
    F_hist = float(row['F_flow'])
    T_serv_hist = float(row['T_servicio'])
    t_ciclo = float(row['t_ciclo'])
    dp = float(row['Delta_P'])
    
    # Check at boundary (5500, 76)
    E1, T1 = predict_with_model(5500, 76, T_in, t_ciclo, dp, model, sX, sY)
    # Check at higher T_serv
    E2, T2 = predict_with_model(5500, 82, T_in, t_ciclo, dp, model, sX, sY)
    E3, T3 = predict_with_model(5500, 88, T_in, t_ciclo, dp, model, sX, sY)
    
    print(f"  T_in={T_in:.1f}: (F=5500, T=76)->E={E1:.1f},T_out={T1:.2f} | (T=82)->E={E2:.1f},T_out={T2:.2f} | (T=88)->E={E3:.1f},T_out={T3:.2f}")
