import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from src.training.artifacts import load_artifacts
from src.predict.inference import predict_with_model

model, sX, sY, cfg = load_artifacts()
model.eval()

# Check T_out prediction with different T_in values (should matter!)
T_in_vals = [0.4, 2, 4, 6, 8]
print("T_out prediction at F=5500, T_serv=76, varying T_in:")
for t_in in T_in_vals:
    E, T = predict_with_model(5500, 76, t_in, 295, 0.63, model, sX, sY)
    print(f"  T_in={t_in:.1f}: E={E:.2f}, T_out={T:.2f}")

print()
print("T_out prediction at F=5500, T_serv=82, varying T_in:")
for t_in in T_in_vals:
    E, T = predict_with_model(5500, 82, t_in, 295, 0.63, model, sX, sY)
    print(f"  T_in={t_in:.1f}: E={E:.2f}, T_out={T:.2f}")

print()
print("T_out prediction at F=5500, T_serv=88, varying T_in:")
for t_in in T_in_vals:
    E, T = predict_with_model(5500, 88, t_in, 295, 0.63, model, sX, sY)
    print(f"  T_in={t_in:.1f}: E={E:.2f}, T_out={T:.2f}")

print()
# Check training data statistics
import pandas as pd
df = pd.read_csv('data/splits/train.csv')
print("Training data stats:")
print(f"  T_in_leche: min={df['T_in_leche'].min():.1f}, max={df['T_in_leche'].max():.1f}, mean={df['T_in_leche'].mean():.1f}")
print(f"  T_servicio: min={df['T_servicio'].min():.1f}, max={df['T_servicio'].max():.1f}, mean={df['T_servicio'].mean():.1f}")
print(f"  T_out_leche: min={df['T_out_leche'].min():.2f}, max={df['T_out_leche'].max():.2f}, mean={df['T_out_leche'].mean():.2f}")
print(f"  E_consumo: min={df['E_consumo'].min():.1f}, max={df['E_consumo'].max():.1f}, mean={df['E_consumo'].mean():.1f}")
print()

# Check what T_out range exists in training data for T_serv near 76
df_low = df[df['T_servicio'] < 78]
print(f"Training data with T_serv < 78: {len(df_low)} rows")
if len(df_low) > 0:
    print(f"  T_out: min={df_low['T_out_leche'].min():.2f}, max={df_low['T_out_leche'].max():.2f}")
    print(f"  T_in: min={df_low['T_in_leche'].min():.1f}, max={df_low['T_in_leche'].max():.1f}")
    print(f"  E_consumo: min={df_low['E_consumo'].min():.1f}, max={df_low['E_consumo'].max():.1f}")

# What about high T_serv?
df_high = df[df['T_servicio'] > 84]
print(f"\nTraining data with T_serv > 84: {len(df_high)} rows")
if len(df_high) > 0:
    print(f"  T_out: min={df_high['T_out_leche'].min():.2f}, max={df_high['T_out_leche'].max():.2f}")
    print(f"  E_consumo: min={df_high['E_consumo'].min():.1f}, max={df_high['E_consumo'].max():.1f}")
