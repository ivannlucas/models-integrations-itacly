import numpy as np
import torch
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def get_metrics(y_real, y_pred, save_path=None):
    if torch.is_tensor(y_real):
        y_real = y_real.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    # ---------------------------------------------
    r2 = r2_score(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae_relativo = (mae / np.mean(y_real)) * 100
    rmse_mae_ratio = rmse/mae 

    if save_path:
        # Guardar reporte de métricas
        with open(os.path.join(save_path), 'w', encoding='utf-8') as f:
            f.write("================================================\n")
            f.write("INFORME DE MÉTRICAS - PROYECTO DATAGIA\n")
            f.write("================================================\n")
            f.write(f"R2 Score:      {r2:.4f}\n")
            f.write(f"MAE:           {mae:.2f} L\n")
            f.write(f"RMSE:          {rmse:.2f} L\n")
            f.write(f"MAE Relativo:  {mae_relativo:.2f}%\n")
            f.write(f"RMSE/MAE:      {rmse_mae_ratio:.2f}\n")
            f.write("------------------------------------------------\n")
            f.write(f"Veredicto KPI: {'APTO' if r2 > 0.90 else 'REVISAR'}\n")

    return r2, mae, rmse, mae_relativo, rmse_mae_ratio
