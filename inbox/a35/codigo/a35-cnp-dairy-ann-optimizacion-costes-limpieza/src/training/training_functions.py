import torch
import os 
import copy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from src.predict.metrics import get_metrics

def train_function(model, optimizer, criterion, scheduler, X_train, y_train, X_val, y_val, epochs=300, scaler_y=None):
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 50
    best_model_state = None

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    print(f"Iniciando entrenamiento: {epochs} épocas máx. (Patience: {patience_limit})")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)

            # Desescalar para métricas interpretables en el log
            if scaler_y is not None:
                y_val_orig = scaler_y.inverse_transform(y_val_t.numpy())
                val_outputs_orig = scaler_y.inverse_transform(val_outputs.numpy())
            else:
                y_val_orig = y_val_t.numpy()
                val_outputs_orig = val_outputs.numpy()

            r2, mae, rmse, mae_relativo, rmse_mae_ratio = get_metrics(y_val_orig, val_outputs_orig)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}')
            print(f'Metrics: R2: {r2:.4f}, mae_relativo: {mae_relativo:.2f}%, RMSE/MAE: {rmse_mae_ratio:.2f}')

        if patience_counter >= patience_limit:
            print(f"Early stopping en epoch {epoch+1}. Recuperando mejor modelo...")
            break

    return best_model_state, best_val_loss


def split_and_save_splits(df, splits_path):
    os.makedirs(splits_path, exist_ok=True)

    features = ['temp_entrada_leche', 'temp_ambiente', 'temp_setpoint_leche', 'temp_proceso_leche',
          'temp_agua_servicio', 'flujo_leche_lh', 'horas_desde_limpieza', 'presion_diferencial_bar']
    target  = ['consumo_agua_l']

    X = df[features].values
    y = df[target].values.reshape(-1, 1)

    # 2. SPLIT TRAIN/VALIDATION/TEST (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, shuffle=True  # 0.176 * 0.85 ≈ 0.15
    )

# --- Guardado de datos originales (sin escalar) ---
    save_as_csv(X_train, features, os.path.join(splits_path, 'X_train.csv'))
    save_as_csv(X_test, features, os.path.join(splits_path, 'X_test.csv'))
    save_as_csv(X_val, features, os.path.join(splits_path, 'X_val.csv'))
    
    save_as_csv(y_train, [target], os.path.join(splits_path, 'y_train.csv'))
    save_as_csv(y_test, [target], os.path.join(splits_path, 'y_test.csv'))
    save_as_csv(y_val, [target], os.path.join(splits_path, 'y_val.csv'))

    print(f"Splits guardados correctamente en: {splits_path}")

    return X_train, X_test, X_val, y_train, y_test, y_val

def save_as_csv(data, column_names, file_path):
    pd.DataFrame(data, columns=column_names).to_csv(file_path, index=False)
