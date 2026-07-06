"""
fine_tuner.py — Módulo de Calibración / Fine-Tuning del Modelo DNSL

Propósito
---------
Adaptar los pesos del modelo preentrenado con datos de aceite hidráulico
(laboratorio UCI) a los datos reales de una planta pasteurizadora con leche.

Fundamento técnico (Transfer Learning por congelación selectiva)
----------------------------------------------------------------
El backbone CNN (`CNN_Pasteurizer.features`) ya ha aprendido a detectar
formas de onda características de fallos (picos de presión, transitorios de
caudal, oscilaciones de temperatura). Estas representaciones son **invariantes
al fluido** y no necesitan reaprenderse.

Lo que SÍ cambia con el fluido (aceite → leche) son los umbrales de decisión
final, ya que la leche tiene:
  - Mayor densidad  (~1.03 kg/L vs ~0.87 kg/L del aceite)
  - Mayor viscosidad cinemática (~2–3 cSt vs ~46 cSt del aceite a 40°C)
  - Mayor calor específico (~3.93 kJ/kg·K vs ~1.88 kJ/kg·K del aceite)

Esto desplaza las curvas de eficiencia hidráulica y térmica, haciendo que los
umbrales aprendidos durante el entrenamiento sean incorrectos en producción.

Estrategia de calibración
--------------------------
1. Congelar el backbone: `CNN_Pasteurizer.features`  →  requires_grad = False
2. Descongelar las cabezas de clasificación:
      dropout_final, head_fouling, head_valvula, head_bomba, head_acumulador
3. Entrenar con learning rate muy bajo (×10 menor) y Early Stopping automático.
4. Guardar el modelo calibrado como artefacto separado para no sobreescribir
   el modelo base.

Con unos pocos cientos de ciclos reales de planta es suficiente para que las
4 cabezas lineales ajusten sus hiperplanos de decisión a la nueva distribución.
"""

import os
import copy
import time
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.model import CNN_Pasteurizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers de preprocesamiento (sin Data Leakage: scaler ya fijado en train)
# ---------------------------------------------------------------------------

def _build_tensor_from_df(df: pd.DataFrame, feature_cols: list, scaler, config: dict):
    """
    Convierte un DataFrame de planta (un ciclo por Cycle_ID) en un tensor
    PyTorch listo para el modelo.

    Aplica el mismo feature engineering que el pipeline de entrenamiento
    (rolling mean, rolling std, lag-1), usa el scaler YA fijado en
    entrenamiento (no hay re-fit) y genera los tensores con padding/truncado
    al mismo window_size del modelo.

    Returns
    -------
    X_tensor : torch.Tensor  shape (n_ciclos, n_features, window_size)
    y_tensor : torch.Tensor  shape (n_ciclos, 4)   ← solo si hay etiquetas
    """
    cols_targets = config['features']['cols_targets']
    cols_sensores = config['features']['cols_sensores_base']
    window_size = config['data']['window_size']

    has_labels = all(t in df.columns for t in cols_targets)

    # Ordenar por tiempo dentro de cada ciclo
    if 'Cycle_ID' in df.columns:
        group_col = 'Cycle_ID'
    else:
        # Si no hay Cycle_ID, tratamos todo como un único ciclo
        df = df.copy()
        df['Cycle_ID'] = 0
        group_col = 'Cycle_ID'

    time_col = 'Time_Segundos' if 'Time_Segundos' in df.columns else 'Time'

    x_list, y_list = [], []

    for cid, group in df.groupby(group_col):
        group = group.sort_values(time_col)

        # Feature engineering (mismo que preprocess.py)
        X_base = group[cols_sensores]
        rmean = X_base.rolling(5, min_periods=1).mean().add_suffix('_rmean')
        rstd  = X_base.rolling(5, min_periods=1).std().fillna(0).add_suffix('_rstd')
        lag   = X_base.shift(1).bfill().add_suffix('_lag1')
        # Construir df_feat solo con las columnas base y las calculadas
        cols_to_keep = [group_col, time_col] + cols_sensores
        if has_labels:
            cols_to_keep += cols_targets
            
        # Filtramos para no duplicar columnas si el CSV de planta ya las traía calculadas
        df_feat = pd.concat([group[cols_to_keep].reset_index(drop=True),
                             rmean.reset_index(drop=True),
                             rstd.reset_index(drop=True),
                             lag.reset_index(drop=True)], axis=1)

        # Selección y escalado (sin re-fit del scaler)
        try:
            X_raw = df_feat[feature_cols].values
        except KeyError as e:
            logger.error(f"Faltan columnas en los datos de planta: {e}")
            continue

        X_scaled = scaler.transform(X_raw)

        # Padding / truncado al window_size del modelo
        n_steps = X_scaled.shape[0]
        if n_steps < window_size:
            pad = np.zeros((window_size - n_steps, X_scaled.shape[1]))
            X_ready = np.vstack([X_scaled, pad])
        else:
            X_ready = X_scaled[:window_size, :]

        x_list.append(X_ready)

        if has_labels:
            y_list.append(group[cols_targets].iloc[0].values.astype(int))

    if not x_list:
        raise ValueError("No se pudo construir ningún tensor. Revisa el formato del CSV de planta.")

    X_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).permute(0, 2, 1)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.long) if y_list else None

    return X_tensor, y_tensor


# ---------------------------------------------------------------------------
# Función principal de Fine-Tuning
# ---------------------------------------------------------------------------

def run_fine_tuning(
    train_csv: str,
    val_csv: str,
    config: dict,
    fluid_density_kg_l: float = None,
    fluid_cp_kj_kgK: float = None,
    ft_epochs: int = None,
    ft_patience: int = None,
):
    """
    Calibra el modelo DNSL preentrenado adaptando las capas de decisión
    a los datos reales de la planta pasteurizadora.

    Parámetros
    ----------
    train_csv : str
        Ruta al CSV con ciclos reales de planta para entrenamiento del
        fine-tuning. Debe tener las columnas de sensores y opcionalmente
        las columnas de etiquetas (Target_*).
    val_csv : str
        Ruta al CSV con ciclos reales de planta para validación/early stopping.
    config : dict
        Configuración global del proyecto (leída desde config/config.yaml).
    fluid_density_kg_l : float
        Densidad del fluido real en kg/L.
        Por defecto 1.03 (leche entera aproximada).
    fluid_cp_kj_kgK : float
        Calor específico del fluido real en kJ/(kg·K).
        Por defecto 3.93 (leche entera aproximada).
    ft_epochs : int
        Número máximo de épocas de calibración. Por defecto 50.
    ft_patience : int
        Paciencia del Early Stopping. Por defecto 7.

    Artefactos generados
    --------------------
    models/artifacts/neurosymbolic_cnn_finetuned.pth
        Pesos del modelo calibrado. El modelo base NO se sobreescribe.
    models/artifacts/finetuning_report.txt
        Resumen textual del proceso (épocas, pérdida final, parámetros de fluido).
    """

    # ------------------------------------------------------------------
    # 0. Resolución de hiperparámetros (CLI > config.yaml > defaults)
    # ------------------------------------------------------------------
    ft_config = config.get('fine_tuning', {})

    if fluid_density_kg_l is None:
        fluid_density_kg_l = ft_config.get('fluid_density_kg_l', 1.03)
    if fluid_cp_kj_kgK is None:
        fluid_cp_kj_kgK = ft_config.get('fluid_cp_kj_kgK', 3.93)
    if ft_epochs is None:
        ft_epochs = ft_config.get('epochs', 50)
    if ft_patience is None:
        ft_patience = ft_config.get('patience', 7)

    lr_divisor = ft_config.get('lr_divisor', 10)
    ft_batch_size = ft_config.get('batch_size', 32)
    output_model_name = ft_config.get('output_model_name', 'neurosymbolic_cnn_finetuned.pth')

    logger.info("=" * 65)
    logger.info("   INICIO DEL PROCESO DE CALIBRACIÓN / FINE-TUNING")
    logger.info("=" * 65)
    logger.info(f"   Fluido real -> Densidad: {fluid_density_kg_l} kg/L | Cp: {fluid_cp_kj_kgK} kJ/(kg·K)")
    logger.info(f"   Datos de entrenamiento: {train_csv}")
    logger.info(f"   Datos de validación:    {val_csv}")

    # ------------------------------------------------------------------
    # 1. Selección de dispositivo
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU CUDA detectada. Calibración en GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple Silicon MPS detectado. Calibración en MPS.")
    else:
        device = torch.device("cpu")
        logger.warning("No se detectó GPU. Calibración en CPU (proceso más lento).")

    # ------------------------------------------------------------------
    # 2. Cargar artefactos preentrenados
    # ------------------------------------------------------------------
    models_dir = config['paths']['models_dir']
    base_model_path = os.path.join(models_dir, "neurosymbolic_cnn.pth")
    scaler_path     = os.path.join(models_dir, "scaler_cnn_dns.pkl")
    feature_path    = os.path.join(models_dir, "feature_columns.pkl")

    for path in [base_model_path, scaler_path, feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró el artefacto requerido: {path}\n"
                "Asegúrese de haber ejecutado el entrenamiento base antes de calibrar."
            )

    scaler       = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_path)
    n_canales    = len(feature_cols)
    n_classes    = config['training'].get('n_classes', 3)
    dropout_prob = config['training'].get('dropout_rate', 0.5)

    model = CNN_Pasteurizer(
        n_sensors=n_canales,
        n_classes=n_classes,
        dropout_prob=dropout_prob
    ).to(device)
    model.load_state_dict(
        torch.load(base_model_path, map_location=device, weights_only=True)
    )
    logger.info("Modelo base y artefactos cargados correctamente.")

    # ------------------------------------------------------------------
    # 3. Transfer Learning: congelar backbone, descongelar cabezas
    # ------------------------------------------------------------------
    # Congelar todo primero
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar SOLO las capas de decisión final
    layers_to_unfreeze = [
        model.dropout_final,
        model.head_fouling,
        model.head_valvula,
        model.head_bomba,
        model.head_acumulador,
    ]
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Parámetros congelados: {n_total - n_trainable:,} | "
        f"Parámetros a calibrar: {n_trainable:,} "
        f"({100 * n_trainable / n_total:.1f}% del modelo)"
    )

    # ------------------------------------------------------------------
    # 4. Cargar datos de planta y construir tensores
    # ------------------------------------------------------------------
    logger.info("Cargando y procesando datos de planta...")

    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)

    logger.info(f"  CSV de entrenamiento cargado: {len(df_train)} filas.")
    logger.info(f"  CSV de validación cargado:    {len(df_val)} filas.")

    X_train, y_train = _build_tensor_from_df(df_train, feature_cols, scaler, config)
    X_val,   y_val   = _build_tensor_from_df(df_val,   feature_cols, scaler, config)

    if y_train is None or y_val is None:
        raise ValueError(
            "Los CSV de planta deben incluir las columnas de etiquetas "
            f"({config['features']['cols_targets']}) para poder calibrar el modelo."
        )

    from torch.utils.data import TensorDataset, DataLoader
    batch_size = min(ft_batch_size, len(X_train))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    logger.info(
        f"  Tensores construidos -> Train: {X_train.shape} | Val: {X_val.shape}"
    )

    # ------------------------------------------------------------------
    # 5. Configurar optimizador y criterio
    # ------------------------------------------------------------------
    # LR ×10 menor al de entrenamiento original → ajuste fino suave
    base_lr = config['training'].get('learning_rate', 0.001)
    ft_lr   = base_lr / lr_divisor
    logger.info(f"Learning rate de calibración: {ft_lr:.2e} (base LR / {lr_divisor})")

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=ft_lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 6. Bucle de Fine-Tuning con Early Stopping
    # ------------------------------------------------------------------
    best_val_loss   = float('inf')
    best_weights    = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history         = {'train_loss': [], 'val_loss': []}
    start_time      = time.time()

    logger.info(f"Iniciando calibración: máx. {ft_epochs} épocas | paciencia: {ft_patience}")

    for epoch in range(ft_epochs):
        # --- Entrenamiento ---
        model.train()
        run_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            p_foul, p_valv, p_bomb, p_acum = model(inputs)
            loss = (
                criterion(p_foul, labels[:, 0]) +
                criterion(p_valv, labels[:, 1]) +
                criterion(p_bomb, labels[:, 2]) +
                criterion(p_acum, labels[:, 3])
            )
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        avg_train_loss = run_loss / len(train_loader)

        # --- Validación ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                p_foul, p_valv, p_bomb, p_acum = model(inputs)
                val_loss += (
                    criterion(p_foul, labels[:, 0]) +
                    criterion(p_valv, labels[:, 1]) +
                    criterion(p_bomb, labels[:, 2]) +
                    criterion(p_acum, labels[:, 3])
                ).item()

        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Época {epoch + 1:03d}/{ft_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_weights     = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= ft_patience:
                logger.info(f"Early Stopping activado en época {epoch + 1}.")
                break

    elapsed = time.time() - start_time
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    logger.info(f"Calibración completada en {mins}m {secs}s.")

    # Restaurar mejores pesos
    model.load_state_dict(best_weights)

    # ------------------------------------------------------------------
    # 7. Guardar modelo calibrado (NO sobreescribe el base)
    # ------------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    finetuned_path = os.path.join(models_dir, output_model_name)
    torch.save(model.state_dict(), finetuned_path)
    logger.info(f"Modelo calibrado guardado en: {finetuned_path}")

    # ------------------------------------------------------------------
    # 8. Reporte de calibración
    # ------------------------------------------------------------------
    n_epochs_run = len(history['train_loss'])
    report_lines = [
        "=" * 65,
        "   REPORTE DE CALIBRACIÓN / FINE-TUNING",
        "=" * 65,
        f"  Fluido real:",
        f"    - Densidad:        {fluid_density_kg_l} kg/L",
        f"    - Calor específico: {fluid_cp_kj_kgK} kJ/(kg·K)",
        "",
        f"  Datos de entrenamiento: {train_csv}",
        f"  Datos de validación:    {val_csv}",
        f"  Ciclos de calibración:  {len(X_train)} (train) | {len(X_val)} (val)",
        "",
        f"  Configuración del fine-tuning:",
        f"    - Learning rate:   {ft_lr:.2e}",
        f"    - Épocas máximas:  {ft_epochs}",
        f"    - Paciencia ES:    {ft_patience}",
        f"    - Épocas ejecutadas: {n_epochs_run}",
        f"    - Tiempo total:    {mins}m {secs}s",
        "",
        f"  Parámetros entrenados: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.1f}%)",
        f"  (Backbone CNN congelado — cabezas de clasificación calibradas)",
        "",
        f"  Pérdida final en validación: {best_val_loss:.4f}",
        "",
        f"  Artefacto generado:",
        f"    {finetuned_path}",
        "=" * 65,
    ]
    report_str = "\n".join(report_lines)
    logger.info(report_str)

    report_path = os.path.join(models_dir, "finetuning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str + "\n")
    logger.info(f"Reporte guardado en: {report_path}")

    return model
