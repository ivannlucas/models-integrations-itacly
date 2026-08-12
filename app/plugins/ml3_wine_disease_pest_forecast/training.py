"""Retraining for ml3 — faithful port of inbox/a03/src/training/train.py.

Trains the FULL Deep Ensemble (LSTM + CNN + BiGRU, two outputs each: softmax class +
sigmoid severity) from scratch on a user-provided labeled CSV, replicating the delivered
procedure: apply_feature_engineering over each full series, 70/15/15 split by ID_Serie
(seed 42), StandardScaler fitted on train, 168-row windows with stride 1, early stopping
patience 2, Adam lr 0.001, 50 epochs. Hold-out metrics come from the ensemble's soft
voting on the test split — comparable with the memoria's Tabla 11/12.

Ports the delivered code 1:1 (function arity, local variable counts, long logging lines);
TensorFlow stays imported lazily to keep the module light for unit tests.
"""
from __future__ import annotations

import gc
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-statements,import-outside-toplevel,no-name-in-module,no-member,line-too-long

from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml3_wine_disease_pest_forecast.constants import (
    ARCHITECTURES,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATES,
    MODEL_FEATURES,
    SERIES_COLUMN,
    STRATIFY_COLUMN,
    TARGET_CLASS_COLUMN,
    TARGET_CLASS_LABEL,
    TARGET_REG_COLUMN,
    TRAIN_HARD_REQUIRED_COLUMNS,
    TRAIN_RANDOM_SEED,
    TRAIN_TEST_SIZE,
    TRAIN_VAL_TEST_RATIO,
    WINDOW_SIZE,
)

logger = logging.getLogger(__name__)


def _crear_secuencias(
    df: pd.DataFrame, window_size: int, features: list[str],
    t_class: str, t_reg: str, series_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of train.crear_secuencias: tabular frame -> 3D window tensors (stride 1)."""
    x, y_c, y_r = [], [], []
    for _, grupo in df.groupby(series_col, sort=False):
        data_features = grupo[features].values
        data_class = grupo[t_class].values
        data_reg = grupo[t_reg].values
        for i in range(len(data_features) - window_size):
            x.append(data_features[i:i + window_size])
            y_c.append(data_class[i + window_size])
            y_r.append(data_reg[i + window_size])
    return (
        np.array(x, dtype=np.float32),
        np.array(y_c, dtype=np.int8),
        np.array(y_r, dtype=np.float32),
    )


def _build_models(
    input_shape: tuple, num_classes: int, learning_rates: dict, arch_params: dict,
) -> list:
    """Port of train.build_models — same layers, names, losses and metrics."""
    from tensorflow.keras.layers import (
        GRU,
        LSTM,
        BatchNormalization,
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    def compilar(modelo, lr):
        modelo.compile(
            optimizer=Adam(learning_rate=lr),
            loss={"out_class": "sparse_categorical_crossentropy", "out_reg": "mse"},
            loss_weights={"out_class": 1.0, "out_reg": 1.0},
            metrics={"out_class": ["accuracy"], "out_reg": ["mae"]},
        )
        return modelo

    inp = Input(shape=input_shape, name="Input_M1")
    x = LSTM(arch_params["lstm"]["lstm_1"], return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(arch_params["lstm"]["dropout"])(x)
    x = LSTM(arch_params["lstm"]["lstm_2"])(x)
    x = BatchNormalization()(x)
    out_class1 = Dense(num_classes, activation="softmax", name="out_class")(x)
    out_reg1 = Dense(1, activation="sigmoid", name="out_reg")(x)
    m1 = compilar(
        Model(inputs=inp, outputs=[out_class1, out_reg1], name="M1_LSTM"),
        learning_rates.get("lstm", 0.001),
    )

    inp = Input(shape=input_shape, name="Input_M2")
    x = Conv1D(filters=arch_params["cnn"]["filters_1"], kernel_size=arch_params["cnn"]["kernel_1"],
               activation="relu", padding="same")(inp)
    x = BatchNormalization()(x)
    x = Conv1D(filters=arch_params["cnn"]["filters_2"], kernel_size=arch_params["cnn"]["kernel_2"],
               activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(arch_params["cnn"]["dropout"])(x)
    out_class2 = Dense(num_classes, activation="softmax", name="out_class")(x)
    out_reg2 = Dense(1, activation="sigmoid", name="out_reg")(x)
    m2 = compilar(
        Model(inputs=inp, outputs=[out_class2, out_reg2], name="M2_CNN"),
        learning_rates.get("cnn", 0.001),
    )

    inp = Input(shape=input_shape, name="Input_M3")
    x = Bidirectional(GRU(arch_params["bigru"]["gru_units"], return_sequences=False))(inp)
    x = BatchNormalization()(x)
    x = Dropout(arch_params["bigru"]["dropout"])(x)
    x = Dense(arch_params["bigru"]["dense_units"], activation="relu")(x)
    out_class3 = Dense(num_classes, activation="softmax", name="out_class")(x)
    out_reg3 = Dense(1, activation="sigmoid", name="out_reg")(x)
    m3 = compilar(
        Model(inputs=inp, outputs=[out_class3, out_reg3], name="M3_BiGRU"),
        learning_rates.get("bigru", 0.001),
    )

    return [m1, m2, m3]


def run_retraining(raw_df: pd.DataFrame, model_paths: dict[str, str]) -> dict[str, Any]:
    """Retrain the full ensemble from a labeled raw frame; save artifacts under *model_paths*.

    Returns {metrics, le, scaler, models, n_windows_train, n_windows_val, n_windows_test,
    epochs_executed}.
    """
    from app.plugins.ml3_wine_disease_pest_forecast.feature_engineering import apply_feature_engineering

    df = apply_feature_engineering(raw_df).copy()
    df = df.reindex(sorted(df.columns), axis=1)
    if df.empty:
        raise ValueError("No quedaron filas tras aplicar la ingeniería de variables.")

    le = LabelEncoder()
    df[TARGET_CLASS_LABEL] = le.fit_transform(df[TARGET_CLASS_COLUMN].astype(str))
    logger.info("Clases a entrenar: %s", le.classes_)

    df_series = df.drop_duplicates(subset=[SERIES_COLUMN])[[SERIES_COLUMN, STRATIFY_COLUMN]]
    test_size = TRAIN_TEST_SIZE
    train_ids, temp_ids, _, temp_etiquetas = train_test_split(
        df_series[SERIES_COLUMN], df_series[STRATIFY_COLUMN],
        test_size=test_size, random_state=TRAIN_RANDOM_SEED, stratify=df_series[STRATIFY_COLUMN],
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=TRAIN_VAL_TEST_RATIO, random_state=TRAIN_RANDOM_SEED,
        stratify=temp_etiquetas,
    )

    train_df = df[df[SERIES_COLUMN].isin(train_ids)].copy()
    val_df = df[df[SERIES_COLUMN].isin(val_ids)].copy()
    test_df = df[df[SERIES_COLUMN].isin(test_ids)].copy()
    logger.info("Series repartidas: Train %d, Val %d, Test %d", len(train_ids), len(val_ids), len(test_ids))

    scaler = StandardScaler()
    scaler.fit(train_df[MODEL_FEATURES])
    train_df[MODEL_FEATURES] = scaler.transform(train_df[MODEL_FEATURES])
    val_df[MODEL_FEATURES] = scaler.transform(val_df[MODEL_FEATURES])
    test_df[MODEL_FEATURES] = scaler.transform(test_df[MODEL_FEATURES])

    logger.info("Empaquetando secuencias en tensores 3D...")
    x_train, y_train_c, y_train_r = _crear_secuencias(
        train_df, WINDOW_SIZE, MODEL_FEATURES, TARGET_CLASS_LABEL, TARGET_REG_COLUMN, SERIES_COLUMN
    )
    x_val, y_val_c, y_val_r = _crear_secuencias(
        val_df, WINDOW_SIZE, MODEL_FEATURES, TARGET_CLASS_LABEL, TARGET_REG_COLUMN, SERIES_COLUMN
    )
    x_test, y_test_c, y_test_r = _crear_secuencias(
        test_df, WINDOW_SIZE, MODEL_FEATURES, TARGET_CLASS_LABEL, TARGET_REG_COLUMN, SERIES_COLUMN
    )
    logger.info(
        "Tensores: train %s, val %s, test %s", x_train.shape, x_val.shape, x_test.shape,
    )
    del train_df, val_df, test_df, df
    gc.collect()

    # TensorFlow is imported lazily so the plugin modules stay light for unit tests.
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    input_shape = (WINDOW_SIZE, len(MODEL_FEATURES))
    num_classes = len(le.classes_)
    tf.keras.utils.set_random_seed(TRAIN_RANDOM_SEED)

    modelos = _build_models(input_shape, num_classes, LEARNING_RATES, ARCHITECTURES)
    epochs_executed = []
    for modelo in modelos:
        logger.info("Entrenando: %s", modelo.name)
        c_stop = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                               restore_best_weights=True)
        c_check = ModelCheckpoint(filepath=model_paths[modelo.name], monitor="val_loss",
                                  save_best_only=True, mode="min")
        hist = modelo.fit(
            x_train, {"out_class": y_train_c, "out_reg": y_train_r},
            validation_data=(x_val, {"out_class": y_val_c, "out_reg": y_val_r}),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[c_stop, c_check], verbose=0,
        )
        epochs_executed.append(len(hist.history["loss"]))

    joblib.dump(scaler, model_paths["scaler"])
    joblib.dump(le, model_paths["label_encoder"])

    # Hold-out metrics: ensemble soft voting on test (same as train.py save_metrics DEL block).
    preds_clases_prob, preds_regresion = [], []
    for modelo in modelos:
        p_class, p_reg = modelo.predict(x_test, verbose=0)
        preds_clases_prob.append(p_class)
        preds_regresion.append(p_reg)
    media_probs = np.mean(preds_clases_prob, axis=0)
    y_pred_c = np.argmax(media_probs, axis=1)
    y_pred_r = np.mean(preds_regresion, axis=0).flatten()

    metrics = {
        "accuracy": float(accuracy_score(y_test_c, y_pred_c)),
        "precision_macro": float(precision_score(y_test_c, y_pred_c, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test_c, y_pred_c, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test_c, y_pred_c, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test_c, y_pred_c, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_test_r, y_pred_r)),
        "mse": float(mean_squared_error(y_test_r, y_pred_r)),
        "r2": float(r2_score(y_test_r, y_pred_r)),
    }

    logger.info("Ensemble retrained — test acc=%.4f f1_macro=%.4f mae=%.4f r2=%.4f",
                metrics["accuracy"], metrics["f1_macro"], metrics["mae"], metrics["r2"])
    return {
        "metrics": metrics,
        "le": le,
        "scaler": scaler,
        "models": modelos,
        "n_windows_train": int(len(x_train)),
        "n_windows_val": int(len(x_val)),
        "n_windows_test": int(len(x_test)),
        "epochs_executed": epochs_executed,
    }


def load_retraining_input(data_path: str) -> pd.DataFrame:
    """Load the labeled CSV/parquet and validate the training contract columns."""
    with local_file_path(data_path) as local_path:
        raw = pd.read_csv(local_path) if str(local_path).endswith(".csv") else pd.read_parquet(local_path)
    missing = [c for c in TRAIN_HARD_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV de entrenamiento — faltan columnas requeridas del contrato: {missing}. "
            f"Además, se recomienda incluir {STRATIFY_COLUMN} para el split estratificado."
        )
    return raw
