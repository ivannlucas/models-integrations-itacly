"""Retraining for ml3 — faithful port of inbox/a03/src/training/train.py.

Trains the FULL Deep Ensemble (LSTM + CNN + BiGRU, two outputs each: softmax class +
sigmoid severity) from scratch on a user-provided labeled CSV, replicating the delivered
procedure: apply_feature_engineering over each full series, 70/15/15 split by ID_Serie
(seed 42), StandardScaler fitted on train, 168-row windows with stride 1, early stopping
patience 2, Adam lr 0.001, 50 epochs. Hold-out metrics come from the ensemble's soft
voting on the test split — comparable with the memoria's Tabla 11/12.

Ports the delivered code 1:1 (same operations, in the same order; the long functions are split
into helpers only); TensorFlow stays imported lazily to keep the module light for unit tests.
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
from app.plugins.ml3_wine_disease_pest_forecast.feature_engineering import (
    apply_feature_engineering,
)

logger = logging.getLogger(__name__)


def _crear_secuencias(
    df: pd.DataFrame, window_size: int, features: list[str], columns: tuple[str, str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of train.crear_secuencias: tabular frame -> 3D window tensors (stride 1).

    ``columns`` is (t_class, t_reg, series_col) of the original signature.
    """
    t_class, t_reg, series_col = columns
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


def _compilar(modelo, lr):
    """Port of train.build_models.compilar — Adam + class/severity losses and metrics."""
    # pylint: disable-next=import-outside-toplevel,import-error,no-name-in-module
    from tensorflow.keras.optimizers import Adam

    modelo.compile(
        optimizer=Adam(learning_rate=lr),
        loss={"out_class": "sparse_categorical_crossentropy", "out_reg": "mse"},
        loss_weights={"out_class": 1.0, "out_reg": 1.0},
        metrics={"out_class": ["accuracy"], "out_reg": ["mae"]},
    )
    return modelo


def _two_heads(layers, x, num_classes: int) -> list:
    """Softmax class head + sigmoid severity head shared by the 3 networks."""
    out_class = layers.Dense(num_classes, activation="softmax", name="out_class")(x)
    out_reg = layers.Dense(1, activation="sigmoid", name="out_reg")(x)
    return [out_class, out_reg]


def _build_lstm(layers, model_cls, input_shape: tuple, num_classes: int, arch: dict):
    """M1_LSTM of train.build_models (uncompiled)."""
    inp = layers.Input(shape=input_shape, name="Input_M1")
    x = layers.LSTM(arch["lstm_1"], return_sequences=True)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(arch["dropout"])(x)
    x = layers.LSTM(arch["lstm_2"])(x)
    x = layers.BatchNormalization()(x)
    return model_cls(inputs=inp, outputs=_two_heads(layers, x, num_classes), name="M1_LSTM")


def _build_cnn(layers, model_cls, input_shape: tuple, num_classes: int, arch: dict):
    """M2_CNN of train.build_models (uncompiled)."""
    inp = layers.Input(shape=input_shape, name="Input_M2")
    x = layers.Conv1D(filters=arch["filters_1"], kernel_size=arch["kernel_1"],
                      activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=arch["filters_2"], kernel_size=arch["kernel_2"],
                      activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(arch["dropout"])(x)
    return model_cls(inputs=inp, outputs=_two_heads(layers, x, num_classes), name="M2_CNN")


def _build_bigru(layers, model_cls, input_shape: tuple, num_classes: int, arch: dict):
    """M3_BiGRU of train.build_models (uncompiled)."""
    inp = layers.Input(shape=input_shape, name="Input_M3")
    x = layers.Bidirectional(layers.GRU(arch["gru_units"], return_sequences=False))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(arch["dropout"])(x)
    x = layers.Dense(arch["dense_units"], activation="relu")(x)
    return model_cls(inputs=inp, outputs=_two_heads(layers, x, num_classes), name="M3_BiGRU")


def _build_models(
    input_shape: tuple, num_classes: int, learning_rates: dict, arch_params: dict,
) -> list:
    """Port of train.build_models — same layers, names, losses and metrics, built in order."""
    # pylint: disable-next=import-outside-toplevel,import-error,no-name-in-module
    from tensorflow.keras import layers
    # pylint: disable-next=import-outside-toplevel,import-error,no-name-in-module
    from tensorflow.keras.models import Model

    m1 = _compilar(
        _build_lstm(layers, Model, input_shape, num_classes, arch_params["lstm"]),
        learning_rates.get("lstm", 0.001),
    )
    m2 = _compilar(
        _build_cnn(layers, Model, input_shape, num_classes, arch_params["cnn"]),
        learning_rates.get("cnn", 0.001),
    )
    m3 = _compilar(
        _build_bigru(layers, Model, input_shape, num_classes, arch_params["bigru"]),
        learning_rates.get("bigru", 0.001),
    )
    return [m1, m2, m3]


_SEQ_COLUMNS = (TARGET_CLASS_LABEL, TARGET_REG_COLUMN, SERIES_COLUMN)


def _encode_and_split(df: pd.DataFrame) -> tuple[LabelEncoder, tuple, tuple]:
    """Encode the class label and split series 70/15/15 (stratified, seed 42).

    Returns (le, (train_df, val_df, test_df), (n_train_ids, n_val_ids, n_test_ids)).
    """
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

    frames = (
        df[df[SERIES_COLUMN].isin(train_ids)].copy(),
        df[df[SERIES_COLUMN].isin(val_ids)].copy(),
        df[df[SERIES_COLUMN].isin(test_ids)].copy(),
    )
    return le, frames, (len(train_ids), len(val_ids), len(test_ids))


def _scale_and_window(frames: tuple) -> tuple[StandardScaler, tuple, tuple, tuple]:
    """Fit the StandardScaler on train, scale the 3 splits and build their window tensors.

    Returns (scaler, (x, y_c, y_r) of train, of val, of test).
    """
    train_df, val_df, test_df = frames
    scaler = StandardScaler()
    scaler.fit(train_df[MODEL_FEATURES])
    train_df[MODEL_FEATURES] = scaler.transform(train_df[MODEL_FEATURES])
    val_df[MODEL_FEATURES] = scaler.transform(val_df[MODEL_FEATURES])
    test_df[MODEL_FEATURES] = scaler.transform(test_df[MODEL_FEATURES])

    logger.info("Empaquetando secuencias en tensores 3D...")
    train = _crear_secuencias(train_df, WINDOW_SIZE, MODEL_FEATURES, _SEQ_COLUMNS)
    val = _crear_secuencias(val_df, WINDOW_SIZE, MODEL_FEATURES, _SEQ_COLUMNS)
    test = _crear_secuencias(test_df, WINDOW_SIZE, MODEL_FEATURES, _SEQ_COLUMNS)
    return scaler, train, val, test


def _fit_models(modelos: list, train: tuple, val: tuple, model_paths: dict[str, str]) -> list[int]:
    """Fit each network with early stopping + best checkpoint; return the epochs each ran."""
    # pylint: disable-next=import-outside-toplevel,import-error,no-name-in-module
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    epochs_executed = []
    for modelo in modelos:
        logger.info("Entrenando: %s", modelo.name)
        c_stop = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                               restore_best_weights=True)
        c_check = ModelCheckpoint(filepath=model_paths[modelo.name], monitor="val_loss",
                                  save_best_only=True, mode="min")
        hist = modelo.fit(
            train[0], {"out_class": train[1], "out_reg": train[2]},
            validation_data=(val[0], {"out_class": val[1], "out_reg": val[2]}),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[c_stop, c_check], verbose=0,
        )
        epochs_executed.append(len(hist.history["loss"]))
    return epochs_executed


def _ensemble_test_metrics(modelos: list, test: tuple) -> dict[str, float]:
    """Hold-out metrics: ensemble soft voting on test (same as train.py save_metrics DEL block)."""
    x_test, y_test_c, y_test_r = test
    preds_clases_prob, preds_regresion = [], []
    for modelo in modelos:
        preds = modelo.predict(x_test, verbose=0)  # [class probabilities, severity]
        preds_clases_prob.append(preds[0])
        preds_regresion.append(preds[1])
    media_probs = np.mean(preds_clases_prob, axis=0)
    y_pred_c = np.argmax(media_probs, axis=1)
    y_pred_r = np.mean(preds_regresion, axis=0).flatten()

    return {
        "accuracy": float(accuracy_score(y_test_c, y_pred_c)),
        "precision_macro": float(
            precision_score(y_test_c, y_pred_c, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test_c, y_pred_c, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test_c, y_pred_c, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test_c, y_pred_c, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_test_r, y_pred_r)),
        "mse": float(mean_squared_error(y_test_r, y_pred_r)),
        "r2": float(r2_score(y_test_r, y_pred_r)),
    }


def run_retraining(raw_df: pd.DataFrame, model_paths: dict[str, str]) -> dict[str, Any]:
    """Retrain the full ensemble from a labeled raw frame; save artifacts under *model_paths*.

    Returns {metrics, le, scaler, models, n_windows_train, n_windows_val, n_windows_test,
    epochs_executed}.
    """
    df = apply_feature_engineering(raw_df).copy()
    df = df.reindex(sorted(df.columns), axis=1)
    if df.empty:
        raise ValueError("No quedaron filas tras aplicar la ingeniería de variables.")

    le, frames, n_ids = _encode_and_split(df)
    logger.info("Series repartidas: Train %d, Val %d, Test %d", *n_ids)

    scaler, train, val, test = _scale_and_window(frames)
    logger.info(
        "Tensores: train %s, val %s, test %s", train[0].shape, val[0].shape, test[0].shape,
    )
    del frames, df
    gc.collect()

    # TensorFlow is imported lazily so the plugin modules stay light for unit tests.
    import tensorflow as tf  # noqa: F401  # pylint: disable=import-outside-toplevel

    tf.keras.utils.set_random_seed(TRAIN_RANDOM_SEED)  # pylint: disable=no-member

    modelos = _build_models(
        (WINDOW_SIZE, len(MODEL_FEATURES)), len(le.classes_), LEARNING_RATES, ARCHITECTURES
    )
    epochs_executed = _fit_models(modelos, train, val, model_paths)

    joblib.dump(scaler, model_paths["scaler"])
    joblib.dump(le, model_paths["label_encoder"])

    metrics = _ensemble_test_metrics(modelos, test)

    logger.info("Ensemble retrained — test acc=%.4f f1_macro=%.4f mae=%.4f r2=%.4f",
                metrics["accuracy"], metrics["f1_macro"], metrics["mae"], metrics["r2"])
    return {
        "metrics": metrics,
        "le": le,
        "scaler": scaler,
        "models": modelos,
        "n_windows_train": int(len(train[0])),
        "n_windows_val": int(len(val[0])),
        "n_windows_test": int(len(test[0])),
        "epochs_executed": epochs_executed,
    }


def load_retraining_input(data_path: str) -> pd.DataFrame:
    """Load the labeled CSV/parquet and validate the training contract columns."""
    with local_file_path(data_path) as local_path:
        if str(local_path).endswith(".csv"):
            raw = pd.read_csv(local_path)
        else:
            raw = pd.read_parquet(local_path)
    missing = [c for c in TRAIN_HARD_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV de entrenamiento — faltan columnas requeridas del contrato: {missing}. "
            f"Además, se recomienda incluir {STRATIFY_COLUMN} para el split estratificado."
        )
    return raw
