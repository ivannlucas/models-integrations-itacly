"""Retraining for ml3 — faithful port of inbox/a03/src/training/train.py.

Trains the FULL Deep Ensemble (LSTM + CNN + BiGRU, two outputs each: softmax class +
sigmoid severity) from scratch on a user-provided labeled CSV, replicating the delivered
procedure: apply_feature_engineering over each full series, 70/15/15 split by ID_Serie
(seed 42), StandardScaler fitted on train, 168-row windows with stride 1, early stopping
patience 2, Adam lr 0.001, 50 epochs. Hold-out metrics come from the ensemble's soft
voting on the test split — comparable with the memoria's Tabla 11/12.

Ports the delivered code 1:1 (same operations, in the same order; the long functions are split
into helpers only); TensorFlow stays imported lazily to keep the module light for unit tests.

One deliberate difference: the original stacks every window into one array (crear_secuencias)
before fitting. With stride 1 that is 168x the scaled rows (~11 GiB for the delivered 1.6M-row
dataset, plus Keras' own copy), which kills the pod. Here each split keeps its scaled rows once
and the windows are cut batch by batch (_Windows + _window_batches): the same windows, targets,
batch size, epochs and per-epoch shuffle, with memory that no longer grows 168x with the CSV.
"""
from __future__ import annotations

import functools
import gc
import logging
import math
from dataclasses import dataclass
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

from app.domain.services.exceptions import InsufficientDataError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml3_wine_disease_pest_forecast.constants import (
    ARCHITECTURES,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATES,
    MIN_SERIES_PER_STRATUM,
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


@dataclass(frozen=True)
class _Windows:
    """The stride-1 windows of one split, not materialized.

    Window ``i`` is ``features[starts[i]:starts[i] + window]`` (what train.crear_secuencias
    stacked) and its targets are ``y_class[i]`` / ``y_reg[i]``.
    """

    features: np.ndarray  # (rows, n_features) float32: the split's series, concatenated
    starts: np.ndarray  # (n_windows,) int64
    y_class: np.ndarray  # (n_windows,) int8
    y_reg: np.ndarray  # (n_windows,) float32
    window: int

    def __len__(self) -> int:
        return len(self.starts)

    def take(self, idx: np.ndarray) -> np.ndarray:
        """Windows ``idx`` stacked: ``(len(idx), window, n_features)`` float32."""
        return self.features[self.starts[idx][:, None] + np.arange(self.window)]


def _crear_ventanas(
    df: pd.DataFrame, window_size: int, features: list[str], columns: tuple[str, str, str],
) -> _Windows:
    """train.crear_secuencias without stacking: same windows (stride 1) and targets.

    ``columns`` is (t_class, t_reg, series_col) of the original signature. A series of n rows
    gives the windows starting at rows 0..n-window_size-1, each labeled with the row after it.
    """
    t_class, t_reg, series_col = columns
    feats, starts, y_c, y_r = [], [], [], []
    offset = 0
    for _, grupo in df.groupby(series_col, sort=False):
        n_rows = len(grupo)
        feats.append(grupo[features].to_numpy(dtype=np.float32))
        if n_rows > window_size:
            starts.append(offset + np.arange(n_rows - window_size, dtype=np.int64))
            y_c.append(grupo[t_class].to_numpy()[window_size:])
            y_r.append(grupo[t_reg].to_numpy()[window_size:])
        offset += n_rows
    return _Windows(
        features=(np.concatenate(feats) if feats
                  else np.zeros((0, len(features)), dtype=np.float32)),
        starts=np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64),
        y_class=(np.concatenate(y_c) if y_c else np.zeros(0)).astype(np.int8),
        y_reg=(np.concatenate(y_r) if y_r else np.zeros(0)).astype(np.float32),
        window=window_size,
    )


@functools.cache
def _window_batches_class():
    """keras.utils.PyDataset over a _Windows (built lazily: TensorFlow is imported lazily)."""
    # pylint: disable-next=import-outside-toplevel,import-error
    import keras

    class WindowBatches(keras.utils.PyDataset):
        """Batches of windows cut on demand; ``shuffle_seed`` reshuffles them every epoch.

        Shuffling is what ``model.fit(x, y, shuffle=True)`` did over the stacked windows.
        Without ``targets`` it yields only the windows (for ``predict``).
        """

        def __init__(self, windows: _Windows, batch_size: int, *,
                     shuffle_seed: int | None = None, targets: bool = True) -> None:
            super().__init__()
            self._windows = windows
            self._batch_size = batch_size
            self._targets = targets
            self._rng = None if shuffle_seed is None else np.random.default_rng(shuffle_seed)
            self._order = np.arange(len(windows))
            self.on_epoch_end()

        def __len__(self) -> int:
            return math.ceil(len(self._windows) / self._batch_size)

        def __getitem__(self, index: int):
            idx = self._order[index * self._batch_size:(index + 1) * self._batch_size]
            x = self._windows.take(idx)
            if not self._targets:
                return x
            return x, {"out_class": self._windows.y_class[idx],
                       "out_reg": self._windows.y_reg[idx]}

        def on_epoch_end(self) -> None:
            if self._rng is not None:
                self._rng.shuffle(self._order)

    return WindowBatches


def _window_batches(windows: _Windows, *, shuffle_seed: int | None = None,
                    targets: bool = True):
    """The PyDataset of ``windows`` in batches of BATCH_SIZE."""
    return _window_batches_class()(windows, BATCH_SIZE, shuffle_seed=shuffle_seed,
                                   targets=targets)


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


def _validate_series_split(df: pd.DataFrame) -> None:
    """Fail with a clear message when the series cannot be split 70/15/15 stratified.

    Without this, too few series surface as raw sklearn errors ("With n_samples=1,
    test_size=0.5 ... the resulting train set will be empty", "The least populated class
    in y has only 1 member"). The split itself is the original's; this only checks that
    it is feasible before running it.
    """
    labels_per_series = df.groupby(SERIES_COLUMN)[STRATIFY_COLUMN].nunique()
    mixed = labels_per_series[labels_per_series > 1].index.tolist()
    if mixed:
        raise InsufficientDataError(
            f"{STRATIFY_COLUMN} debe ser constante dentro de cada {SERIES_COLUMN} (es la "
            f"patología de toda la serie); estas series tienen varios valores: {mixed[:10]}."
        )
    classes = sorted(df[TARGET_CLASS_COLUMN].astype(str).unique())
    if len(classes) < 2:
        raise InsufficientDataError(
            f"{TARGET_CLASS_COLUMN} necesita al menos 2 clases distintas para entrenar el "
            f"clasificador; el fichero solo tiene {classes}."
        )
    per_stratum = df.drop_duplicates(subset=[SERIES_COLUMN])[STRATIFY_COLUMN].value_counts()
    short = per_stratum[per_stratum < MIN_SERIES_PER_STRATUM]
    if not short.empty:
        detail = ", ".join(f"{label}: {count}" for label, count in short.sort_index().items())
        raise InsufficientDataError(
            f"Datos insuficientes para el reparto de series 70/15/15 (entrenamiento/validación/"
            f"test) estratificado por {STRATIFY_COLUMN}: se necesitan al menos "
            f"{MIN_SERIES_PER_STRATUM} series ({SERIES_COLUMN} distintos) por cada valor de "
            f"{STRATIFY_COLUMN}. Series por valor con menos de las necesarias: {detail}."
        )


def _require_windows(splits: dict[str, _Windows]) -> None:
    """Each split needs at least one WINDOW_SIZE-hour window, or training cannot run."""
    empty = [name for name, windows in splits.items() if len(windows) == 0]
    if empty:
        raise InsufficientDataError(
            f"Ninguna serie del conjunto de {', '.join(empty)} tiene más de {WINDOW_SIZE} filas "
            f"horarias, así que no se puede formar ninguna ventana de {WINDOW_SIZE} h. Cada serie "
            f"necesita al menos {WINDOW_SIZE + 1} filas consecutivas tras limpiar los datos."
        )


def _encode_and_split(df: pd.DataFrame) -> tuple[LabelEncoder, tuple, tuple]:
    """Encode the class label and split series 70/15/15 (stratified, seed 42).

    Returns (le, (train_df, val_df, test_df), (n_train_ids, n_val_ids, n_test_ids)).
    """
    _validate_series_split(df)
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


def _scale_and_window(frames: tuple) -> tuple[StandardScaler, _Windows, _Windows, _Windows]:
    """Fit the StandardScaler on train, scale the 3 splits and index their windows.

    Returns (scaler, windows of train, of val, of test).
    """
    train_df, val_df, test_df = frames
    scaler = StandardScaler()
    scaler.fit(train_df[MODEL_FEATURES])
    train_df[MODEL_FEATURES] = scaler.transform(train_df[MODEL_FEATURES])
    val_df[MODEL_FEATURES] = scaler.transform(val_df[MODEL_FEATURES])
    test_df[MODEL_FEATURES] = scaler.transform(test_df[MODEL_FEATURES])

    logger.info("Indexando ventanas de %d h...", WINDOW_SIZE)
    train = _crear_ventanas(train_df, WINDOW_SIZE, MODEL_FEATURES, _SEQ_COLUMNS)
    val = _crear_ventanas(val_df, WINDOW_SIZE, MODEL_FEATURES, _SEQ_COLUMNS)
    test = _crear_ventanas(test_df, WINDOW_SIZE, MODEL_FEATURES, _SEQ_COLUMNS)
    return scaler, train, val, test


def _fit_models(
    modelos: list, train: _Windows, val: _Windows, model_paths: dict[str, str]
) -> list[int]:
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
            _window_batches(train, shuffle_seed=TRAIN_RANDOM_SEED),
            validation_data=_window_batches(val),
            epochs=EPOCHS, callbacks=[c_stop, c_check], verbose=0,
        )
        epochs_executed.append(len(hist.history["loss"]))
    return epochs_executed


def _ensemble_test_metrics(modelos: list, test: _Windows) -> dict[str, float]:
    """Hold-out metrics: ensemble soft voting on test (same as train.py save_metrics DEL block)."""
    y_test_c, y_test_r = test.y_class, test.y_reg
    preds_clases_prob, preds_regresion = [], []
    for modelo in modelos:
        # [class probabilities, severity]
        preds = modelo.predict(_window_batches(test, targets=False), verbose=0)
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
    _require_windows({"entrenamiento": train, "validación": val, "test": test})
    logger.info("Ventanas: train %d, val %d, test %d", len(train), len(val), len(test))
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
        "n_windows_train": len(train),
        "n_windows_val": len(val),
        "n_windows_test": len(test),
        "epochs_executed": epochs_executed,
    }


def load_retraining_input(data_path: str) -> pd.DataFrame:
    """Load the labeled CSV/parquet and validate the training contract columns."""
    with local_file_path(data_path) as local_path:
        # Parquet by content (magic "PAR1"), not by suffix: an s3:// key may have no extension.
        with open(local_path, "rb") as fh:
            is_parquet = fh.read(4) == b"PAR1"
        raw = pd.read_parquet(local_path) if is_parquet else pd.read_csv(local_path)
    missing = [c for c in TRAIN_HARD_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV de entrenamiento — faltan columnas requeridas del contrato: {missing}. "
            f"{STRATIFY_COLUMN} es la patología de cada serie (constante por {SERIES_COLUMN}) "
            f"y estratifica el reparto de series en entrenamiento/validación/test."
        )
    return raw
