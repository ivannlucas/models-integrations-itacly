"""Retraining logic for ml37 — faithful port of src/training/trainer.py::train_model.

The original trains the LSTM multitask model from scratch (random init, seed 42) every time —
there is no transfer-learning/fine-tuning step in the delivered procedure (unlike ml1's CNN
ensemble), so this port does the same: a fresh model, never built from the served checkpoint.
Only the configuration of reference (no_virtual, schema_v1, random_stratified_cooler) is
supported — the other 3 run_tag/split_policy combinations are out of scope (see manifest
known_issues).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import pandas as pd  # noqa: E402  pylint: disable=wrong-import-position
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score  # noqa: E402  pylint: disable=wrong-import-position
from sklearn.model_selection import train_test_split  # noqa: E402  pylint: disable=wrong-import-position
from sklearn.preprocessing import StandardScaler  # noqa: E402  pylint: disable=wrong-import-position

from app.domain.services.exceptions import DataContractError  # noqa: E402  pylint: disable=wrong-import-position
from app.plugins.ml37_wine_press_predictive_maintenance import preprocessing, xai  # noqa: E402  pylint: disable=wrong-import-position
from app.plugins.ml37_wine_press_predictive_maintenance.constants import (  # noqa: E402  pylint: disable=wrong-import-position
    CLASS_ORDERS,
    F_SCHEMA,
    PHYSICAL_SENSORS,
    TARGET_NAMES,
    TIMESTEPS,
    TRAIN_BATCH_SIZE,
    TRAIN_DENSE_UNITS,
    TRAIN_DROPOUT,
    TRAIN_EARLY_PATIENCE,
    TRAIN_EPOCHS_MAX,
    TRAIN_LEARNING_RATE,
    TRAIN_LSTM_UNITS,
    TRAIN_MIN_DELTA,
    TRAIN_MIN_LR,
    TRAIN_REDUCE_LR_FACTOR,
    TRAIN_REDUCE_LR_PATIENCE,
    TRAIN_SEED,
    TRAIN_STABLE_WEIGHT,
    TRAIN_TEST_SIZE,
    TRAIN_UNSTABLE_WEIGHT,
)

REQUIRED_COLUMNS = ("cycle_id", "t", *PHYSICAL_SENSORS,
                    "y_cooler", "y_valve", "y_pump", "y_accumulator")
TARGET_COLUMNS = ("y_cooler", "y_valve", "y_pump", "y_accumulator", "y_stable")


@dataclass
class LstmTrainResult:
    """Everything train() needs to build TrainResponse and upload to MLflow.

    ``metrics`` nests accuracy/balanced_accuracy/f1_macro (each a dict per head) and
    ``split_sizes`` nests n_train/n_val/n_test — kept together rather than as ten separate
    fields. ``shap_background`` is the unscaled SHAP background drawn from THIS run's train
    split (xai.build_shap_background), so explanations of the retrained model use its own data.
    """

    model: "keras.Model"
    scaler: dict
    class_mappings: dict
    metrics: dict
    split_sizes: dict
    epochs_run: int
    shap_background: np.ndarray


def _validate_training_frame(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(
            f"Faltan columnas obligatorias en el CSV de entrenamiento: {missing}"
        )
    for col in (*PHYSICAL_SENSORS, *[c for c in TARGET_COLUMNS if c in df.columns]):
        if not np.isfinite(pd.to_numeric(df[col], errors="coerce")).all():
            raise DataContractError(f"La columna '{col}' contiene valores no numéricos, NaN o Inf")


def _build_tensors_and_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Group the long-format table by cycle_id, resample each cycle to (600,14), pad to
    (600,40), and collect one target row (cooler, valve, pump, accumulator, stable|-1) per cycle
    — faithful to how the external tabular adapter builds X per cycle."""
    cycle_ids = sorted(df["cycle_id"].unique(), key=str)
    if len(cycle_ids) < 10:
        raise DataContractError(
            f"Se necesitan al menos 10 ciclos para separar un holdout de test del "
            f"{TRAIN_TEST_SIZE:.0%} (recibidos: {len(cycle_ids)})."
        )
    has_stable = "y_stable" in df.columns

    x = np.zeros((len(cycle_ids), TIMESTEPS, F_SCHEMA), dtype=np.float32)
    y_rows: list[list[int]] = []
    for i, cid in enumerate(cycle_ids):
        group = df[df["cycle_id"] == cid].sort_values("t")
        for target in ("y_cooler", "y_valve", "y_pump", "y_accumulator"):
            if group[target].nunique() != 1:
                raise DataContractError(
                    f"El target '{target}' no es constante dentro del ciclo '{cid}'"
                )
        t_values = group["t"].to_numpy(dtype=np.float64)
        sensor_series = {name: group[name].to_numpy() for name in PHYSICAL_SENSORS}
        resampled = preprocessing.resample_cycle_to_fixed_length(t_values, sensor_series)
        x[i] = preprocessing.build_schema_v1_tensor(resampled)
        row = [int(group["y_cooler"].iloc[0]), int(group["y_valve"].iloc[0]),
               int(group["y_pump"].iloc[0]), int(group["y_accumulator"].iloc[0])]
        row.append(int(group["y_stable"].iloc[0]) if has_stable else -1)
        y_rows.append(row)

    return x, np.asarray(y_rows, dtype=np.int64), cycle_ids


def _build_class_mappings() -> dict:
    """input_mode=schema_v1 always uses the fixed V1_CLASS_ORDERS (CLASS_ORDERS here), never a
    data-driven order — faithful to trainer.py::build_class_mappings."""
    value_to_index, index_to_value, class_order = {}, {}, {}
    for tname in TARGET_NAMES:
        order = list(CLASS_ORDERS[tname])
        class_order[tname] = order
        value_to_index[tname] = {str(v): i for i, v in enumerate(order)}
        index_to_value[tname] = {str(i): v for i, v in enumerate(order)}
    return {
        "targets": list(TARGET_NAMES),
        "class_order_by_target": class_order,
        "mappings_value_to_index": value_to_index,
        "mappings_index_to_value": index_to_value,
        "feature_order_final": list(PHYSICAL_SENSORS) + ["CE", "CP", "SE"]
        + [f"RESERVED_{i:02d}" for i in range(1, F_SCHEMA - len(PHYSICAL_SENSORS) - 3 + 1)],
    }


def _one_hot(y_idx: np.ndarray, class_mappings: dict) -> dict[str, np.ndarray]:
    out = {}
    for k, tname in enumerate(TARGET_NAMES):
        n_classes = len(class_mappings["class_order_by_target"][tname])
        one_hot = keras.utils.to_categorical(y_idx[:, k], num_classes=n_classes)
        out[tname] = one_hot.astype("float32")
    return out


def _build_multitask_lstm(class_mappings: dict) -> "keras.Model":
    """Faithful port of trainer.py::build_multitask_lstm — same shared trunk and 4 independent
    softmax heads, trained from scratch (no pretrained/fine-tuning step in the original)."""
    keras.utils.set_random_seed(TRAIN_SEED)
    inp = keras.layers.Input(shape=(TIMESTEPS, F_SCHEMA), name="X")
    x = keras.layers.Masking(mask_value=0.0)(inp)
    x = keras.layers.LSTM(TRAIN_LSTM_UNITS, dropout=TRAIN_DROPOUT, name="lstm")(x)
    x = keras.layers.Dense(TRAIN_DENSE_UNITS, activation="relu", name="dense")(x)

    outputs = []
    for tname in TARGET_NAMES:
        n_cls = len(class_mappings["class_order_by_target"][tname])
        outputs.append(keras.layers.Dense(n_cls, activation="softmax", name=tname)(x))
    return keras.Model(inp, outputs, name="lstm_multitask")


class _MultiMetricEarlyStopping(keras.callbacks.Callback):  # pylint: disable=too-many-instance-attributes
    """Faithful port of trainer.py::MultiMetricEarlyStopping: monitors a weighted sum of
    val_valve_loss + val_pump_loss + val_accumulator_loss (cooler excluded), patience=10,
    min_delta=1e-3, restore_best_weights=True.

    The 8 attributes are genuine early-stopping bookkeeping state (config + best-so-far +
    wait counter + saved weights), not a data-modelling smell to split up.
    """

    def __init__(self, monitor_keys, *, patience, min_delta, restore_best_weights=True):
        super().__init__()
        self.monitor_keys = monitor_keys
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best = float("inf")
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        kpi = sum(float(logs.get(k, 0.0)) for k in self.monitor_keys)
        logs["val_loss_kpi"] = kpi
        if kpi < self.best - self.min_delta:
            self.best = kpi
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)


def _split_cycle_indices(y_cooler: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """random_stratified_cooler: 70/15/15, stratified on the cooler target, seed=42 — faithful
    to trainer.py's only served split policy."""
    idx_train, idx_tmp = train_test_split(
        np.arange(len(y_cooler)), test_size=TRAIN_TEST_SIZE, random_state=TRAIN_SEED,
        shuffle=True, stratify=y_cooler,
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, random_state=TRAIN_SEED, shuffle=True,
        stratify=y_cooler[idx_tmp],
    )
    return idx_train, idx_val, idx_test


def _encode_targets(y: np.ndarray, class_mappings: dict) -> np.ndarray:
    """Real UCI codes (y[:, k]) -> class indices per head, in CLASS_ORDERS order."""
    y_idx = np.zeros((len(y), len(TARGET_NAMES)), dtype=np.int64)
    for k, tname in enumerate(TARGET_NAMES):
        v2i = class_mappings["mappings_value_to_index"][tname]
        y_idx[:, k] = [v2i[str(int(v))] for v in y[:, k]]
    return y_idx


# Cycles scaled per transform() call: StandardScaler.transform returns float64, so scaling the
# whole (N, 600, 40) tensor at once briefly held a float64 copy (≈420 MB for 2205 cycles).
_SCALE_CHUNK_CYCLES = 256


def _fit_scaler(x: np.ndarray, idx_train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit a StandardScaler on the train split only, then scale every cycle with it.

    The transform is row-wise, so scaling in chunks of cycles into a float32 array gives exactly
    the same values as ``scaler.transform(x.reshape(-1, F)).astype("float32")`` at once."""
    scaler = StandardScaler()
    scaler.fit(x[idx_train].reshape(-1, F_SCHEMA))
    x_scaled = np.empty(x.shape, dtype=np.float32)
    for start in range(0, len(x), _SCALE_CHUNK_CYCLES):
        block = x[start:start + _SCALE_CHUNK_CYCLES]
        x_scaled[start:start + len(block)] = (
            scaler.transform(block.reshape(-1, F_SCHEMA)).reshape(block.shape)
        )
    payload = {
        "type": "StandardScaler", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
    }
    return x_scaled, payload


def _build_sample_weights(y: np.ndarray) -> np.ndarray | None:
    """sample_weight = TRAIN_STABLE_WEIGHT if stable==1 else TRAIN_UNSTABLE_WEIGHT, the same
    vector for all 4 heads — None when the table has no y_stable column (trainer.py:647-657)."""
    stable_vec = y[:, 4]
    if (stable_vec < 0).any():
        return None
    return np.where(stable_vec == 1, TRAIN_STABLE_WEIGHT, TRAIN_UNSTABLE_WEIGHT).astype("float32")


def _compile_and_fit(  # pylint: disable=too-many-arguments
    model: "keras.Model", x_train_val: tuple[np.ndarray, np.ndarray], y_oh: dict[str, np.ndarray],
    sample_weights: np.ndarray | None, idx_train_val: tuple[np.ndarray, np.ndarray],
):
    """Compile with the original's optimizer/loss/metrics and fit with both callbacks.
    ``x_train_val`` are the already-sliced scaled tensors of the train and val cycles."""
    idx_train, idx_val = idx_train_val
    output_names = list(model.output_names)
    losses = [keras.losses.CategoricalCrossentropy(from_logits=False) for _ in output_names]
    metrics = [[keras.metrics.CategoricalAccuracy(name="acc")] for _ in output_names]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LEARNING_RATE),
        loss=losses, metrics=metrics,
    )

    def weights_for(idx: np.ndarray) -> np.ndarray:
        if sample_weights is None:
            return np.ones(len(idx), dtype=np.float32)
        return sample_weights[idx]

    monitor_keys = [f"val_{t}_loss" for t in output_names if t != "cooler"]
    early_stop = _MultiMetricEarlyStopping(
        monitor_keys, patience=TRAIN_EARLY_PATIENCE, min_delta=TRAIN_MIN_DELTA,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=TRAIN_REDUCE_LR_FACTOR, patience=TRAIN_REDUCE_LR_PATIENCE,
        min_delta=TRAIN_MIN_DELTA, min_lr=TRAIN_MIN_LR,
    )
    return model.fit(
        x_train_val[0], [y_oh[t][idx_train] for t in output_names],
        sample_weight=[weights_for(idx_train)] * len(output_names),
        validation_data=(
            x_train_val[1], [y_oh[t][idx_val] for t in output_names],
            [weights_for(idx_val)] * len(output_names),
        ),
        epochs=TRAIN_EPOCHS_MAX, batch_size=TRAIN_BATCH_SIZE,
        callbacks=[early_stop, reduce_lr], verbose=0,
    )


def _evaluate_test(
    model: "keras.Model", x_test: np.ndarray, y_idx: np.ndarray, idx_test: np.ndarray,
) -> dict:
    """Test-split accuracy/balanced_accuracy/f1_macro per head — same formulas as trainer.py.
    ``x_test`` is the already-sliced scaled tensor of the test cycles."""
    probs_test = model.predict(x_test, verbose=0)
    accuracy, balanced_accuracy, f1_macro = {}, {}, {}
    for k, tname in enumerate(TARGET_NAMES):
        y_true = y_idx[idx_test, k]
        y_pred = np.argmax(probs_test[k], axis=1)
        accuracy[tname] = round(float(accuracy_score(y_true, y_pred)), 4)
        balanced_accuracy[tname] = round(float(balanced_accuracy_score(y_true, y_pred)), 4)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro[tname] = round(float(f1), 4)
    return {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "f1_macro": f1_macro}


def train_lstm_multitask(df: pd.DataFrame) -> LstmTrainResult:
    """Validate the input table, build (N,600,40) tensors + targets, split
    random_stratified_cooler (70/15/15, stratified on cooler, seed=42), fit a fresh
    StandardScaler on train only, train a fresh LSTM from scratch with the original's exact
    hyperparameters/callbacks, and evaluate on the held-out test split.

    Memory (the pod runs it in a child next to the serving process, 2 GiB in total): the caller
    must not keep its own reference to ``df`` (plugin.train passes it straight from the reader),
    so the table is freed as soon as the tensors exist; the unscaled and the full scaled tensors
    are dropped once their slices are taken. The computation is unchanged."""
    _validate_training_frame(df)
    x, y = _build_tensors_and_targets(df)[:2]
    del df
    idx_splits = _split_cycle_indices(y[:, 0])

    class_mappings = _build_class_mappings()
    y_idx = _encode_targets(y, class_mappings)
    y_oh = _one_hot(y_idx, class_mappings)
    shap_background = xai.build_shap_background(x, idx_splits[0], len(idx_splits[2]))
    x_scaled, scaler_payload = _fit_scaler(x, idx_splits[0])
    del x
    x_parts = [x_scaled[idx] for idx in idx_splits]  # train, val, test
    del x_scaled
    sample_weights = _build_sample_weights(y)

    model = _build_multitask_lstm(class_mappings)
    history = _compile_and_fit(
        model, (x_parts[0], x_parts[1]), y_oh, sample_weights, idx_splits[:2],
    )
    del x_parts[:2]  # only the test slice is left
    metrics = _evaluate_test(model, x_parts[0], y_idx, idx_splits[2])

    return LstmTrainResult(
        model=model, scaler=scaler_payload, class_mappings=class_mappings, metrics=metrics,
        split_sizes={
            "n_train": len(idx_splits[0]), "n_val": len(idx_splits[1]),
            "n_test": len(idx_splits[2]),
        },
        epochs_run=len(history.history.get("loss", [])),
        shap_background=shap_background,
    )
