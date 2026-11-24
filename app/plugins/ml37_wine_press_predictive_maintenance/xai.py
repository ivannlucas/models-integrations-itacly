"""Explainability layer for ml37 — the per-cycle part of deliverable a39.

a39 ("Implementación del modelo XAI…", inbox/a39/) is not a new model: it explains this plugin's
exact LSTM (MD5-identical best_model.keras, same run_id no_virtual_v1__rand_scool). Ported from
inbox/a39/codigo/src/xai/xai_analysis.py:

- ``integrated_gradients`` = ``_integrated_gradients`` (baseline 0, ``target_class_idx=None``:
  the max over classes re-evaluated at every interpolation step, steps=50 of the "full" profile).
- ``compute_shap_importance`` = the ``shap.GradientExplainer`` block (one sub-model per head,
  |SHAP| averaged over classes and timesteps).
- ``sensor_importance`` = the ``ig_sensor_imp``/``shap_sensor_imp`` aggregation (|attr| mean
  over timesteps, only the 14 physical sensors; the 26 virtual/reserved slots are dropped).
- ``background_positions`` = the original's ``np.random.default_rng(42)`` call sequence for the
  SHAP background.

Keras 3 backend. The original uses ``tf.GradientTape``. In the server this model always runs on
the torch backend (model_loader.py sets it, and ml1's loader does the same as soon as
app.registry is imported), so the gradient comes from torch autograd; when the process locked
Keras to TensorFlow first (e.g. some pytest orders) it comes from ``tf.GradientTape``. Only the
gradient call differs — interpolation, averaging and aggregation are shared — and both give the
20 golden cases of inbox/a39/manifest.yaml. ``shap.GradientExplainer`` picks its TensorFlow or
PyTorch path from the model (a Keras-torch model is a ``torch.nn.Module``); both draw their
random samples with the same ``np.random`` call sequence. This replaces the old separate ml39
plugin, whose ``tf.GradientTape``-only SHAP always failed in the server (backend = torch).

t-SNE, calibration (Brier) and the test-set-wide IG-SHAP / neurosymbolic concordance are audit
analyses over the whole test split in the original, not explanations of one cycle: they stay
offline (``python src/main.py xai``) and their published figures are in constants.METRICS_REPORTED.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import torch  # noqa: E402  pylint: disable=wrong-import-position

from app.plugins.ml37_wine_press_predictive_maintenance.constants import (  # noqa: E402  pylint: disable=wrong-import-position
    IG_STEPS,
    PHYSICAL_SENSORS,
    SHAP_N_BACKGROUND,
    SHAP_NSAMPLES,
    TARGET_NAMES,
    XAI_IG_N_SAMPLES,
    XAI_SEED,
)

_N_ACTIVE = len(PHYSICAL_SENSORS)


def _head_positions(model) -> dict[str, int]:
    output_names = list(getattr(model, "output_names", TARGET_NAMES))
    return {name: i for i, name in enumerate(output_names)}


def _max_score_gradients_torch(model, interp_flat: np.ndarray) -> dict[str, np.ndarray]:
    """d(max softmax)/d(input) per head, torch autograd, one shared forward pass."""
    pos = _head_positions(model)
    x = torch.as_tensor(interp_flat).requires_grad_(True)
    with torch.enable_grad():
        outputs = model(x, training=False)
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        grads: dict[str, np.ndarray] = {}
        for i, target in enumerate(TARGET_NAMES):
            score = outputs[pos[target]].max(dim=1).values
            grad, = torch.autograd.grad(score.sum(), x, retain_graph=i < len(TARGET_NAMES) - 1)
            grads[target] = grad.detach().cpu().numpy()
    return grads


def _max_score_gradients_tf(model, interp_flat: np.ndarray) -> dict[str, np.ndarray]:
    """Same with tf.GradientTape (the original's API), one shared forward pass."""
    import tensorflow as tf  # pylint: disable=import-outside-toplevel  # only on the TF backend

    pos = _head_positions(model)
    x = tf.convert_to_tensor(interp_flat)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        outputs = model(x, training=False)
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        scores = {t: tf.reduce_max(outputs[pos[t]], axis=1) for t in TARGET_NAMES}
    grads = {t: tape.gradient(scores[t], x).numpy() for t in TARGET_NAMES}
    del tape
    return grads


def integrated_gradients(
    model, x_scaled: np.ndarray, steps: int = IG_STEPS,
) -> dict[str, np.ndarray]:
    """Integrated Gradients attributions per head, each shaped like ``x_scaled`` (N, 600, 40).

    Same formula as xai_analysis.py::_integrated_gradients (baseline 0, ``steps + 1``
    interpolation points, score = max softmax over classes at each point, gradients averaged
    over the points, times ``inputs - baseline``). The interpolated batch goes through the model
    once and every head is differentiated from that forward pass: the same attributions as the
    original's one pass per head, at about half the cost.
    """
    inputs = np.asarray(x_scaled, dtype=np.float32)
    baseline = np.zeros_like(inputs)
    n_cycles, timesteps, n_features = inputs.shape
    alphas = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
    interpolated = baseline + alphas[:, None, None, None] * (inputs - baseline)
    interp_flat = interpolated.reshape((steps + 1) * n_cycles, timesteps, n_features)

    if keras.config.backend() == "torch":
        grads = _max_score_gradients_torch(model, interp_flat)
    else:
        grads = _max_score_gradients_tf(model, interp_flat)
    return {
        target: (inputs - baseline) * grad.reshape(
            steps + 1, n_cycles, timesteps, n_features,
        ).mean(axis=0)
        for target, grad in grads.items()
    }


def sensor_importance(attrs: np.ndarray) -> dict[str, float]:
    """|attr| mean over cycles and timesteps per physical sensor — the original's
    ``np.abs(attrs[:, :, :n_active]).mean(axis=(0, 1))``."""
    importance = np.abs(attrs[:, :, :_N_ACTIVE]).mean(axis=(0, 1))
    return {name: float(importance[i]) for i, name in enumerate(PHYSICAL_SENSORS)}


def compute_ig_importance(model, x_scaled: np.ndarray) -> dict[str, dict[str, float]]:
    """head -> sensor -> |IG| mean for one cycle's scaled tensor (1, 600, 40)."""
    attrs = integrated_gradients(model, x_scaled)
    return {target: sensor_importance(attrs[target]) for target in TARGET_NAMES}


def shap_attributions(
    model, x_scaled: np.ndarray, background_scaled: np.ndarray, *, rseed: int | None = XAI_SEED,
) -> dict[str, np.ndarray]:
    """|SHAP| per head, averaged over classes: (N, 600, 40) each — the original's
    ``shap.GradientExplainer(sub_model, X_bg).shap_values(X_expl)`` block.

    ``rseed=XAI_SEED`` makes each request deterministic (same cycle, same explanation).
    ``rseed=None`` reproduces the original's seed chain instead: it never passes ``rseed``, so
    every head draws it from the global ``np.random`` state that ``set_random_seed(42)`` seeded;
    the caller seeds that state (only the verification script does). The global ``np.random``
    state is restored afterwards so other code in the process is not affected.
    """
    import shap  # pylint: disable=import-outside-toplevel  # heavy, only for include_shap

    pos = _head_positions(model)
    x_expl = np.asarray(x_scaled, dtype=np.float32)
    x_bg = np.asarray(background_scaled, dtype=np.float32)
    if keras.config.backend() == "torch":  # shap's PyTorch path wants tensors, not arrays
        x_expl, x_bg = torch.as_tensor(x_expl), torch.as_tensor(x_bg)
    saved_state = np.random.get_state() if rseed is not None else None
    try:
        result: dict[str, np.ndarray] = {}
        for target in TARGET_NAMES:
            sub_model = keras.Model(inputs=model.inputs, outputs=model.outputs[pos[target]])
            explainer = shap.GradientExplainer(sub_model, x_bg)
            shap_vals = explainer.shap_values(x_expl, nsamples=SHAP_NSAMPLES, rseed=rseed)
            sv = (np.stack(shap_vals, axis=-1) if isinstance(shap_vals, list)
                  else np.array(shap_vals))
            result[target] = np.abs(sv).mean(axis=-1) if sv.ndim == 4 else np.abs(sv)
        return result
    finally:
        if saved_state is not None:
            np.random.set_state(saved_state)


def compute_shap_importance(
    model, x_scaled: np.ndarray, background_scaled: np.ndarray,
) -> dict[str, dict[str, float]]:
    """head -> sensor -> |SHAP| mean for one cycle, deterministic per request."""
    attrs = shap_attributions(model, x_scaled, background_scaled)
    return {target: sensor_importance(attrs[target]) for target in TARGET_NAMES}


def background_positions(n_train: int, n_test: int) -> np.ndarray:
    """Positions (into the train split) of the SHAP background cycles.

    Replays xai_analysis.py's ``rng = np.random.default_rng(42)`` sequence: first the IG sample
    over the test split (``rng.choice(len(test_idx), 150)``), then the background over the train
    split (``rng.choice(len(train_idx), 80)``). With the official split of run
    no_virtual_v1__rand_scool this gives the exact background the "full" profile used.
    """
    rng = np.random.default_rng(XAI_SEED)
    if n_test > 0:
        rng.choice(n_test, size=min(XAI_IG_N_SAMPLES, n_test), replace=False)
    return rng.choice(n_train, size=min(SHAP_N_BACKGROUND, n_train), replace=False)


def build_shap_background(
    x_schema_raw: np.ndarray, train_idx: np.ndarray, n_test: int,
) -> np.ndarray:
    """Unscaled background (n, 600, 14) float32 for ``SHAP_BACKGROUND_FILENAME``.

    ``x_schema_raw`` holds every cycle unscaled with the 14 physical sensors first (the
    schema_v1 tensor or the delivered X.npy), ``train_idx`` the cycles of the train split and
    ``n_test`` the size of the test split. Used by ``training.train_lstm_multitask`` for a
    retrained model and, for the base artifact, with the delivered tensor and split::

        X = np.load("inbox/a37/codigo/data/processed/tensors/X.npy")
        splits = json.load(open(".../data/splits/no_virtual_v1__rand_scool/splits.json"))
        bg = build_shap_background(X, np.array(splits["train_idx"]), len(splits["test_idx"]))
        np.save("artifacts/ml37_wine_press_predictive_maintenance/shap_background.npy", bg)
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    chosen = train_idx[background_positions(len(train_idx), n_test)]
    return np.asarray(x_schema_raw[chosen][:, :, :_N_ACTIVE], dtype=np.float32)
