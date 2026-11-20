"""Integrated Gradients + SHAP for ml39 — faithful port of
inbox/a39/codigo/src/xai/xai_analysis.py (``_integrated_gradients``, the SHAP
``GradientExplainer`` block), adapted to run on one cycle at a time instead of a batch sweep
over an entire test split. Steps/background size come from the "full" XAI profile
(constants.IG_STEPS=50, SHAP_N_BACKGROUND=80) — the same profile that produced the memoria's
published figures and this plugin's golden cases (see inbox/a39/manifest.yaml).

IMPORTANT (see model_loader.py and inbox/a39/manifest.yaml known_issues): importing
``app.registry`` — which every real server process does, MODEL filter or not — imports ml1's
(and, once merged, a37's) model_loader module, which sets ``KERAS_BACKEND=torch`` as a process
environment variable. Keras 3 locks its backend on first import, so this silently makes
``tf.keras`` route through the standalone ``keras`` package running on **torch**, not real
TensorFlow, for the rest of that process — breaking ``tf.GradientTape`` (it cannot differentiate
a torch tensor). Rather than depend on luck about import order, Integrated Gradients here
dispatches on ``keras.config.backend()`` at call time and uses the matching autodiff API; the
Integrated Gradients formula and the gradient-averaging math are identical either way, only the
mechanism for getting the gradient changes.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from app.plugins.ml39_wine_predictive_maintenance_xai.constants import (
    IG_STEPS,
    PHYSICAL_SENSORS,
    TARGET_NAMES,
)

_N_ACTIVE = len(PHYSICAL_SENSORS)


def _active_backend() -> str:
    import keras  # pylint: disable=import-outside-toplevel
    return keras.config.backend()


def _head_score(model, interp, head_idx: int, target_class_idx):
    """The scalar-per-row score IG differentiates: a fixed class's logit, or (when
    ``target_class_idx`` is None) the per-row max over classes — xai_analysis.py's own
    semantics, re-evaluated at every interpolation step, not fixed to the final prediction.
    Framework-agnostic: ``interp`` may be a tf or a torch tensor, ``model`` a Keras 3 model."""
    model_input = interp
    model_inputs = getattr(model, "inputs", None)
    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        input_name = model_inputs[0].name.split(":")[0]
        model_input = {input_name: interp}
    preds_interp = model(model_input, training=False)
    if not isinstance(preds_interp, list):
        preds_interp = [preds_interp]
    out = preds_interp[head_idx]
    if target_class_idx is not None:
        return out[:, target_class_idx]
    return out.max(dim=1).values if hasattr(out, "dim") else tf.reduce_max(out, axis=1)


# pylint: disable-next=too-many-locals
def _integrated_gradients_tf(model, inputs: tf.Tensor, target_class_idx, head_idx, steps: int):
    """TensorFlow path — faithful port of xai_analysis.py::_integrated_gradients. One cohesive
    interpolate/forward/gradient/average procedure — splitting it further would hide the math,
    not clarify it (same reasoning as the torch twin below)."""
    baseline = tf.zeros_like(inputs)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated = baseline + alphas[:, None, None, None] * (inputs - baseline)
    orig_shape = tf.shape(inputs)
    n_steps_plus_1 = steps + 1
    interp_flat = tf.reshape(
        interpolated, [n_steps_plus_1 * orig_shape[0], orig_shape[1], orig_shape[2]]
    )

    with tf.GradientTape() as tape:
        tape.watch(interp_flat)
        score = _head_score(model, interp_flat, head_idx, target_class_idx)

    grads = tape.gradient(score, interp_flat)
    grads_reshaped = tf.reshape(
        grads, [n_steps_plus_1, orig_shape[0], orig_shape[1], orig_shape[2]]
    )
    avg_grads = tf.reduce_mean(grads_reshaped, axis=0)
    attrs = (inputs - baseline) * avg_grads
    return attrs.numpy()


# pylint: disable-next=too-many-locals
def _integrated_gradients_torch(model, inputs: np.ndarray, target_class_idx, head_idx, steps: int):
    """Torch path — same Integrated Gradients formula, torch autograd instead of
    ``tf.GradientTape`` (needed when something else in this process has locked Keras 3's
    backend to torch — see module docstring). One cohesive numerical procedure, same shape as
    the TensorFlow twin above."""
    import torch  # pylint: disable=import-outside-toplevel

    inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
    baseline = torch.zeros_like(inputs_t)
    alphas = torch.linspace(0.0, 1.0, steps + 1)
    interpolated = baseline + alphas[:, None, None, None] * (inputs_t - baseline)
    orig_shape = inputs_t.shape
    n_steps_plus_1 = steps + 1
    interp_flat = interpolated.reshape(n_steps_plus_1 * orig_shape[0], orig_shape[1], orig_shape[2])
    interp_flat = interp_flat.clone().detach().requires_grad_(True)

    score = _head_score(model, interp_flat, head_idx, target_class_idx)
    grads, = torch.autograd.grad(score.sum(), interp_flat)

    grads_reshaped = grads.reshape(n_steps_plus_1, orig_shape[0], orig_shape[1], orig_shape[2])
    avg_grads = grads_reshaped.mean(dim=0)
    attrs = (inputs_t - baseline) * avg_grads
    return attrs.detach().numpy()


def _integrated_gradients(
    model, inputs: np.ndarray, target_class_idx: int | None, head_idx: int,
    steps: int = IG_STEPS,
) -> np.ndarray:
    """Dispatch to the TensorFlow or torch autodiff path depending on which backend Keras 3
    actually has active in this process (see module docstring) — same formula either way."""
    if _active_backend() == "torch":
        return _integrated_gradients_torch(model, inputs, target_class_idx, head_idx, steps)
    tf_inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    return _integrated_gradients_tf(model, tf_inputs, target_class_idx, head_idx, steps)


def _sensor_importance(attrs: np.ndarray) -> dict[str, float]:
    """|attrs| mean over the 600 timesteps, per physical sensor (ignores the 26 padding/virtual
    slots) — same aggregation as xai_analysis.py's ``ig_sensor_imp``/``shap_sensor_imp``."""
    active = attrs[:, :, :_N_ACTIVE]
    importance = np.abs(active).mean(axis=(0, 1))
    return {name: float(importance[i]) for i, name in enumerate(PHYSICAL_SENSORS)}


def compute_ig_importance(model, x_scaled: np.ndarray) -> dict[str, dict[str, float]]:
    """Integrated Gradients importance by sensor, per head, for one cycle's scaled tensor
    (shape (1, 600, 40)). ``target_class_idx=None`` — faithful to xai_analysis.py: this tracks
    the max over classes AT EACH interpolation step, not a single class fixed to the final
    prediction. Passing the predicted class's fixed index instead gives numerically different
    (larger) attributions — confirmed against the manifest's golden cases, do not "simplify"
    this back to a fixed class."""
    result: dict[str, dict[str, float]] = {}
    for head_idx, target in enumerate(TARGET_NAMES):
        attrs = _integrated_gradients(model, x_scaled, target_class_idx=None, head_idx=head_idx)
        result[target] = _sensor_importance(attrs)
    return result


def compute_shap_importance(
    model, x_scaled: np.ndarray, shap_background: np.ndarray,
) -> dict[str, dict[str, float]]:
    """SHAP importance by sensor, per head, for one cycle — faithful port of the
    ``shap.GradientExplainer`` block in xai_analysis.py, using the persisted background.

    Only supported when TensorFlow is the active Keras 3 backend in this process (see module
    docstring): ``shap.GradientExplainer`` needs a real TF graph/tape, not a torch-backed Keras
    model — unlike Integrated Gradients, it has no torch-native fallback here.
    """
    if _active_backend() == "torch":
        raise RuntimeError(
            "SHAP no disponible: el backend de Keras 3 activo en este proceso es 'torch' "
            "(contaminado por otro plugin que fija KERAS_BACKEND=torch al importarse, p.ej. "
            "ml1/ml37 — ver inbox/a39/manifest.yaml known_issues), y shap.GradientExplainer "
            "requiere TensorFlow real. Integrated Gradients sigue funcionando (tiene una vía "
            "torch nativa); SHAP no."
        )

    import shap  # pylint: disable=import-outside-toplevel

    x_expl = np.asarray(x_scaled, dtype=np.float32)
    x_bg = np.asarray(shap_background, dtype=np.float32)
    result: dict[str, dict[str, float]] = {}
    for head_idx, target in enumerate(TARGET_NAMES):
        # no-member: tf.keras is a lazy loader pylint cannot see.
        sub_model = tf.keras.Model(  # pylint: disable=no-member
            inputs=model.input, outputs=model.outputs[head_idx]
        )
        explainer = shap.GradientExplainer(sub_model, x_bg)
        shap_vals = explainer.shap_values(x_expl)
        sv = np.stack(shap_vals, axis=-1) if isinstance(shap_vals, list) else np.array(shap_vals)
        if sv.ndim == 4:
            attrs = np.abs(sv).mean(axis=-1)
        else:
            attrs = np.abs(sv)
        result[target] = _sensor_importance(attrs)
    return result
