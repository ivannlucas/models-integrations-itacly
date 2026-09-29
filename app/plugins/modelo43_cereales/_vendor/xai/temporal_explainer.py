"""
Módulo de explicabilidad temporal: análisis LSTM+SHAP y patrones temporales.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional


class _DLBranchForSHAP(nn.Module):
    """Wrapper para explicar solo la rama DL con SHAP."""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x_window):
        lstm_out, _ = self.base.lstm_branch(x_window)
        attn_logits = self.base.attention_layer(lstm_out)
        attn_weights = torch.softmax(attn_logits, dim=1)
        temporal_context = torch.sum(lstm_out * attn_weights, dim=1)
        dl_embed = self.base.dl_projection(temporal_context)
        return self.base.anomaly_dl(dl_embed)


def _resolve_temporal_feature_names(feature_names_candidate, n_features_temporal: int):
    """Resuelve nombres de features temporales."""
    F = int(n_features_temporal)
    if feature_names_candidate is None:
        return [f"feature_{i}" for i in range(F)]
    names = list(feature_names_candidate)
    if len(names) == F:
        return names
    return [f"feature_{i}" for i in range(F)]


def _split_series_spans(T: int) -> Dict[str, slice]:
    """Divide una serie temporal en tres tramos."""
    one_third = max(1, T // 3)
    return {
        "inicio": slice(0, one_third),
        "medio": slice(one_third, min(2 * one_third, T)),
        "final": slice(min(2 * one_third, T), T),
    }


def _top_k_indices(values: np.ndarray, k: int) -> List[int]:
    """Obtiene índices del top-k."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return []
    idx = np.argsort(-values)
    return idx[:min(k, len(idx))].tolist()


def _variable_span_profile(
    abs_shap_var: np.ndarray,
    spans: Dict[str, slice],
) -> Dict[str, Any]:
    """Calcula perfil por tramo para una variable."""
    abs_shap_var = np.asarray(abs_shap_var, dtype=float).reshape(-1)

    span_scores = {}
    for span_name, sl in spans.items():
        if sl.start >= sl.stop:
            span_scores[span_name] = 0.0
        else:
            span_scores[span_name] = float(abs_shap_var[sl].sum())

    dominant_span = max(span_scores, key=span_scores.get) if span_scores else "unknown"
    total = float(sum(span_scores.values()))
    dominant_ratio = float(span_scores.get(dominant_span, 0.0) / total) if total > 0 else 0.0

    return {
        "span_scores": span_scores,
        "dominant_span": dominant_span,
        "dominant_span_ratio": dominant_ratio,
    }


def _set_temporal_explainer_seed(random_state: int) -> None:
    """Fija seeds de NumPy y PyTorch para reproducibilidad."""
    np.random.seed(int(random_state))
    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))


class TemporalExplainer:
    """Generador de explicaciones temporales con SHAP."""

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names

    def explain_sample(
        self,
        x_window: np.ndarray,
        background_windows: np.ndarray,
        decision_threshold: float = 0.5,
        n_background: int = 64,
        random_state: int = 42,
        top_variables: int = 8,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        """Genera explicación temporal para una muestra."""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP no está instalado. Instala con: pip install shap")

        self.model.eval()
        device = next(self.model.parameters()).device
        _set_temporal_explainer_seed(random_state)

        # Preparar x_window
        if not torch.is_tensor(x_window):
            x_one = torch.tensor(x_window, dtype=torch.float32)
        else:
            x_one = x_window.detach().float()

        if x_one.dim() == 2:
            x_one = x_one.unsqueeze(0)

        if x_one.dim() != 3:
            raise ValueError(
                f"x_window debe ser [1,T,F] o [T,F], recibido {tuple(x_one.shape)}"
            )

        x_one = x_one.to(device)
        _, T, F = x_one.shape
        feature_names_local = _resolve_temporal_feature_names(self.feature_names, F)

        # Preparar background
        X_bg = np.asarray(background_windows)
        if X_bg.ndim != 3:
            raise ValueError(f"background_windows debe ser [N,T,F], recibido {X_bg.shape}")
        if X_bg.shape[1] != T or X_bg.shape[2] != F:
            raise ValueError(
                f"background_windows incompatible con x_window. Esperado [N,{T},{F}], recibido {X_bg.shape}"
            )

        N_bg_total = X_bg.shape[0]
        rng = np.random.default_rng(seed=random_state)
        bg_n = int(min(max(8, n_background), N_bg_total))
        bg_idx = rng.choice(np.arange(N_bg_total), size=bg_n, replace=False)
        x_bg = torch.tensor(X_bg[bg_idx], dtype=torch.float32, device=device)

        prev_benchmark = torch.backends.cudnn.benchmark
        prev_deterministic = torch.backends.cudnn.deterministic
        prev_algorithms = torch.are_deterministic_algorithms_enabled()

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

        try:
            # Forward nativo
            with torch.no_grad():
                lstm_out, _ = self.model.lstm_branch(x_one)
                attn_logits = self.model.attention_layer(lstm_out)
                attn_weights = torch.softmax(attn_logits, dim=1)
                temporal_context = torch.sum(lstm_out * attn_weights, dim=1)
                temporal_embed = self.model.dl_projection(temporal_context)
                logit_temporal = self.model.anomaly_dl(temporal_embed)

            temporal_logit = float(logit_temporal.view(-1)[0].detach().cpu().item())
            temporal_prob = float(torch.sigmoid(logit_temporal).view(-1)[0].detach().cpu().item())
            temporal_class = int(temporal_prob >= float(decision_threshold))

            # SHAP
            wrapper = _DLBranchForSHAP(self.model).to(device)
            wrapper.eval()

            _set_temporal_explainer_seed(random_state)
            try:
                with torch.backends.cudnn.flags(enabled=False):
                    explainer = shap.GradientExplainer(wrapper, x_bg)
                    shap_vals = explainer.shap_values(x_one)
            except RuntimeError:
                wrapper.train()
                with torch.backends.cudnn.flags(enabled=False):
                    explainer = shap.GradientExplainer(wrapper, x_bg)
                    shap_vals = explainer.shap_values(x_one)
                wrapper.eval()
        finally:
            torch.backends.cudnn.benchmark = prev_benchmark
            torch.backends.cudnn.deterministic = prev_deterministic
            torch.use_deterministic_algorithms(prev_algorithms, warn_only=True)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        sig = self._aggregate_temporal_shap_signature(
            shap_values_3d=shap_vals,
            attention_weights_3d=attn_weights.detach().cpu().numpy(),
            feature_names=feature_names_local,
            top_variables=top_variables,
            include_raw=include_raw,
        )

        sig.update({
            "logit": float(temporal_logit),
            "probability": float(temporal_prob),
            "predicted_class": int(temporal_class),
            "decision_threshold": float(decision_threshold),
            "explained_target": "anomaly_logit_temporal",
        })

        return sig

    def _aggregate_temporal_shap_signature(
        self,
        shap_values_3d,
        attention_weights_3d=None,
        feature_names=None,
        top_variables: int = 8,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        """Agrega SHAP en una firma temporal."""
        sv = np.asarray(shap_values_3d)

        if sv.ndim == 4 and sv.shape[-1] == 1:
            sv = sv[..., 0]
        if sv.ndim == 3:
            sv = sv[0]
        if sv.ndim != 2:
            raise ValueError(f"Se esperaba SHAP [T,F], recibido {sv.shape}")

        T, F = sv.shape
        feature_names = _resolve_temporal_feature_names(feature_names, F)

        abs_sv = np.abs(sv)
        variable_importance = abs_sv.sum(axis=0)
        time_importance = abs_sv.sum(axis=1)

        attention_time = None
        if attention_weights_3d is not None:
            aw = np.asarray(attention_weights_3d)
            if aw.ndim == 3:
                aw = aw[0]
            if aw.ndim == 2 and aw.shape[-1] == 1:
                aw = aw[:, 0]
            if aw.ndim == 1 and len(aw) == T:
                attention_time = aw.astype(float)

        spans = _split_series_spans(T)

        temporal_by_var = {} if include_raw else None
        span_importance = {}
        span_attention = {}

        for span_name, sl in spans.items():
            if sl.start >= sl.stop:
                if include_raw:
                    temporal_by_var[span_name] = np.zeros(F, dtype=float)
                span_importance[span_name] = 0.0
                span_attention[span_name] = 0.0
            else:
                if include_raw:
                    temporal_by_var[span_name] = abs_sv[sl, :].sum(axis=0)
                span_importance[span_name] = float(time_importance[sl].sum())
                if attention_time is not None:
                    span_attention[span_name] = float(attention_time[sl].sum())
                else:
                    span_attention[span_name] = 0.0

        dominant_span_shap = max(span_importance, key=span_importance.get) if span_importance else "unknown"
        dominant_span_attention = max(span_attention, key=span_attention.get) if span_attention else "unknown"

        top_var_idx = _top_k_indices(variable_importance, top_variables)
        top_vars = []
        for i in top_var_idx:
            p = _variable_span_profile(abs_sv[:, i], spans)
            top_vars.append({
                "feature_idx": int(i),
                "feature": str(feature_names[i]),
                "importance": float(variable_importance[i]),
                "dominant_span": p["dominant_span"],
                "dominant_span_ratio": float(p["dominant_span_ratio"]),
                "span_scores": p["span_scores"],
            })

        report = {
            "span_importance": span_importance,
            "span_attention": span_attention,
            "dominant_span_shap": dominant_span_shap,
            "dominant_span_attention": dominant_span_attention,
            "top_variables": top_vars,
            "feature_names": feature_names,
        }

        if include_raw:
            report.update({
                "raw_shap": sv,
                "abs_shap": abs_sv,
                "variable_importance": variable_importance,
                "time_importance": time_importance,
                "attention_time": attention_time,
                "temporal_by_var": temporal_by_var,
            })

        return report
