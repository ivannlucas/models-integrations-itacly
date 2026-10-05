"""XAI del entregable 48: Grad-CAM temporal, SHAP, puntos críticos de control y motor prescriptivo.

Port del paquete ``src/xai/`` del equipo de IA conservando su semántica:
- Grad-CAM 1D sobre ``features.17``, CAM interpolado a 600 pasos y normalizado a [0, 1].
- SHAP ``GradientExplainer`` por cabeza, |SHAP| medio agrupado por sensor base y normalizado.
- CCP: picos del CAM medio (>= 0.7) + sensor top de SHAP + regla física + severidad agregada.
- Riesgo: severidad x confianza, con bonus SHAP sobre los sensores de la regla física.

Grad-CAM (inmediato) y SHAP (caro, opcional) se calculan por separado: ninguna función de Grad-CAM
depende de SHAP. Si SHAP no se pide, los campos que dependen de él quedan a ``None``.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.constants import (
    CLASS_NAMES,
    DEFAULT_N_SAMPLES,
    GRADCAM_LAYER,
    GRADCAM_PEAK_THRESHOLD,
    HEAD_NAMES,
    MAINTENANCE_ACTIONS_FILENAME,
    RANDOM_STATE,
    RISK_CRITICAL,
    RISK_WARNING,
    SAMPLING_RATE_HZ,
    SENSOR_COLUMNS,
    SEVERITY_CRITICAL_RATIO,
    SEVERITY_WARNING_RATIO,
    SHAP_N_BACKGROUND,
    SHAP_N_SAMPLES,
)

logger = logging.getLogger(__name__)

# Los hooks de Grad-CAM y GradientExplainer mutan/registran sobre la instancia compartida del modelo:
# se serializa todo el cálculo XAI para que peticiones concurrentes no se pisen.
XAI_LOCK = threading.Lock()

PHYSICS_RULES = {
    "Bomba": {
        "description": "Eficiencia hidráulica: P_hyd = PS1 × FS1 × 1.6666, eff = P_hyd / (|EPS1| + 1)",
        "threshold": 0.75,
        "sensors": ["PS1", "FS1", "EPS1"],
        "condition": "efficiency < 0.75",
    },
    "Fouling": {
        "description": "Eficiencia térmica: Heat = m_dot × 3.9 × ΔT, eff = Heat / Q_expected",
        "threshold": 0.80,
        "sensors": ["TS1", "TS2", "FS1"],
        "condition": "thermal_efficiency < 0.80",
    },
}

DEFAULT_MAINTENANCE_ACTIONS = {
    "Fouling": {
        0: {"accion": "Sin acción requerida", "urgencia": "Ninguna", "intervalo_ciclos": None},
        1: {"accion": "Programar limpieza CIP en próximo turno", "urgencia": "Media", "intervalo_ciclos": 50},
        2: {"accion": "Detener producción y ejecutar limpieza CIP inmediata", "urgencia": "Alta", "intervalo_ciclos": 0},
    },
    "Válvula": {
        0: {"accion": "Sin acción requerida", "urgencia": "Ninguna", "intervalo_ciclos": None},
        1: {"accion": "Verificar posición y respuesta del actuador de válvulas", "urgencia": "Media", "intervalo_ciclos": 30},
        2: {"accion": "Reemplazo o reparación urgente de válvula de control", "urgencia": "Alta", "intervalo_ciclos": 0},
    },
    "Bomba": {
        0: {"accion": "Sin acción requerida", "urgencia": "Ninguna", "intervalo_ciclos": None},
        1: {"accion": "Inspección de cavitación y verificar presión diferencial", "urgencia": "Media", "intervalo_ciclos": 40},
        2: {"accion": "Reemplazo de bomba programado urgente — riesgo de fuga", "urgencia": "Alta", "intervalo_ciclos": 0},
    },
    "Acumulador": {
        0: {"accion": "Sin acción requerida", "urgencia": "Ninguna", "intervalo_ciclos": None},
        1: {"accion": "Recargar presión del acumulador hidráulico", "urgencia": "Media", "intervalo_ciclos": 60},
        2: {"accion": "Reemplazo de acumulador — presión fuera de rango operativo", "urgencia": "Alta", "intervalo_ciclos": 0},
    },
}
URGENCY_ORDER = {"Alta": 0, "Media": 1, "Ninguna": 2}


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _get_target_layer(model, layer_name: str):
    module = model
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def compute_gradcam(model, input_tensor, target_head: int = 0, target_class: int | None = None,
                    target_layer_name: str = GRADCAM_LAYER):
    """Grad-CAM temporal (1, n_sensors, T) -> (cam (T,) en [0,1], predicción)."""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    target_layer = _get_target_layer(model, target_layer_name)

    activations: dict = {}
    gradients: dict = {}
    handle_fwd = target_layer.register_forward_hook(lambda m, i, o: activations.__setitem__("value", o))
    handle_bwd = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.__setitem__("value", go[0]))

    try:
        with torch.enable_grad():
            logits = model(input_tensor)[target_head]
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, target_class].item()

            model.zero_grad()
            logits[0, target_class].backward()

            weights = gradients["value"].mean(dim=2, keepdim=True)
            cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=input_tensor.shape[2], mode="linear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()

    prediction = {
        "head": HEAD_NAMES[target_head],
        "predicted_class": target_class,
        "class_name": CLASS_NAMES[target_class],
        "confidence": confidence,
    }
    return cam, prediction


def compute_gradcam_all_heads(model, input_tensor, target_layer_name: str = GRADCAM_LAYER):
    results = []
    for head_idx in range(4):
        cam, prediction = compute_gradcam(model, input_tensor, target_head=head_idx,
                                          target_layer_name=target_layer_name)
        results.append({"cam": cam, "prediction": prediction})
    return results


def find_temporal_peaks(cam, threshold: float = GRADCAM_PEAK_THRESHOLD):
    """Segmentos (inicio, fin, intensidad media) donde el CAM supera el umbral."""
    peaks = []
    start = None
    for i, val in enumerate(cam >= threshold):
        if val and start is None:
            start = i
        elif not val and start is not None:
            peaks.append((start, i - 1, float(cam[start:i].mean())))
            start = None
    if start is not None:
        peaks.append((start, len(cam) - 1, float(cam[start:].mean())))
    return peaks


# ── SHAP ──────────────────────────────────────────────────────────────────────

class _HeadWrapper(torch.nn.Module):
    def __init__(self, model, target_head: int):
        super().__init__()
        self.model = model
        self.target_head = target_head

    def forward(self, x):
        return self.model(x)[self.target_head]


def compute_shap_values(model, X_background, X_explain, feature_cols, target_head: int = 0) -> dict:
    """|SHAP| medio por canal agrupado por sensor base y normalizado (suma = 1)."""
    import shap  # import perezoso: dependencia pesada, solo necesaria si se pide SHAP

    device = next(model.parameters()).device
    model.eval()
    explainer = shap.GradientExplainer(_HeadWrapper(model, target_head), X_background.to(device))
    raw = explainer.shap_values(X_explain.to(device))

    if isinstance(raw, list):
        raw = [v.cpu().numpy() if torch.is_tensor(v) else v for v in raw]
        all_shap = np.stack([np.abs(sv) for sv in raw]).mean(axis=0)
    else:
        raw = raw.cpu().numpy() if torch.is_tensor(raw) else raw
        all_shap = np.abs(raw)

    channel_importance = np.asarray(all_shap.mean(axis=(0, 2))).flatten()

    sensor_importance: dict[str, float] = {}
    for i, col in enumerate(feature_cols):
        for sensor in SENSOR_COLUMNS:
            if col.startswith(sensor):
                sensor_importance[sensor] = sensor_importance.get(sensor, 0.0) + float(channel_importance[i])
                break

    total = float(sum(sensor_importance.values()))
    if total > 0:
        sensor_importance = {k: v / total for k, v in sensor_importance.items()}
    return sensor_importance


def compute_shap_all_heads(model, X_background, X_explain, feature_cols) -> dict:
    return {
        HEAD_NAMES[h]: compute_shap_values(model, X_background, X_explain, feature_cols, target_head=h)
        for h in range(4)
    }


def seed_xai() -> None:
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


# ── Riesgo y prescriptivo ─────────────────────────────────────────────────────

def compute_risk_scores(predictions, confidences, shap_importance: dict | None = None) -> dict:
    risk_scores = {}
    severity_weights = {0: 0.0, 1: 0.5, 2: 1.0}
    for i, head in enumerate(HEAD_NAMES):
        score = severity_weights[predictions[i]] * confidences[i]
        if shap_importance and head in shap_importance:
            physics_sensors = PHYSICS_RULES.get(head, {}).get("sensors", [])
            if physics_sensors:
                score = score * (1.0 + sum(shap_importance[head].get(s, 0.0) for s in physics_sensors))
        score = min(score, 1.0)
        level = "CRÍTICO" if score >= RISK_CRITICAL else "WARNING" if score >= RISK_WARNING else "NORMAL"
        risk_scores[head] = {"score": round(score, 4), "level": level}
    return risk_scores


def load_maintenance_actions(path: Path | None = None) -> dict:
    """Acciones por componente/clase desde maintenance_actions.yaml; fallback a la tabla por defecto."""
    path = path or Path(__file__).with_name(MAINTENANCE_ACTIONS_FILENAME)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return DEFAULT_MAINTENANCE_ACTIONS

    configured: dict = {}
    for head in HEAD_NAMES:
        head_actions = raw.get(head)
        if not isinstance(head_actions, dict):
            return DEFAULT_MAINTENANCE_ACTIONS
        configured[head] = {}
        for idx, class_name in enumerate(CLASS_NAMES):
            info = head_actions.get(class_name)
            if not isinstance(info, dict):
                return DEFAULT_MAINTENANCE_ACTIONS
            configured[head][idx] = {
                "accion": info.get("accion"),
                "urgencia": info.get("urgencia"),
                "intervalo_ciclos": info.get("intervalo_ciclos"),
            }
    return configured


def generate_maintenance_recommendations(predictions, confidences, risk_scores: dict | None = None) -> list[dict]:
    actions = load_maintenance_actions()
    recommendations = []
    for i, head in enumerate(HEAD_NAMES):
        info = actions[head][predictions[i]]
        rec = {
            "componente": head,
            "estado_predicho": CLASS_NAMES[predictions[i]],
            "confianza": round(confidences[i], 4),
            "accion": info["accion"],
            "urgencia": info["urgencia"],
            "intervalo_ciclos": info["intervalo_ciclos"],
        }
        if risk_scores and head in risk_scores:
            rec["riesgo_score"] = risk_scores[head]["score"]
            rec["riesgo_nivel"] = risk_scores[head]["level"]
        recommendations.append(rec)
    recommendations.sort(key=lambda r: URGENCY_ORDER.get(r["urgencia"], 99))
    return recommendations


def build_local_report(predictions, confidences, gradcam_results, shap_all: dict | None,
                       cycle_id=None, include_cam: bool = False) -> list[dict]:
    """Informe local por componente (equivalente a xai_report_cycle.csv).

    Sin SHAP: ``sensor_mas_relevante`` / ``importancia_sensor`` = None y el riesgo se calcula sin el
    bonus SHAP (``riesgo_incluye_shap`` = False), de modo que no se mezclan ambos resultados.
    """
    risk = compute_risk_scores(predictions, confidences, shap_importance=shap_all)
    recs = {r["componente"]: r for r in generate_maintenance_recommendations(predictions, confidences, risk)}

    rows = []
    for i, head in enumerate(HEAD_NAMES):
        cam = gradcam_results[i]["cam"]
        peaks = find_temporal_peaks(cam)
        if peaks:
            start, end, intensity = max(peaks, key=lambda p: p[2])
            win = (round(start / SAMPLING_RATE_HZ, 1), round((end + 1) / SAMPLING_RATE_HZ, 1), round(intensity, 4))
        else:
            win = (None, None, 0.0)

        top_sensor = top_importance = None
        if shap_all and shap_all.get(head):
            top_sensor = max(shap_all[head], key=shap_all[head].get)
            top_importance = round(shap_all[head][top_sensor], 4)

        rec = recs[head]
        row = {
            "cycle_id": cycle_id,
            "componente": head,
            "prediccion": CLASS_NAMES[predictions[i]],
            "confianza": round(confidences[i], 4),
            "sensor_mas_relevante": top_sensor,
            "importancia_sensor": top_importance,
            "ventana_critica_inicio_s": win[0],
            "ventana_critica_fin_s": win[1],
            "intensidad_ventana": win[2],
            "accion": rec["accion"],
            "urgencia": rec["urgencia"],
            "intervalo_ciclos": rec["intervalo_ciclos"],
            "riesgo_score": rec["riesgo_score"],
            "riesgo_nivel": rec["riesgo_nivel"],
            "riesgo_incluye_shap": shap_all is not None,
        }
        if include_cam:
            row["gradcam_cam"] = [round(float(v), 4) for v in cam]
        rows.append(row)
    return rows


# ── Puntos críticos de control (análisis global) ──────────────────────────────

def detect_critical_control_points(model, X_all: torch.Tensor, feature_cols, cycle_ids=None,
                                   n_samples: int | None = DEFAULT_N_SAMPLES, include_shap: bool = False,
                                   random_state: int = RANDOM_STATE):
    """CCP a partir de Grad-CAM promedio (+ SHAP opcional) sobre hasta ``n_samples`` ciclos.

    X_all: tensor (N, n_features, T) ya escalado. Muestra aleatoria (seed 42) si N > n_samples.
    Returns (ccps: list[dict], summary: dict).
    """
    device = next(model.parameters()).device
    model.eval()

    n_total = X_all.shape[0]
    ids = [int(c) for c in cycle_ids] if cycle_ids is not None else None
    if n_samples and n_samples < n_total:
        rng = np.random.default_rng(random_state)
        sample_idx = np.sort(rng.choice(n_total, size=n_samples, replace=False))
        X_all = X_all[sample_idx]
        analyzed = [ids[i] for i in sample_idx] if ids is not None else None
    else:
        analyzed = ids
    n_cycles = X_all.shape[0]
    analyzed_str = ",".join(str(c) for c in analyzed) if analyzed is not None else "no disponible"

    cam_acc = {h: np.zeros(X_all.shape[2]) for h in HEAD_NAMES}
    predictions_all = []
    for i in range(n_cycles):
        results = compute_gradcam_all_heads(model, X_all[i:i + 1].to(device))
        predictions_all.append([r["prediction"]["predicted_class"] for r in results])
        for h_idx, res in enumerate(results):
            cam_acc[HEAD_NAMES[h_idx]] += res["cam"]
    for h in HEAD_NAMES:
        avg = cam_acc[h] / n_cycles
        mx = avg.max()
        cam_acc[h] = avg / mx if mx > 0 else avg

    shap_all: dict | None = None
    shap_n_explained = None
    shap_explained_cycles = None
    if include_shap:
        rng = np.random.default_rng(random_state)
        n_bg = min(SHAP_N_BACKGROUND, n_cycles)
        bg_idx = rng.choice(n_cycles, n_bg, replace=False)
        shap_n_explained = min(SHAP_N_SAMPLES, n_cycles)
        exp_idx = rng.choice(n_cycles, shap_n_explained, replace=False)
        if analyzed is not None:
            shap_explained_cycles = sorted(analyzed[i] for i in exp_idx)
        seed_xai()
        shap_all = compute_shap_all_heads(model, X_all[bg_idx].to(device), X_all[exp_idx].to(device), feature_cols)

    ccps = []
    for h_idx, head in enumerate(HEAD_NAMES):
        peaks = find_temporal_peaks(cam_acc[head])

        top_sensor = top_importance = None
        if shap_all and shap_all.get(head):
            top_sensor = max(shap_all[head], key=shap_all[head].get)
            top_importance = round(shap_all[head][top_sensor], 4)

        head_preds = [p[h_idx] for p in predictions_all]
        if sum(1 for p in head_preds if p == 2) / n_cycles > SEVERITY_CRITICAL_RATIO:
            severity = "CRÍTICO"
        elif sum(1 for p in head_preds if p == 1) / n_cycles > SEVERITY_WARNING_RATIO:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        rule_desc = PHYSICS_RULES.get(head, {}).get("description", "Sin regla física directa")
        sensor_txt = f"el sensor {top_sensor}" if top_sensor else "los sensores del componente"
        base = {
            "componente": head,
            "sensor_critico": top_sensor,
            "sensor_importancia": top_importance,
            "regla_fisica": rule_desc,
            "severidad": severity,
            "n_ciclos_analizados": n_cycles,
            "ciclos_analizados": analyzed_str,
        }
        if peaks:
            for start, end, intensity in peaks:
                start_s = round(start / SAMPLING_RATE_HZ, 1)
                end_s = round((end + 1) / SAMPLING_RATE_HZ, 1)
                ccps.append({
                    **base,
                    "ventana_inicio": start,
                    "ventana_fin": end,
                    "ventana_inicio_s": start_s,
                    "ventana_fin_s": end_s,
                    "intensidad_media": round(intensity, 4),
                    "descripcion_operador": (
                        f"Vigilar {sensor_txt} entre el segundo {start_s} y el segundo {end_s} de cada "
                        f"ciclo de pasteurizado (60 s). Aplica a los {n_cycles} ciclos analizados."
                    ),
                })
        else:
            ccps.append({
                **base,
                "ventana_inicio": None, "ventana_fin": None,
                "ventana_inicio_s": None, "ventana_fin_s": None,
                "intensidad_media": 0.0,
                "descripcion_operador": (
                    f"Sin ventana temporal crítica detectada para {head} en los {n_cycles} ciclos analizados."
                ),
            })

    shap_rows = []
    if shap_all:
        for head, sensors in shap_all.items():
            for sensor, imp in sensors.items():
                shap_rows.append({
                    "componente": head,
                    "sensor": sensor,
                    "importancia": round(imp, 4),
                    "n_ciclos_agregados": shap_n_explained,
                    "ciclos_agregados": ",".join(str(c) for c in shap_explained_cycles)
                    if shap_explained_cycles is not None else "no disponible",
                })

    summary = {
        "n_cycles_analyzed": n_cycles,
        "cycles_analyzed": analyzed,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "shap_applied": shap_all is not None,
        "shap": shap_rows,
    }
    return ccps, summary
