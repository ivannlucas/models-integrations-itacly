"""XAI del entregable 48: Grad-CAM temporal, SHAP, puntos críticos de control y motor prescriptivo.

Port del paquete ``src/xai/`` del equipo de IA conservando su semántica:
- Grad-CAM 1D sobre ``features.17``, CAM interpolado a 600 pasos y normalizado a [0, 1].
- SHAP ``GradientExplainer`` por cabeza, |SHAP| medio agrupado por sensor base y normalizado.
- Riesgo: severidad x confianza, con bonus SHAP sobre los sensores de la regla física.
- XAI por ciclo con el contrato de 43/44 (``xai_result`` / ``xai_error``): ver ``build_xai_result`` y ``explain_cycle``.

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

from app.plugins.ml48_dairy_dnsl_critical_point_detection.constants import (
    CLASS_NAMES,
    GRADCAM_LAYER,
    GRADCAM_PEAK_THRESHOLD,
    HEAD_NAMES,
    MAINTENANCE_ACTIONS_FILENAME,
    RANDOM_STATE,
    RISK_CRITICAL,
    RISK_WARNING,
    SAMPLING_RATE_HZ,
    SENSOR_COLUMNS,
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


# ── XAI por ciclo con el contrato de 43/44 ───────────────────────────────────

ESTADO_POR_CLASE = {0: "Normal", 1: "Warning", 2: "Crítico"}
MENSAJE_NORMAL = "Todos los componentes en estado Sano: no se requiere intervención."


def build_xai_result(local_rows: list[dict]) -> dict:
    """Construye el ``xai_result`` de un ciclo reutilizando las claves del ``final_report`` de modelo-43.

    Claves de 43 (las que lee la plataforma): ``Estado_del_sistema.Estado_interpretativo``,
    ``Bloques_principales`` y ``Variables_clave`` (ausentes si el estado es Normal), ``Acciones_sugeridas``
    (estado no Normal) o ``Mensaje_operativo`` (estado Normal). ``Detalle_componentes`` es una clave adicional
    con la ventana crítica, el riesgo y la acción de cada componente (la plataforma la ignora por ahora).

    - Estado: el peor estado de los cuatro componentes (Normal / Warning / Crítico).
    - Bloques: componentes alterados (estado distinto de Sano).
    - Variables: sensores de la regla física de cada componente alterado y, si hubo SHAP, su sensor más relevante.
    - Acciones: acción prescriptiva de cada componente alterado, por urgencia.
    """
    clase = {"Sano": 0, "Warning": 1, "Crítico": 2}
    peor = max(clase[r["prediccion"]] for r in local_rows)
    estado = ESTADO_POR_CLASE[peor]
    result: dict = {
        "Estado_del_sistema": {
            "Estado_interpretativo": estado,
            "Estado_por_componente": {r["componente"]: r["prediccion"] for r in local_rows},
        },
    }
    alterados = [r for r in local_rows if clase[r["prediccion"]] > 0]
    if not alterados:
        result["Mensaje_operativo"] = [MENSAJE_NORMAL]
    else:
        result["Bloques_principales"] = [r["componente"] for r in alterados]
        variables: list[str] = []
        for r in alterados:
            for sensor in PHYSICS_RULES.get(r["componente"], {}).get("sensors", []):
                if sensor not in variables:
                    variables.append(sensor)
            if r.get("sensor_mas_relevante") and r["sensor_mas_relevante"] not in variables:
                variables.append(r["sensor_mas_relevante"])
        result["Variables_clave"] = variables
        por_urgencia = sorted(alterados, key=lambda r: URGENCY_ORDER.get(r["urgencia"], 99))
        result["Acciones_sugeridas"] = [f"{r['componente']}: {r['accion']}" for r in por_urgencia]
    result["Detalle_componentes"] = [{k: v for k, v in r.items() if k != "cycle_id"} for r in local_rows]
    return result


def explain_cycle(model, input_tensor, predictions, confidences, feature_cols, cycle_id=None,
                  include_shap: bool = False, shap_background=None, include_cam: bool = False):
    """XAI de un ciclo -> ``(xai_result, xai_error)``; nunca lanza: un fallo se devuelve como ``xai_error``.

    Grad-CAM se calcula siempre; SHAP solo con ``include_shap`` (coste alto). Un error aquí no afecta a la
    predicción del ciclo ni a la de los demás ciclos del lote.
    """
    try:
        device = next(model.parameters()).device
        tensor = input_tensor.to(device)
        with XAI_LOCK:
            gradcam = compute_gradcam_all_heads(model, tensor)
            shap_all = None
            if include_shap:
                if shap_background is None:
                    logger.warning("No packaged SHAP background; using the cycle itself as background (degenerate)")
                    background = tensor.detach().cpu()
                else:
                    background = torch.as_tensor(shap_background, dtype=torch.float32)
                seed_xai()
                shap_all = compute_shap_all_heads(model, background, tensor, feature_cols)
        rows = build_local_report(predictions, confidences, gradcam, shap_all, cycle_id=cycle_id, include_cam=include_cam)
        return build_xai_result(rows), None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("XAI failed for cycle %s: %s", cycle_id, exc)
        return None, str(exc)
