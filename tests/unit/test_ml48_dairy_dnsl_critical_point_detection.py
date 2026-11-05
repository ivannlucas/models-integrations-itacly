"""ml48 (ml48-dairy-dnsl-critical-point-detection): plugin que implementa modelo-47 con XAI
integrada.

Contrato (mismo patron que 43/44): cada ciclo/fila devuelve su propia XAI como ``xai_result``
(claves del final_report de modelo-43) o ``xai_error``; inline y batch. La correctitud contra el
golden dataset se valida en el skill verification (inbox/a48/manifest.yaml); aqui, wiring y
comportamiento con un modelo sintetico.

Los modulos del plugin que dependen de torch se importan dentro de cada prueba, despues del
``pytest.importorskip("torch")`` de mas abajo.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.plugins.ml48_dairy_dnsl_critical_point_detection.predict_dto import PredictBatchRequest

PREFIX = "/models/ml48-dairy-dnsl-critical-point-detection"
MODEL_ID = "ml48-dairy-dnsl-critical-point-detection"

INLINE_PAYLOAD = {
    "mode": "inline",
    "PS1": [100.0] * 60, "PS3": [50.0] * 60, "EPS1": [500.0] * 60, "FS1": [20.0] * 60,
    "TS1": [55.0] * 60, "TS2": [50.0] * 60, "VS1": [0.5] * 60,
    "Time_Segundos": [round(i * 0.1, 1) for i in range(60)],
    "Cycle_ID": 1,
}


# ── Wiring (FakePlugin) ───────────────────────────────────────────────────────

def test_health(client):
    """/health informa status ok, el id del modelo y loaded=True."""
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok" and body["model"] == MODEL_ID and body["loaded"] is True


def test_stats(client):
    """/stats informa el id del modelo como model_name."""
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == MODEL_ID


def test_predict_inline_returns_xai_result(client):
    """/predict inline devuelve la prediccion con su xai_result (sin la lista xai antigua)."""
    body = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD).json()
    assert body["Enfriador_Fouling"] in (0, 1, 2)
    assert body["xai_error"] is None
    assert body["xai_result"]["Estado_del_sistema"]["Estado_interpretativo"] in (
        "Normal", "Warning", "Crítico"
    )
    assert "xai" not in body, "el contrato antiguo (lista xai) ya no existe"


def test_predict_inline_without_xai(client):
    """Con include_xai=False, xai_result y xai_error son None."""
    body = client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "include_xai": False}).json()
    assert body["xai_result"] is None and body["xai_error"] is None


def test_shap_requires_xai(client):
    """include_shap sin include_xai responde 422 en inline y en batch."""
    resp = client.post(
        f"{PREFIX}/predict",
        json={**INLINE_PAYLOAD, "include_xai": False, "include_shap": True},
    )
    assert resp.status_code == 422
    resp = client.post(
        f"{PREFIX}/predict",
        json={"mode": "batch", "data_path": "x.csv", "include_shap": True, "include_xai": False},
    )
    assert resp.status_code == 422


def test_inline_requires_sensors_or_path(client):
    """Inline sin todos los sensores ni data_path responde 422."""
    payload = {k: v for k, v in INLINE_PAYLOAD.items() if k != "VS1"}
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_batch_has_xai_per_row(client):
    """Batch devuelve xai_result/xai_error en cada fila y ninguna XAI global de lote."""
    body = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cycles.csv"}
    ).json()
    assert body["model_id"] == MODEL_ID and len(body["predictions"]) == 3
    for row in body["predictions"]:
        assert ("xai_result" in row) and ("xai_error" in row)
    assert "xai" not in body, "no hay XAI global a nivel de lote: va por ciclo"
    assert (
        body["predictions"][1]["xai_result"] is None
        and body["predictions"][1]["xai_error"] == "boom"
    )


def test_train(client):
    """/train responde 200 con el modo, las metricas y los tamanos de cada particion."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/t.csv", "mlflow_run_id": "run"})
    assert resp.status_code == 200
    assert {
        "mode", "exact_match", "accuracy", "f1_macro", "recall_macro", "n_train", "n_val", "n_test",
    } <= set(resp.json())


def test_predict_dto_batch_defaults():
    """PredictBatchRequest: XAI activa, SHAP desactivado, n_samples=50 y n_samples>=1."""
    req = PredictBatchRequest(data_path="x.csv")
    assert req.include_xai is True and req.include_shap is False and req.n_samples == 50
    with pytest.raises(ValidationError):
        PredictBatchRequest(data_path="x.csv", n_samples=0)


# ── XAI: lógica (sin artefactos) ──────────────────────────────────────────────

torch = pytest.importorskip("torch")


def _random_model():
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.model_loader import (  # pylint: disable=import-outside-toplevel
        CNN_Pasteurizer,
    )
    torch.manual_seed(0)
    return CNN_Pasteurizer(n_sensors=28, n_classes=3, dropout_prob=0.2).eval()


def _rows(predictions, shap=None):
    """Filas del informe local (formato de build_local_report) para pruebas de build_xai_result."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection import xai  # pylint: disable=import-outside-toplevel
    cams = [{"cam": np.linspace(0, 1, 600)} for _ in range(4)]
    return xai.build_local_report(predictions, [0.9, 0.8, 0.7, 0.6], cams, shap, cycle_id=1)


def test_find_temporal_peaks():
    """find_temporal_peaks agrupa los tramos contiguos por encima del umbral con su media."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.xai import (  # pylint: disable=import-outside-toplevel
        find_temporal_peaks,
    )
    cam = np.array([0.1, 0.8, 0.9, 0.2, 0.75, 0.75])
    assert find_temporal_peaks(cam) == [(1, 2, pytest.approx(0.85)), (4, 5, pytest.approx(0.75))]


def test_risk_scores_and_shap_bonus():
    """compute_risk_scores asigna score y nivel por componente y aplica el bonus de SHAP."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.xai import (  # pylint: disable=import-outside-toplevel
        compute_risk_scores,
    )
    risk = compute_risk_scores([2, 1, 0, 0], [0.9, 0.8, 1.0, 1.0])
    assert risk["Fouling"] == {"score": 0.9, "level": "CRÍTICO"}
    assert risk["Válvula"] == {"score": 0.4, "level": "NORMAL"}
    boosted = compute_risk_scores(
        [1, 0, 0, 0], [0.9, 1, 1, 1], {"Fouling": {"TS1": 0.3, "TS2": 0.2, "FS1": 0.1}}
    )
    assert boosted["Fouling"]["score"] == pytest.approx(0.45 * 1.6)


def test_recommendations_sorted_by_urgency():
    """Las recomendaciones de mantenimiento salen ordenadas por urgencia."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.xai import (  # pylint: disable=import-outside-toplevel
        generate_maintenance_recommendations,
    )
    recs = generate_maintenance_recommendations([0, 2, 1, 0], [1.0, 0.9, 0.8, 1.0])
    assert [r["urgencia"] for r in recs] == ["Alta", "Media", "Ninguna", "Ninguna"]


def test_xai_result_normal_has_mensaje_operativo_and_no_blocks():
    """En estado Normal hay Mensaje_operativo y no hay bloques, variables ni acciones."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.xai import (  # pylint: disable=import-outside-toplevel
        build_xai_result,
    )
    r = build_xai_result(_rows([0, 0, 0, 0]))
    assert r["Estado_del_sistema"]["Estado_interpretativo"] == "Normal"
    assert isinstance(r["Mensaje_operativo"], list) and r["Mensaje_operativo"]
    # igual que modelo-43: con estado Normal no existen Bloques_principales / Variables_clave /
    # Acciones_sugeridas
    for absent in ("Bloques_principales", "Variables_clave", "Acciones_sugeridas"):
        assert absent not in r
    assert len(r["Detalle_componentes"]) == 4


def test_xai_result_worst_state_blocks_variables_and_actions():
    """El peor estado fija bloques, variables clave (sin duplicados) y acciones por urgencia."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.xai import (  # pylint: disable=import-outside-toplevel
        build_xai_result,
    )
    r = build_xai_result(_rows([2, 0, 1, 0]))  # Fouling Crítico, Bomba Warning
    assert r["Estado_del_sistema"]["Estado_interpretativo"] == "Crítico"
    assert r["Estado_del_sistema"]["Estado_por_componente"] == {
        "Fouling": "Crítico", "Válvula": "Sano", "Bomba": "Warning", "Acumulador": "Sano"}
    assert r["Bloques_principales"] == ["Fouling", "Bomba"]
    # sensores de la regla física de Fouling (TS1, TS2, FS1) y de Bomba (PS1, FS1, EPS1), sin
    # duplicados
    assert r["Variables_clave"] == ["TS1", "TS2", "FS1", "PS1", "EPS1"]
    # urgencia Alta primero
    assert r["Acciones_sugeridas"][0].startswith("Fouling: Detener producción")
    assert r["Acciones_sugeridas"][1].startswith("Bomba:")
    assert "Mensaje_operativo" not in r


def test_xai_result_adds_shap_sensor_to_variables():
    """Sin regla fisica propia, el sensor mas relevante segun SHAP pasa a Variables_clave."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.xai import (  # pylint: disable=import-outside-toplevel
        build_xai_result,
    )
    shap = {"Válvula": {"PS3": 0.7, "FS1": 0.3}, "Fouling": {"TS2": 1.0},
            "Bomba": {"VS1": 1.0}, "Acumulador": {"TS2": 1.0}}
    # solo Válvula alterada: sin regla física propia
    r = build_xai_result(_rows([0, 1, 0, 0], shap))
    assert r["Variables_clave"] == ["PS3"]
    assert r["Detalle_componentes"][1]["sensor_mas_relevante"] == "PS3"


def test_explain_cycle_ok_and_error_never_raises(monkeypatch):
    """explain_cycle devuelve (resultado, None) o (None, error), sin lanzar nunca."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection import xai  # pylint: disable=import-outside-toplevel
    model = _random_model()
    x = torch.randn(1, 28, 600)
    result, error = xai.explain_cycle(
        model, x, [0, 1, 2, 0], [0.9, 0.8, 0.7, 0.6], [f"c{i}" for i in range(28)], cycle_id=7
    )
    assert error is None and result["Estado_del_sistema"]["Estado_interpretativo"] == "Crítico"
    assert len(result["Detalle_componentes"]) == 4

    def boom(*_a, **_k):
        raise RuntimeError("grad-cam roto")
    monkeypatch.setattr(xai, "compute_gradcam_all_heads", boom)
    result, error = xai.explain_cycle(
        model, x, [0, 0, 0, 0], [1, 1, 1, 1], [f"c{i}" for i in range(28)], cycle_id=7
    )
    assert result is None and "grad-cam roto" in error


# ── Plugin real con modelo sintético: inline y batch por ciclo ───────────────

def _synthetic_plugin():
    # pylint: disable=import-outside-toplevel
    from sklearn.preprocessing import StandardScaler

    from app.plugins.ml48_dairy_dnsl_critical_point_detection.constants import SENSOR_COLUMNS
    from app.plugins.ml48_dairy_dnsl_critical_point_detection.plugin import (
        Ml48DairyDnslCriticalPointDetectionPlugin,
    )
    # pylint: enable=import-outside-toplevel
    cols = SENSOR_COLUMNS + [f"{s}_rmean" for s in SENSOR_COLUMNS] \
        + [f"{s}_rstd" for s in SENSOR_COLUMNS] + [f"{s}_lag1" for s in SENSOR_COLUMNS]
    p = Ml48DairyDnslCriticalPointDetectionPlugin()
    p._model = _random_model()
    p._scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(200, 28)))
    p._feature_columns = cols
    p._ts1_mean_train = 45.0
    p._device = torch.device("cpu")
    p._shap_background = None
    return p, pd


def _cycles_csv(tmp_path, n_cycles):
    rng = np.random.default_rng(1)
    frames = []
    for cid in range(1, n_cycles + 1):
        t = np.round(np.arange(0, 60, 0.1), 1)
        frames.append(pd.DataFrame({
            "Cycle_ID": cid, "Time_Segundos": t,
            "PS1": rng.normal(160, 5, 600), "PS3": rng.normal(2, .3, 600),
            "EPS1": rng.normal(2500, 50, 600), "FS1": rng.normal(6, .5, 600),
            "TS1": rng.normal(45, 1, 600), "TS2": rng.normal(50, 1, 600),
            "VS1": rng.normal(.6, .02, 600)}))
    path = tmp_path / "cycles.csv"
    pd.concat(frames).to_csv(path, index=False)
    return str(path)


def test_batch_returns_xai_result_for_every_cycle(tmp_path):
    """Batch devuelve una fila por ciclo, cada una con su prediccion y su xai_result."""
    plugin, _ = _synthetic_plugin()
    resp = plugin.predict_batch(data_path=_cycles_csv(tmp_path, 4))
    # pydantic fields are seen as FieldInfo by pylint
    predictions = resp.predictions
    assert [r["Cycle_ID"] for r in predictions] == [1, 2, 3, 4]  # pylint: disable=not-an-iterable
    for row in predictions:  # pylint: disable=not-an-iterable
        assert row["xai_error"] is None
        assert row["xai_result"]["Estado_del_sistema"]["Estado_interpretativo"] in (
            "Normal", "Warning", "Crítico"
        )
        assert len(row["xai_result"]["Detalle_componentes"]) == 4
        assert all(
            k in row for k in ("Enfriador_Fouling", "Enfriador_Fouling_Texto", "Confianza_Fouling")
        )


# pydantic fields are seen as FieldInfo by pylint (rows = response.predictions)
def test_batch_partial_xai_error_does_not_break_other_cycles(  # pylint: disable=not-an-iterable,unsubscriptable-object
    tmp_path, monkeypatch,
):
    """Un fallo de XAI en un ciclo solo afecta a ese ciclo y conserva su prediccion."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection import xai  # pylint: disable=import-outside-toplevel
    plugin, _ = _synthetic_plugin()
    real = xai.compute_gradcam_all_heads
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("fallo XAI ciclo 2")
        return real(*args, **kwargs)
    monkeypatch.setattr(xai, "compute_gradcam_all_heads", flaky)
    rows = plugin.predict_batch(data_path=_cycles_csv(tmp_path, 3)).predictions
    assert [r["Cycle_ID"] for r in rows] == [1, 2, 3]
    assert rows[0]["xai_result"] is not None and rows[0]["xai_error"] is None
    assert rows[1]["xai_result"] is None and "fallo XAI ciclo 2" in rows[1]["xai_error"]
    assert rows[2]["xai_result"] is not None and rows[2]["xai_error"] is None
    # la predicción del ciclo con XAI fallida se conserva
    assert rows[1]["Enfriador_Fouling"] in (0, 1, 2) and rows[1]["Confianza_Fouling"] > 0


def test_batch_without_xai_leaves_columns_empty(tmp_path):
    """Batch con include_xai=False deja xai_result y xai_error a None en cada fila."""
    plugin, _ = _synthetic_plugin()
    rows = plugin.predict_batch(data_path=_cycles_csv(tmp_path, 2), include_xai=False).predictions
    assert all(
        r["xai_result"] is None and r["xai_error"] is None
        for r in rows  # pylint: disable=not-an-iterable
    )


def test_batch_shap_is_limited_to_n_samples(tmp_path, monkeypatch):
    """SHAP solo se calcula para los n_samples primeros ciclos y nunca si no se pide."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection import xai  # pylint: disable=import-outside-toplevel
    plugin, _ = _synthetic_plugin()
    calls = []

    def fake_shap(model, background, tensor, feature_cols):
        _ = model, background, tensor, feature_cols
        calls.append(1)
        return {h: {"PS3": 0.9, "VS1": 0.1} for h in ("Fouling", "Válvula", "Bomba", "Acumulador")}
    monkeypatch.setattr(xai, "compute_shap_all_heads", fake_shap)
    rows = plugin.predict_batch(
        data_path=_cycles_csv(tmp_path, 4), include_shap=True, n_samples=2
    ).predictions
    assert len(calls) == 2, "SHAP solo se calcula para los primeros n_samples ciclos"
    with_shap = [
        r["xai_result"]["Detalle_componentes"][0]["riesgo_incluye_shap"]
        for r in rows  # pylint: disable=not-an-iterable
    ]
    assert with_shap == [True, True, False, False]
    assert all(
        r["xai_result"] is not None for r in rows  # pylint: disable=not-an-iterable
    ), "todos los ciclos conservan su XAI (Grad-CAM)"

    calls.clear()
    plugin.predict_batch(data_path=_cycles_csv(tmp_path, 3))
    assert not calls, "SHAP nunca se ejecuta si no se solicita"


def test_inline_csv_returns_xai_result_and_optional_shap(tmp_path, monkeypatch):
    """Inline desde CSV devuelve xai_result, con SHAP solo si se pide y nada sin include_xai."""
    from app.plugins.ml48_dairy_dnsl_critical_point_detection import xai  # pylint: disable=import-outside-toplevel
    plugin, _ = _synthetic_plugin()
    csv = _cycles_csv(tmp_path, 1)
    r = plugin.predict_inline(data_path=csv, features={})
    # pydantic fields are seen as FieldInfo by pylint
    assert r.xai_error is None and (
        r.xai_result[  # pylint: disable=unsubscriptable-object
            "Detalle_componentes"][0]["riesgo_incluye_shap"] is False
    )
    monkeypatch.setattr(
        xai, "compute_shap_all_heads",
        lambda *a, **k: {h: {"TS2": 1.0} for h in ("Fouling", "Válvula", "Bomba", "Acumulador")},
    )
    r = plugin.predict_inline(data_path=csv, features={"include_shap": True})
    assert r.xai_result[  # pylint: disable=unsubscriptable-object
        "Detalle_componentes"][0]["riesgo_incluye_shap"] is True
    r = plugin.predict_inline(data_path=csv, features={"include_xai": False})
    assert r.xai_result is None and r.xai_error is None
