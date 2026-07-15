PREFIX = "/models/m47-dnsl-fallas-maquinaria-pasteurizado"

INLINE_PAYLOAD = {
    "mode": "inline",
    "PS1": [100.0] * 60,
    "PS3": [50.0] * 60,
    "EPS1": [500.0] * 60,
    "FS1": [20.0] * 60,
    "TS1": [55.0] * 60,
    "TS2": [50.0] * 60,
    "VS1": [0.5] * 60,
    "Time_Segundos": [round(i * 0.1, 1) for i in range(60)],
    "Cycle_ID": 1,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "m47-dnsl-fallas-maquinaria-pasteurizado"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "m47-dnsl-fallas-maquinaria-pasteurizado"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "m47-dnsl-fallas-maquinaria-pasteurizado"
    assert body["Enfriador_Fouling"] in (0, 1, 2)
    assert body["Valvula_Switch"] in (0, 1, 2)
    assert body["Bomba_Leakage"] in (0, 1, 2)
    assert body["Acumulador_Gas"] in (0, 1, 2)
    assert 0.0 <= body["Confianza_Fouling"] <= 1.0


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/cycle_data.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "m47-dnsl-fallas-maquinaria-pasteurizado"
    assert len(body["predictions"]) == 1
