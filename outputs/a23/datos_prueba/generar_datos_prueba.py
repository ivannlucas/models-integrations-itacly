"""Generate ml23 test payloads from the AI team's dataset/predictions + manifest golden cases (no invented values)."""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a23/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "inbox/a23/codigo/data"
FEATURES = json.loads((ROOT / "artifacts/ml23_lactic_market_price_forecast/manifest.json").read_text())["feature_cols"]

# ── batch + train: the AI team's dataset_forecast_ready.csv as delivered ──────────────────────
shutil.copy(DATA / "processed/dataset_forecast_ready.csv", OUT / "dataset_forecast_ready.csv")
ref = pd.read_csv(DATA / "predictions/inference_predictions_gru.csv", encoding="utf-8-sig")
ref.rename(columns={"y_pred": "esperado_y_pred"}).to_csv(OUT / "expected_batch.csv", index=False)

# ── golden cases (expected = original predictor on the full 6-month window) ───────────────────
golden = yaml.safe_load((ROOT / "inbox/a23/manifest.yaml").read_text())["golden_cases"]
pd.DataFrame([{"case_id": c["id"], **c["input"], "esperado_y_pred": c["expected"]["y_pred"],
               "atol": c["tolerance"]["atol"]} for c in golden]).to_csv(OUT / "expected_golden.csv", index=False)

# ── inline: the feature row of each golden case (approximate by design: the plugin tiles it 6x) ─
df = pd.read_csv(DATA / "processed/dataset_forecast_ready.csv", encoding="utf-8-sig")
df["current_price"] = df["target_precio_medio"]  # same derivation as predictor.py::_prepare_input_df
for c in golden:
    i = c["input"]
    row = df[(df.fecha == i["fecha"]) & (df.producto == i["producto"]) & (df.canal == i["canal"])].iloc[0]
    body = {"mode": "inline", **{k: float(row[k]) for k in FEATURES}}
    (OUT / f"predict_inline_{c['id'].removeprefix('caso_')}.json").write_text(json.dumps(body, indent=2))
print(f"batch/train={len(df)} filas | expected_batch={len(ref)} ventanas | golden={len(golden)}")
