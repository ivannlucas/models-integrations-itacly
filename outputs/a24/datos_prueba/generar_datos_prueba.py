"""Generate ml24 test payloads from the AI team's MAPA series/predictions + manifest golden cases (no invented values)."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a24/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "inbox/a24/codigo/data"
PRODUCTS = ["TOTAL LECHE LIQUIDA", "LECHE ESTERILIZADA", "LECHE ENTERA", "LECHE DESNATADA", "LECHE SEMIDESNATAD"]

# ── batch + train: the delivered series, only the 5 products the model uses (337 MB -> ~14 MB) ─
series = pd.read_csv(DATA / "processed/series_consumo/series_consumo_lacteos_ccaa.csv", encoding="utf-8-sig")
series = series[series.producto.isin(PRODUCTS)].drop(columns=["archivo_origen"])
series.to_csv(OUT / "serie_5_productos.csv", index=False)
smoke = pd.read_csv(DATA / "predictions/spatial_price_model/inference_smoke.csv", encoding="utf-8-sig")
smoke[["ccaa", "horizon_months", "feature_period", "period", "y_pred", "y_baseline_calibrated"]].rename(
    columns={"y_pred": "esperado_y_pred", "y_baseline_calibrated": "esperado_y_baseline_calibrated"}
).to_csv(OUT / "expected_batch_ultimo_mes.csv", index=False)

# ── inline: golden cases (backtesting with as_of_period; reference = original predictor) ──────
golden = yaml.safe_load((ROOT / "inbox/a24/manifest.yaml").read_text())["golden_cases"]
for c in golden:
    (OUT / f"predict_inline_{c['id']}.json").write_text(json.dumps(c["input"], indent=2))
# the two rejection cases (memoria §8.4) must answer 422
rejections = [c for c in golden if "http_status" in c["expected"]]
pd.DataFrame([{"case_id": c["id"], **c["expected"]} for c in rejections]).to_csv(OUT / "expected_rechazos.csv", index=False)
golden = [c for c in golden if c not in rejections]
pd.DataFrame([{"case_id": c["id"], "ccaa": c["input"]["ccaa"][0], "horizon": c["input"]["horizons"][0],
               "as_of_period": c["input"]["as_of_period"],
               "esperado_y_pred_original": c["reference_original_predictor"]["y_pred"],
               "y_true": c["expected"]["y_true"], "target_period": c["expected"]["target_period"],
               "tolerance_rel": c["tolerance_rel"]} for c in golden]).to_csv(OUT / "expected_golden.csv", index=False)
print(f"serie={len(series)} filas | smoke={len(smoke)} | golden={len(golden)} + rechazos={len(rejections)}")
