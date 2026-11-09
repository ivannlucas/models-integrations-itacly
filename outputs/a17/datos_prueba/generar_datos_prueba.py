"""Generate ml17 test payloads from the AI team's official_v1_4 splits + manifest golden cases (no invented values)."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a17/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
CODE = ROOT / "inbox/a17/codigo/a17-retech-lote2-market-price-analysis-carnico"
TARGET = "target_price_pigmeat_class_e_es"
RAW = ["date", TARGET, "eurostat_pigmeat_slaughter_tonnes_es", "eurostat_pigmeat_slaughter_tonnes_eu",
       "cereal_feed_barley_price_monthly", "cereal_feed_maize_price_monthly", "mapa_porcino_otras_razas_price_monthly"]

# ── inline: 18 golden cases (expected = real price observed at t+1, not the model output) ──
golden = yaml.safe_load((ROOT / "inbox/a17/manifest.yaml").read_text())["golden_cases"]
for case in golden:
    (OUT / f"predict_inline_{case['id']}.json").write_text(json.dumps({"mode": "inline", **case["input"]}, indent=2))

# ── inline: the AI team's own h1 inference example (exact expected model output) ──
example = pd.read_csv(CODE / "data/examples/cu05_inference_example.csv").iloc[0].to_dict()
h1 = pd.read_csv(next((CODE / "data/predictions").glob("official_v1_4_ridge_h1_*.csv"))).iloc[0]
assert h1.base_date == example["date"]
(OUT / "predict_inline_ejemplo_2025_12.json").write_text(json.dumps({"mode": "inline", **example}, indent=2))

# ── batch: the 31 out-of-sample origins of the official benchmark (t = 2023-05 … 2025-11) ──
split = CODE / "data/splits/official_v1_4"
train = pd.read_csv(split / "train.csv")
test = pd.read_csv(split / "test.csv")
series = pd.concat([train, test], ignore_index=True)
origins = series.iloc[len(train) - 1:-1].reset_index(drop=True)
origins[RAW].to_csv(OUT / "predict_batch_test.csv", index=False)
exp = pd.DataFrame({"date": origins.date, "forecast_date": test.date, "y_real_t_mas_1": test[TARGET]})
exp.to_csv(OUT / "expected_test.csv", index=False)
(OUT / "expected_ejemplo_2025_12.json").write_text(json.dumps(
    {"base_date": h1.base_date, "forecast_date": h1.forecast_date, "esperado_y_pred": float(h1.y_pred)}, indent=2))

# ── train: official train split (121 months, with month_sin/month_cos) ──
train.to_csv(OUT / "train.csv", index=False)
print(f"inline={len(golden)}+1 batch={len(origins)} train={len(train)} filas")
