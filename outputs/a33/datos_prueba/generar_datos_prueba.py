"""Generate ml33 test payloads from the AI team's test split + manifest golden cases (no invented values).

data/predictions/capacity_sensitivity/ is NOT used: those runs come from the previous NEAT engine
(inference_base.csv only matches the deployed MILP output on 39 % of rows).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a33/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "inbox/a33/codigo/data"
LOT_COLS = ["subproduct_type", "season", "generated_volume_tons", "moisture_pct"]

golden = {c["id"]: c for c in yaml.safe_load((ROOT / "inbox/a33/manifest.yaml").read_text())["golden_cases"]}
for cid in ("bloque_001_filas_1_15", "bloque_002_filas_16_30"):
    (OUT / f"predict_inline_{cid}.json").write_text(json.dumps({"mode": "inline", **golden[cid]["input"]}, indent=2))

test = pd.read_csv(DATA / "split/dataset_optimization_cereal_co2_test_raw.csv")
test[LOT_COLS].to_csv(OUT / "predict_batch_test_completo.csv", index=False)
test[LOT_COLS].head(150).to_csv(OUT / "predict_batch_150.csv", index=False)

ref = pd.read_csv(DATA / "predictions/inference_with_constraints.csv")
exp = ref[LOT_COLS].copy()
exp.insert(0, "row", range(1, len(exp) + 1))
exp["esperado_ai_assigned_strategy"] = ref.ai_assigned_strategy
exp["esperado_ai_assignment_source"] = ref.ai_assignment_source
exp["esperado_ai_is_fallback"] = ref.ai_is_fallback
exp.to_csv(OUT / "expected_predictions.csv", index=False)
(OUT / "expected_batch_test_completo.json").write_text(
    json.dumps(golden["batch_full_test_split_distribution"]["expected"] | {"tolerance_pct": 0.5}, indent=2))
print("inline: 2 bloques | batch: 150 y", len(test), "filas | expected:", len(exp))
