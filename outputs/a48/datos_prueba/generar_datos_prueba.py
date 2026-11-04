"""Generate ml48 test payloads from the AI team's data + manifest golden cases (no invented values).

The manifest cites codigo/golden/golden_cycles_10hz.csv, which is not in the delivery: the same cycles
are taken from data/processed/hydraulic_10hz_raw.csv (raw UCI bench, ~35-45 °C), so every predict
request goes with apply_digital_twin=true, as in the verification.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a48/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
CODE = ROOT / "inbox/a48/codigo"
SENSORS = ["PS1", "PS3", "EPS1", "FS1", "TS1", "TS2", "VS1"]
TARGETS = ["Target_Fouling", "Target_Valvula", "Target_Bomba", "Target_Acumulador"]
COLS = ["Cycle_ID", "Time_Segundos"] + SENSORS
N_TRAIN_CYCLES = 100
TS1_MEAN_TRAIN = 45.31107344717894  # artifacts/<model>/ts1_mean_train.pkl

manifest = yaml.safe_load((ROOT / "inbox/a48/manifest.yaml").read_text())
golden = manifest["golden_cases"]
# only the local XAI case (one cycle); the global one runs SHAP over the whole test split (~10 min)
golden_xai = [c for c in manifest["golden_cases_xai"] if "cycle_id" in c["input"]]
raw = pd.read_csv(CODE / "data/processed/hydraulic_10hz_raw.csv", usecols=COLS + TARGETS)


def inline_body(cycle_id: int, **extra) -> dict:
    cyc = raw[raw.Cycle_ID == cycle_id]
    return {"mode": "inline", "apply_digital_twin": True, **extra, "Cycle_ID": int(cycle_id),
            "Time_Segundos": cyc.Time_Segundos.tolist(), **{s: cyc[s].tolist() for s in SENSORS}}


ids = [c["input"]["cycle_id"] for c in golden]
raw[raw.Cycle_ID.isin(ids)][COLS].to_csv(OUT / "predict_batch_golden.csv", index=False)
for case in golden:
    (OUT / f"predict_inline_{case['id']}.json").write_text(json.dumps(inline_body(case["input"]["cycle_id"])))
pd.DataFrame([{"case_id": c["id"], "Cycle_ID": c["input"]["cycle_id"],
               **{f"esperado_{k}": v for k, v in c["expected"].items()}} for c in golden]
             ).to_csv(OUT / "expected_predictions.csv", index=False)

for case in golden_xai:
    body = inline_body(case["input"]["cycle_id"], include_xai=True, include_shap=True)
    (OUT / f"predict_inline_{case['id']}.json").write_text(json.dumps(body))
    (OUT / f"expected_{case['id']}.json").write_text(json.dumps(case["expected"], indent=2, ensure_ascii=False))

# train: first N cycles of the train split + the digital-twin offset (fine_tune expects plant domain)
train_ids = json.loads((CODE / "data/splits/cycle_splits.json").read_text())["train_cycles"][:N_TRAIN_CYCLES]
train = raw[raw.Cycle_ID.isin(train_ids)].copy()
train[["TS1", "TS2"]] += 65.0 - TS1_MEAN_TRAIN
train.to_csv(OUT / f"train_{N_TRAIN_CYCLES}ciclos.csv", index=False)
print(f"golden={len(golden)} + xai={len(golden_xai)} | train={train.Cycle_ID.nunique()} ciclos ({len(train)} filas)")
