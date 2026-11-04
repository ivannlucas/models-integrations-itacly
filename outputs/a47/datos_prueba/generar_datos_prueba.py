"""Generate ml47 test payloads from the AI team's splits + manifest golden cases (no invented values)."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a47/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
CODE = ROOT / "inbox/a47/codigo"
SENSORS = ["PS1", "PS3", "EPS1", "FS1", "TS1", "TS2", "VS1"]
TARGETS = ["Target_Fouling", "Target_Valvula", "Target_Bomba", "Target_Acumulador"]
COLS = ["Cycle_ID", "Time_Segundos"] + SENSORS
N_TRAIN_CYCLES = 100
TS1_MEAN_TRAIN = 45.31107344717894  # artifacts/<model>/ts1_mean_train.pkl

golden = yaml.safe_load((ROOT / "inbox/a47/manifest.yaml").read_text())["golden_cases"]
ids = [c["input"]["Cycle_ID"] for c in golden]

# test_split.csv already carries the digital-twin offset (TS1 ~65-68 °C): plant-domain data,
# so the plugin is used with its default APPLY_DIGITAL_TWIN=false.
test = pd.read_csv(CODE / "data/splits/test_split.csv", usecols=COLS + TARGETS)
gold = test[test.Cycle_ID.isin(ids)]
gold[COLS].to_csv(OUT / "predict_batch_golden.csv", index=False)
for case in golden:
    cyc = gold[gold.Cycle_ID == case["input"]["Cycle_ID"]]
    body = {"mode": "inline", "Cycle_ID": int(cyc.Cycle_ID.iloc[0]),
            "Time_Segundos": cyc.Time_Segundos.tolist(), **{s: cyc[s].tolist() for s in SENSORS}}
    (OUT / f"predict_inline_{case['id']}.json").write_text(json.dumps(body))
pd.DataFrame([{"case_id": c["id"], "Cycle_ID": c["input"]["Cycle_ID"],
               **{f"esperado_{k}": v for k, v in c["expected"].items()},
               "real": "/".join(map(str, c["true_targets"])), "acierto_modelo": c["match"]} for c in golden]
             ).to_csv(OUT / "expected_predictions.csv", index=False)

# train: first N cycles of the AI team's train split, raw 10 Hz + the same digital-twin offset
# (fine_tune does not apply it; it expects plant-domain temperatures like inference).
train_ids = json.loads((CODE / "data/splits/cycle_splits.json").read_text())["train_cycles"][:N_TRAIN_CYCLES]
raw = pd.read_csv(CODE / "data/processed/hydraulic_10hz_raw.csv", usecols=COLS + TARGETS)
train = raw[raw.Cycle_ID.isin(train_ids)].copy()
train[["TS1", "TS2"]] += 65.0 - TS1_MEAN_TRAIN
train.to_csv(OUT / f"train_{N_TRAIN_CYCLES}ciclos.csv", index=False)
print(f"golden={len(golden)} ciclos ({len(gold)} filas) | train={train.Cycle_ID.nunique()} ciclos ({len(train)} filas)")
