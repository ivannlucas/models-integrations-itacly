"""Generate ml26 test payloads from the AI team's real dataset (no invented values)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a26/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
CODE = ROOT / "inbox/a26/codigo"

seq = pd.read_csv(CODE / "data/raw/sequential.csv.gz")
manifest = yaml.safe_load((ROOT / "inbox/a26/manifest.yaml").read_text())
golden = manifest["golden_cases"]

LOT_COLS = ["lot_id", "wine_type", "volume_l", "ambient_temp_c"]
READ_COLS = ["lot_id", "timestamp", "timestamp_index", "stage", "stage_progress", "temperature_c",
             "dissolved_oxygen_mg_l", "density_g_ml", "co2_g_l", "ph_lab", "free_sulfite_lab",
             "total_sulfite_lab", "dose_mg_l"]


def winery_readings(lot_id: str, end_idx: int) -> pd.DataFrame:
    """Manifest 'modo bodega' transform: lab only where lab_sample==1, dose only if > 0."""
    df = seq[(seq.lot_id == lot_id) & (~seq.target_only.astype(bool))
             & (seq.timestamp_index <= end_idx)].sort_values("timestamp_index").copy()
    no_lab = df.lab_sample != 1
    df.loc[no_lab, ["ph_lab", "free_sulfite_lab", "total_sulfite_lab"]] = np.nan
    df["dose_mg_l"] = df.actual_dose_mg_l.where(df.actual_dose_mg_l > 0)
    return df


# ── predict inline: one JSON per golden case ─────────────────────────────────
expected_rows = []
batch_frames = []
for case in golden:
    inp, exp = case["input"], case["expected"]
    df = winery_readings(inp["lot_id"], inp["window_end_timestamp_index"])
    assert len(df) == inp["n_readings"], (case["id"], len(df), inp["n_readings"])
    lot = {"lot_id": inp["lot_id"], **inp["lot"]}
    readings = [{k: (None if pd.isna(v) else v) for k, v in r.items()}
                for r in df[READ_COLS].to_dict("records")]
    for r in readings:
        r["timestamp_index"] = int(r["timestamp_index"])
        r = {k: v for k, v in r.items() if v is not None}
    readings = [{k: v for k, v in r.items() if v is not None} for r in readings]
    body = {"mode": "inline", "lot": lot, "readings": readings}
    (OUT / f"predict_inline_{case['id']}.json").write_text(json.dumps(body, indent=2))
    expected_rows.append({
        "case_id": case["id"], "lot_id": inp["lot_id"],
        "timestamp_index": inp["window_end_timestamp_index"], "n_readings": inp["n_readings"],
        "expected_future_free_sulfite_72h": exp["future_free_sulfite_72h"],
        "expected_underprotection_risk_72h": exp["underprotection_risk_72h"],
        "y_true_future_free_sulfite_72h": exp["y_true_future_free_sulfite_72h"],
        "y_true_underprotection_risk_72h": exp["y_true_underprotection_risk_72h"],
        "risk_band_real": exp["risk_band_real"],
    })
    out = df[READ_COLS].copy()
    for k in ("wine_type", "volume_l", "ambient_temp_c"):
        out[k] = lot[k]
    batch_frames.append(out[LOT_COLS + READ_COLS[1:]])

pd.DataFrame(expected_rows).to_csv(OUT / "expected_predictions.csv", index=False)
pd.concat(batch_frames).to_csv(OUT / "predict_batch.csv", index=False)

# ── train: 12 full lots from the AI team's TRAIN split (4 per wine type) ─────
train_lots = pd.read_csv(CODE / "data/splits/train.csv.gz", usecols=["lot_id"]).lot_id.unique()
lot_types = seq.drop_duplicates("lot_id").set_index("lot_id").wine_type
picked = []
for wt in ("red", "rose", "white"):
    picked += sorted(l for l in train_lots if lot_types[l] == wt)[:4]
TRAIN_COLS = ["lot_id", "timestamp", "timestamp_index", "target_only", "wine_type", "volume_l",
              "ambient_temp_c", "stage", "stage_progress", "temperature_c", "dissolved_oxygen_mg_l",
              "density_g_ml", "co2_g_l", "free_sulfite_mg_l", "ph_lab", "free_sulfite_lab",
              "total_sulfite_lab", "actual_dose_mg_l", "hours_since_last_lab", "lab_sample"]
train = seq[seq.lot_id.isin(picked)].sort_values(["lot_id", "timestamp_index"])[TRAIN_COLS]
train.to_csv(OUT / "train.csv", index=False)
print("golden:", len(golden), "batch rows:", sum(len(f) for f in batch_frames),
      "train lots:", picked, "train rows:", len(train))
