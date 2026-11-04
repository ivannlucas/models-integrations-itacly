"""Generate ml30 test payloads from the AI team's real splits + manifest golden cases (no invented values)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a30/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = ROOT / "inbox/a30/codigo/data/splits"
sys.path.insert(0, str(ROOT))
from app.plugins.ml30_meat_traceability_detection.constants import FEATURE_COLUMNS  # noqa: E402

IDS = ["row_id", "event_uid"]
TARGET = "target_traceability_incident"

golden = pd.read_csv(ROOT / "inbox/a30/golden_cases_test_selected.csv")
golden[IDS + FEATURE_COLUMNS].to_csv(OUT / "predict_batch_golden.csv", index=False)
golden[IDS + [TARGET, "pred_traceability_incident", "pred_score"]].rename(
    columns={"pred_traceability_incident": "esperado_pred_traceability_incident",
             "pred_score": "esperado_pred_score"}).to_csv(OUT / "expected_predictions.csv", index=False)

test = pd.read_csv(SPLITS / "test.csv")
test[IDS + FEATURE_COLUMNS].to_csv(OUT / "predict_batch_test.csv", index=False)
test[IDS + [TARGET]].to_csv(OUT / "test_labels.csv", index=False)

train = pd.read_csv(SPLITS / "train.csv")
train[IDS + FEATURE_COLUMNS + [TARGET]].to_csv(OUT / "train.csv", index=False)
print(f"golden={len(golden)} test={len(test)} train={len(train)} (incidencias train={train[TARGET].mean():.1%})")

# ── expected scores for the whole test split, from the AI team's ORIGINAL predictor ──────────
import os  # noqa: E402

import yaml  # noqa: E402

CODE = ROOT / "inbox/a30/codigo"
os.chdir(CODE)
sys.path.insert(0, str(CODE))
from src.predict.predictor import predict_dataframe  # noqa: E402

y_pred, y_score = predict_dataframe(yaml.safe_load(open("config/config.yaml")), "neuroevolution_mlp", test)
out = test[IDS + [TARGET]].copy()
out["esperado_pred_traceability_incident"], out["esperado_pred_score"] = y_pred, y_score
out["sensor_numerico_vacio"] = test[[c for c in FEATURE_COLUMNS if c.startswith("sensor_") and "delta" not in c]].isna().any(axis=1)
out.to_csv(OUT / "test_labels.csv", index=False)
print("test_labels.csv: rows with empty numeric sensor =", int(out.sensor_numerico_vacio.sum()))
