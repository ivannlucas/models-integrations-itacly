"""Generate ml36 test payloads from the AI team's splits/predictions + manifest golden cases (no invented values)."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a36/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "inbox/a36/codigo/data"
BASE = ["T_serv", "F_milk", "T_in", "Fat_perc", "Viscosity", "Regeneration_perc", "Delta_P", "t_ciclo"]
CONTEXT = ["F_milk", "T_in", "Fat_perc", "Viscosity", "t_ciclo"]
CONTROLS = ["T_serv", "Delta_P", "Regeneration_perc"]
N_OPT_BATCH = 20

golden = yaml.safe_load((ROOT / "inbox/a36/manifest.yaml").read_text())["golden_cases"]
for case in golden:
    if case["mode"] == "predict":
        body = {"mode": "inline", **case["input"]}
    else:  # optimize: original row seed = 42 + (val_row_index + 1)
        body = {"mode": "inline", "model_key": "optimize", "decision_mode": "hybrid",
                "seed": 43 + case["val_row_index"], **case["input"], **case["input_current_setpoints"]}
    (OUT / f"predict_inline_{case['id']}.json").write_text(json.dumps(body, indent=2))

# ── MLP digital twin, whole test split ────────────────────────────────────────
test = pd.read_csv(DATA / "splits/test_raw.csv")
test[BASE].to_csv(OUT / "predict_batch_mlp_test.csv", index=False)
preds = pd.read_csv(DATA / "predictions/final_test_predictions.csv")
pd.concat([test[BASE], preds.rename(columns={"T_out_pred": "esperado_T_out_pred",
                                             "CO2_emissions_pred": "esperado_CO2_emissions_pred"})],
          axis=1).to_csv(OUT / "expected_mlp_test.csv", index=False)

# ── GA optimize, first validation rows (hybrid, original seeds) ───────────────
val = pd.read_csv(DATA / "splits/val_raw.csv").head(N_OPT_BATCH)
opt = val[CONTEXT + CONTROLS].copy()
opt["seed"] = 43 + opt.index
opt["decision_mode"] = "hybrid"
opt.to_csv(OUT / "predict_batch_optimize_val20.csv", index=False)
dec = pd.read_csv(DATA / "predictions/realtime_decisions.csv").head(N_OPT_BATCH)
dec.drop(columns=["policy_source"]).to_csv(OUT / "expected_optimize_val20.csv", index=False)

# ── train: 40 complete cycles of the master dataset ───────────────────────────
master = pd.read_csv(DATA / "processed/df_pasteurizacion_master.csv")
cycles = sorted(master.cycle_id.unique())[:40]
master[master.cycle_id.isin(cycles)].to_csv(OUT / "train_40ciclos.csv", index=False)
print(f"inline={len(golden)} mlp_test={len(test)} opt_batch={len(opt)} train_rows={master.cycle_id.isin(cycles).sum()}")
