"""Genera los datos de prueba de ml37 a partir de la entrega del equipo de IA (sin valores inventados).

- Entradas: tensor a 10 Hz del equipo de IA (data/processed/tensors/X.npy, 600 pasos x 17 columnas;
  se usan los 14 sensores físicos, como en la verificación) con t = paso / 10 s.
- Esperados: golden_cases de inbox/a37/manifest.yaml (predicción, risk y alertas), probabilidades de
  data/predictions/no_virtual_v1__rand_scool/predictions_test.parquet (de donde salen esos golden
  cases), etiqueta real de y.npy y top-3 de Integrated Gradients de inbox/a39/manifest.yaml.
- Train: los primeros 100 ciclos de train_idx (splits.json de la variante servida).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
OUT = ROOT / "outputs/a37/datos_prueba"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "inbox/a37/codigo/data"
RUN = "no_virtual_v1__rand_scool"
SENSORS = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2",
           "TS1", "TS2", "TS3", "TS4", "VS1"]  # slots 0-13 de X.npy, mismo orden que el plugin
HEADS = ["cooler", "valve", "pump", "accumulator"]
CLASSES = {"cooler": (3, 20, 100), "valve": (73, 80, 90, 100), "pump": (0, 1, 2),
           "accumulator": (90, 100, 115, 130)}
N_TRAIN_CYCLES = 100

x_all = np.load(DATA / "processed/tensors/X.npy", mmap_mode="r")  # (2205, 600, 17) float32
y_all = np.load(DATA / "processed/tensors/y.npy")  # (2205, 5): cooler, valve, pump, accumulator, stable
splits = json.loads((DATA / f"splits/{RUN}/splits.json").read_text())
golden = yaml.safe_load((ROOT / "inbox/a37/manifest.yaml").read_text())["golden_cases"]["cases"]
golden_ig = {c["cycle_id"]: c["ig_top3_por_head"] for c in
             yaml.safe_load((ROOT / "inbox/a39/manifest.yaml").read_text())["golden_cases"]["cases"]}
pred_test = pd.read_parquet(DATA / f"predictions/{RUN}/predictions_test.parquet").set_index("cycle_id")


def cycle_frame(cycle_id: int) -> pd.DataFrame:
    """Un ciclo en formato long: cycle_id, t (s), 14 sensores (float32, como en X.npy)."""
    df = pd.DataFrame(np.asarray(x_all[cycle_id - 1, :, :len(SENSORS)]), columns=SENSORS)
    df.insert(0, "t", np.round(np.arange(len(df)) * 0.1, 1))
    df.insert(0, "cycle_id", cycle_id)
    return df


def readings(df: pd.DataFrame) -> list[dict]:
    # str(np.float32) = repr más corto del float32: al leerlo el plugin recupera el mismo valor
    return [{"t": float(r.t), **{s: float(str(np.float32(getattr(r, s)))) for s in SENSORS}}
            for r in df.itertuples(index=False)]


ids = [c["cycle_id"] for c in golden]
test_idx = set(splits["test_idx"])
assert all(cid - 1 in test_idx for cid in ids), "golden case fuera del split de test"

frames = {cid: cycle_frame(cid) for cid in ids}
pd.concat(frames.values(), ignore_index=True).to_csv(OUT / "predict_batch_golden.csv", index=False)
for cid, df in frames.items():
    body = {"mode": "inline", "include_xai": True, "readings": readings(df)}
    (OUT / f"predict_inline_cycle_{cid:04d}.json").write_text(json.dumps(body))
# SHAP (opt-in, ≈65 s en CPU): mismo ciclo 33 con include_shap; no hay golden por ciclo de SHAP
body = {"mode": "inline", "include_xai": True, "include_shap": True, "readings": readings(frames[ids[0]])}
(OUT / f"predict_inline_shap_cycle_{ids[0]:04d}.json").write_text(json.dumps(body))

rows = []
for case in golden:
    cid, par = case["cycle_id"], pred_test.loc[case["cycle_id"]]
    row = {"case_id": f"cycle_{cid:04d}", "cycle_id": cid}
    row.update({f"real_{h}": int(y_all[cid - 1, k]) for k, h in enumerate(HEADS)})
    assert all(row[f"real_{h}"] == case["esperado"][h] for h in HEADS), f"etiqueta real {cid}"
    for h in HEADS:
        assert int(par[f"pred_{h}"]) == case["predicho"][h], f"predicho {cid}/{h}"
        assert abs(float(par[f"risk_{h}"]) - case["risk"][h]) < 1e-4, f"risk {cid}/{h}"
        row[f"pred_{h}"] = case["predicho"][h]
        row[f"risk_{h}"] = case["risk"][h]
        row.update({f"p_{h}_{c}": round(float(par[f"p_{h}_{c}"]), 6) for c in CLASSES[h]})
    row["heads_acertadas"] = sum(row[f"real_{h}"] == row[f"pred_{h}"] for h in HEADS)
    row["alertas"] = "; ".join(f"{a['component']}:{a['source']}:{a['severity']}" for a in case["alertas"])
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "expected_predictions.csv", index=False)
(OUT / "expected_ig_top3.json").write_text(json.dumps(
    {f"cycle_{cid:04d}": golden_ig[cid] for cid in ids}, indent=2, ensure_ascii=False))

# train: primeros N ciclos de train_idx (índice = cycle_id - 1) con sus 4 targets + stable
train_ids = [i + 1 for i in splits["train_idx"][:N_TRAIN_CYCLES]]
train = pd.concat([cycle_frame(cid) for cid in train_ids], ignore_index=True)
for k, name in enumerate(["y_cooler", "y_valve", "y_pump", "y_accumulator", "y_stable"]):
    train[name] = train.cycle_id.map({cid: int(y_all[cid - 1, k]) for cid in train_ids})
train.to_csv(OUT / f"train_{N_TRAIN_CYCLES}ciclos.csv", index=False)
print(f"golden={len(ids)} (+1 SHAP) | train={train.cycle_id.nunique()} ciclos ({len(train)} filas), "
      f"cooler={train.groupby('cycle_id').y_cooler.first().value_counts().to_dict()}")
