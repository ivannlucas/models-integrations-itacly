"""Verification of ml38 against the REAL server (verification skill, parts A-endpoints and B).

Usage: python -I verify_ml38.py <repo_root> <base_url> <run_ok> <run_empty> <out_json>
"""
import json
import shutil
import sys
import time

import httpx
import numpy as np
import pandas as pd
import yaml

REPO, BASE, RUN_OK, RUN_EMPTY, OUT = sys.argv[1:6]
MID = "ml38-wine-machinery-fault-detection"
URL = f"{BASE}/models/{MID}"
COD = f"{REPO}/inbox/a38/codigo"
TOL = 0.05
client = httpx.Client(timeout=3600)
res: dict = {"endpoints": {}, "golden": [], "batch": {}, "raw": {}, "train": {}, "user_model": {}}


def call(name, method, path, **kw):
    t0 = time.perf_counter()
    r = client.request(method, f"{URL}{path}", **kw)
    dt = time.perf_counter() - t0
    try:
        body = r.json()
    except ValueError:
        body = r.text
    res["endpoints"][name] = {"status": r.status_code, "seconds": round(dt, 2),
                              "body_excerpt": json.dumps(body, ensure_ascii=False)[:600]}
    print(f"{name:45s} {r.status_code} {dt:7.2f}s")
    return r.status_code, body


# ── A: endpoints ────────────────────────────────────────────────────────────────────────────
call("health", "GET", "/health")
_, stats = call("stats", "GET", "/stats")
res["stats_metrics"] = stats.get("metrics") if isinstance(stats, dict) else None

manifest = yaml.safe_load(open(f"{REPO}/inbox/a38/manifest.yaml", encoding="utf-8"))
cases = manifest["golden_cases"]

# ── B: golden cases (inline) ───────────────────────────────────────────────────────────────
for case in cases:
    st, body = call(f"golden_{case['id']}", "POST", "/predict",
                    json={"mode": "inline", **case["input"]})
    exp = case["expected"]
    row = {"id": case["id"], "status": st, "expected": exp, "trace": case.get("trace")}
    if st == 200:
        got = {k: body[k] for k in ("family", "severity", "severity_label", "severity_confidence")}
        row["obtained"] = got
        if exp["severity_confidence"] is None:
            diff = None if got["severity_confidence"] is None else float("inf")
        else:
            diff = abs(got["severity_confidence"] - exp["severity_confidence"]) \
                if got["severity_confidence"] is not None else float("inf")
        row["diff_confidence"] = diff
        row["ok"] = (got["family"] == exp["family"] and got["severity"] == exp["severity"]
                     and got["severity_label"] == exp["severity_label"]
                     and (diff is None or diff <= TOL))
    else:
        row["ok"] = False
    res["golden"].append(row)

# ── extra: batch parity on the 877 rows vs predictions.csv ─────────────────────────────────
df = pd.read_parquet(f"{COD}/data/processed/feature_dataset.parquet")
ref = pd.read_csv(f"{COD}/data/predictions/predictions.csv")
df.to_csv("/tmp/ml38_batch_full.csv", index=False)
df.to_parquet("/tmp/ml38_batch_full.parquet")
for name, path in (("batch_csv", "/tmp/ml38_batch_full.csv"),
                   ("batch_parquet", "/tmp/ml38_batch_full.parquet")):
    st, body = call(name, "POST", "/predict", json={"mode": "batch", "data_path": path})
    if st != 200:
        res["batch"][name] = {"status": st}
        continue
    preds = pd.DataFrame(body["predictions"]).sort_values("row")
    fam_eq = (preds["family"].values == ref["pred_family"].values)
    sev_got = preds["severity"].astype("float").values
    sev_ref = ref["pred_severity"].astype("float").values
    sev_eq = (np.isnan(sev_got) & np.isnan(sev_ref)) | (sev_got == sev_ref)
    cg, cr = preds["severity_confidence"].astype(float).values, ref["severity_confidence"].values
    m = ~np.isnan(cr)
    res["batch"][name] = {
        "status": st, "rows": int(len(preds)), "errors": int(preds.get("error", pd.Series()).notna().sum()),
        "family_match": int(fam_eq.sum()), "severity_match": int(sev_eq.sum()),
        "confidence_nan_pattern_match": bool((np.isnan(cg) == np.isnan(cr)).all()),
        "confidence_max_abs_diff": float(np.nanmax(np.abs(cg[m] - cr[m]))),
        "in_sample_family_accuracy_vs_true": float((preds["family"].values == ref["true_family"].values).mean()),
    }

bad = df.drop(columns=["vib_band_low"])
bad.to_csv("/tmp/ml38_batch_bad.csv", index=False)
call("batch_missing_column_422", "POST", "/predict",
     json={"mode": "batch", "data_path": "/tmp/ml38_batch_bad.csv"})
nan_df = df.head(3).copy()
nan_df.loc[1, "vib_rms"] = np.nan
nan_df.to_csv("/tmp/ml38_batch_nan.csv", index=False)
st, body = call("batch_row_with_nan", "POST", "/predict",
                json={"mode": "batch", "data_path": "/tmp/ml38_batch_nan.csv"})
res["batch"]["nan_row"] = body["predictions"] if st == 200 else st

# ── raw mode (reference check, not golden) ─────────────────────────────────────────────────
for f in ("vibration_example.csv", "current_example.csv"):
    shutil.copy(f"{COD}/data/examples/raw/{f}", f"/tmp/ml38_{f}")
raw = {"mode": "raw", "vibration_csv": "/tmp/ml38_vibration_example.csv",
       "current_csv": "/tmp/ml38_current_example.csv", "motor": "Motor-4", "speed": 70,
       "window_index": 0}
exp_raw = json.load(open(f"{COD}/data/predictions/raw_example_prediction.json"))
st, body = call("raw_example", "POST", "/predict", json=raw)
res["raw"]["example"] = {"status": st, "expected": exp_raw,
                         "obtained": {k: body.get(k) for k in ("family", "severity", "severity_label",
                                                               "severity_confidence", "input_warnings")}
                         if st == 200 else body}
if st == 200:
    res["raw"]["example"]["confidence_abs_diff"] = abs(body["severity_confidence"] - exp_raw["confidence"])
    res["raw"]["example"]["match"] = (body["family"] == exp_raw["family"]
                                      and body["severity"] == exp_raw["severity"])
call("raw_undocumented_config_422", "POST", "/predict", json={**raw, "speed": 50})
call("raw_window_index_out_of_range_422", "POST", "/predict", json={**raw, "window_index": 99})
call("raw_unknown_motor_schema_422", "POST", "/predict", json={**raw, "motor": "Motor-1"})

# ── /train ─────────────────────────────────────────────────────────────────────────────────
call("train_without_run_422", "POST", "/train", json={"data_path": "/tmp/ml38_batch_full.csv"})
call("train_empty_run_422", "POST", "/train",
     json={"data_path": "/tmp/ml38_batch_full.csv", "mlflow_run_id": "  "})
call("train_missing_columns_400", "POST", "/train",
     json={"data_path": "/tmp/ml38_batch_bad.csv", "mlflow_run_id": RUN_OK})
st, body = call("train_full_dataset", "POST", "/train",
                json={"data_path": "/tmp/ml38_batch_full.csv", "mlflow_run_id": RUN_OK})
res["train"] = {"status": st, "body": body}

# ── user model from MLflow ──────────────────────────────────────────────────────────────────
c1 = cases[0]
st, body = call("predict_with_user_run", "POST", "/predict",
                json={"mode": "inline", **c1["input"], "mlflow_run_id": RUN_OK})
res["user_model"]["predict_user_run"] = {"status": st, "body": body}
call("predict_with_empty_run_422", "POST", "/predict",
     json={"mode": "inline", **c1["input"], "mlflow_run_id": RUN_EMPTY})
st, body = call("stats_with_user_run", "GET", "/stats", params={"mlflow_run_id": RUN_OK})
res["user_model"]["stats_mlflow_block"] = body.get("metrics", {}).get("mlflow") if st == 200 else body
st, body = call("predict_base_after_train", "POST", "/predict", json={"mode": "inline", **c1["input"]})
res["user_model"]["base_unchanged_after_train"] = (
    st == 200 and body["family"] == c1["expected"]["family"]
    and abs(body["severity_confidence"] - c1["expected"]["severity_confidence"]) < 1e-6
)

json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False, default=str)
ok = sum(r["ok"] for r in res["golden"])
print(f"GOLDEN {ok}/{len(res['golden'])} dentro de tolerancia")
print("BATCH", json.dumps(res["batch"], default=str)[:800])
print("RAW", json.dumps(res["raw"], ensure_ascii=False)[:700])
print("TRAIN", json.dumps(res["train"], ensure_ascii=False)[:1500])
print("USER", json.dumps(res["user_model"], ensure_ascii=False, default=str)[:900])
