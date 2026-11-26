"""Genera los datos de prueba de ml38-wine-machinery-fault-detection (a38, DNLS-VINO).

Todo sale de la entrega del equipo de IA y del manifest, sin valores inventados:

* inputs: data/processed/feature_dataset.parquet (las 24 características de las 877 ventanas,
  calculadas por el equipo desde el dataset NLN-EMP con vibración ch1 + corriente ch1);
* esperados: golden_cases de inbox/a38/manifest.yaml (= data/predictions/predictions.csv, salida
  del modelo final entregado). El script comprueba que los inputs del manifest son exactamente las
  filas del parquet y que los esperados coinciden con predictions.csv; si no, se para.

Uso: .venv/bin/python -I outputs/a38/datos_prueba/generar_datos_prueba.py "$PWD"
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

FEATURES = [
    "vib_mean", "vib_std", "vib_max", "vib_min", "vib_rms", "vib_kurtosis", "vib_skewness",
    "vib_spectral_energy", "vib_spectral_centroid", "vib_band_low", "vib_band_mid",
    "vib_band_high", "vib_amp_1x", "vib_amp_2x", "vib_ratio_2x_1x", "ele_mean", "ele_std",
    "ele_max", "ele_min", "ele_rms", "ele_kurtosis", "ele_skewness", "ele_spectral_energy",
    "ele_spectral_centroid",
]  # orden del contrato (constants.FEATURE_NAMES = build_dataset.FEATURE_NAMES)
LABELS = ["family", "severity", "group_id"]
TOL_CONFIDENCE = 0.05  # manifest tolerance_policy.severity_confidence


def _check_golden(cases, df, pred):
    """Los inputs del manifest son filas del parquet y los esperados, las de predictions.csv."""
    for case in cases:
        sid = case["trace"]["sample_id"]
        row, ref, exp = df.loc[sid], pred.loc[sid], case["expected"]
        if any(row[k] != v for k, v in case["input"].items()):
            raise SystemExit(f"{case['id']}: el input del manifest no es la fila {sid} del parquet")
        ref_sev = None if math.isnan(ref.pred_severity) else int(ref.pred_severity)
        ref_conf = None if math.isnan(ref.severity_confidence) else float(ref.severity_confidence)
        same_conf = (ref_conf is None and exp["severity_confidence"] is None) or (
            ref_conf is not None and abs(ref_conf - exp["severity_confidence"]) < 1e-12)
        if ref.pred_family != exp["family"] or ref_sev != exp["severity"] or not same_conf:
            raise SystemExit(f"{case['id']}: el esperado del manifest no es predictions.csv")


def main(repo: Path) -> None:
    out = repo / "outputs/a38/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    code = repo / "inbox/a38/codigo"
    df = pd.read_parquet(code / "data/processed/feature_dataset.parquet")
    df.index.name = "sample_id"
    pred = pd.read_csv(code / "data/predictions/predictions.csv").set_index("sample_id")
    cases = yaml.safe_load((repo / "inbox/a38/manifest.yaml").read_text(encoding="utf-8"))[
        "golden_cases"]
    _check_golden(cases, df, pred)
    sample_ids = [c["trace"]["sample_id"] for c in cases]

    # ── predict inline: un JSON por golden case (las 24 características de la ventana) ──
    for case in cases:
        row = df.loc[case["trace"]["sample_id"]]
        body = {"mode": "inline", **{f: float(row[f]) for f in FEATURES}}
        name = f"predict_inline_{case['id']}_{case['expected']['family']}.json"
        (out / name).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

    # ── predict batch: los 21 golden cases en un CSV (sample_id + 24 características) ──
    df.loc[sample_ids, FEATURES].reset_index().to_csv(out / "predict_batch_golden.csv",
                                                      index=False)

    # ── esperados ──
    rows = []
    for case in cases:
        tr, exp = case["trace"], case["expected"]
        rows.append({
            "id": case["id"], "sample_id": tr["sample_id"], "group_id": tr["group_id"],
            "fold_oof": tr["fold"], "motor": tr["motor"], "speed": tr["speed"],
            "fault": tr["fault"], "true_family": tr["true_family"],
            "true_severity": tr["true_severity"],
            "esperado_family": exp["family"], "esperado_severity": exp["severity"],
            "esperado_severity_label": exp["severity_label"],
            "esperado_severity_confidence": exp["severity_confidence"],
            "tolerancia_severity_confidence": TOL_CONFIDENCE,
        })
    expected = pd.DataFrame(rows)
    expected["esperado_severity"] = expected["esperado_severity"].astype("Int64")
    expected.to_csv(out / "expected_golden.csv", index=False)

    # ── train: la tabla completa con la que se entrenó el modelo servido (877 ventanas) ──
    # Mismo esquema que feature_dataset.parquet: 24 características + family, severity, group_id.
    # sample_id va como primera columna (train() la ignora; sirve también como entrada batch).
    train = df[FEATURES + LABELS].reset_index()
    train.to_csv(out / "train_feature_dataset_877.csv", index=False)
    # Los decimales escritos son exactos (repr de float64). Ojo: el lector por defecto de pandas
    # (el que usa el plugin) redondea la última cifra en algunos valores (≤ 1e-12 relativo).
    back = pd.read_csv(out / "train_feature_dataset_877.csv",
                       float_precision="round_trip").set_index("sample_id")
    if not back[FEATURES].equals(df[FEATURES]):
        raise SystemExit("El CSV de train no reproduce exactamente el parquet")

    print(f"OK: {len(cases)} golden cases, train {len(train)} filas / "
          f"{train.group_id.nunique()} grupos →", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
