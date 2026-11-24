"""Analyse the output of rebuild_from_raw.py against the delivered parquet and predictions.

Prints: ETL reproducibility (raw → delivered parquet), plugin port vs original code, raw-mode
end-to-end parity with predictions.csv, prediction drift on cut signals (KI-05) and the measured
1X peak vs the code's rotation frequency (KI-07).

Usage (from the repo root, PYTHONPATH=., KERAS_BACKEND=torch like the server):
    python analyze_rebuild.py <codigo_dir> <out_dir_of_rebuild>
"""
import sys

import numpy as np
import pandas as pd

from app.plugins.ml38_wine_machinery_fault_detection.constants import FEATURE_NAMES
from app.plugins.ml38_wine_machinery_fault_detection.inference import predict_matrix
from app.plugins.ml38_wine_machinery_fault_detection.model_loader import load_bundle_from

TRUNCATION_LENGTHS = [4096, 20000, 60000, 120000]


def predict(bundle, frame):
    """(families, severities, confidences) of the plugin's hierarchical inference."""
    res = predict_matrix(bundle, frame.to_numpy(float))
    fam = np.array([r["family"] for r in res])
    sev = np.array([np.nan if r["severity"] is None else r["severity"] for r in res], float)
    conf = np.array([np.nan if r["severity_confidence"] is None else r["severity_confidence"]
                     for r in res], float)
    return fam, sev, conf


def main(codigo_dir, out_dir):
    parquet = pd.read_parquet(f"{codigo_dir}/data/processed/feature_dataset.parquet")
    manifest = pd.read_csv(f"{codigo_dir}/data/processed/input_manifest.csv")
    ref = pd.read_csv(f"{codigo_dir}/data/predictions/predictions.csv")
    feat_orig = pd.read_parquet(f"{out_dir}/feat_orig.parquet")
    feat_plugin = pd.read_parquet(f"{out_dir}/feat_plugin.parquet")
    extra = pd.read_csv(f"{out_dir}/extra.csv")
    assert (extra.sample_id.to_numpy() == np.arange(len(parquet))).all()

    windows_used = manifest.groupby("group_id").size().reindex(extra.group_id).to_numpy()
    print("Ventanas del fichero == ventanas usadas:", bool((extra.n_win_file == windows_used).all()),
          "| longitudes vib", extra.len_vib.unique().tolist(), "ele", extra.len_ele.unique().tolist())

    print("Port del plugin vs código original: max |dif| =",
          float(np.abs(feat_plugin.values - feat_orig.values).max()))
    expected = parquet[FEATURE_NAMES].to_numpy(float)
    rel = np.abs(feat_orig.values - expected) / np.maximum(np.abs(expected), 1e-12)
    print("Original desde raw vs parquet entregado: max dif relativa =", float(rel.max()))
    print(pd.Series(rel.max(axis=0), index=FEATURE_NAMES).sort_values(ascending=False).head(3)
          .to_string())

    bundle = load_bundle_from(lambda name: f"{codigo_dir}/models/artifacts/{name}")
    fam, sev, conf = predict(bundle, feat_plugin)
    print("\nModo raw de punta a punta vs predictions.csv:")
    print("  familia", int((fam == ref.pred_family.to_numpy()).sum()), "/", len(ref))
    print("  severidad", int(np.nansum(sev == ref.pred_severity.to_numpy())), "/",
          int(ref.pred_severity.notna().sum()))
    print("  max |dif confianza|", float(np.nanmax(np.abs(conf - ref.severity_confidence))))

    true = ref.true_family.to_numpy()
    print("\nSeñal recortada (KI-05) — acierto de familia (in-sample) y concordancia con la ventana "
          "completa:")
    print(f"  completa 240000/300000: acierto {np.mean(fam == true):.3f}")
    for length in TRUNCATION_LENGTHS:
        fam_cut, _, _ = predict(bundle, pd.read_parquet(f"{out_dir}/feat_trunc_{length}.parquet"))
        dist = pd.Series(fam_cut).value_counts(normalize=True).round(2).to_dict()
        print(f"  L={length:>6} ({length / 20000:.2f} s): acierto {np.mean(fam_cut == true):.3f} "
              f"concordancia {np.mean(fam_cut == fam):.3f} distribución {dist}")

    extra["cfg"] = manifest.motor + "@" + manifest.speed.astype(str)
    extra["diff_hz"] = extra.f1x_measured - extra.f1x_code
    print("\n1X (KI-07): pico medido en ±15 % − frecuencia de giro del código (Hz); "
          "la tolerancia del código es ±2 Hz")
    print(extra.groupby("cfg").diff_hz.describe()[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
