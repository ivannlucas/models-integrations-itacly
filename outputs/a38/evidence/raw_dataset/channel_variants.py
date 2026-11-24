"""KI-06: predict every training window with the other measurement channels of the same test.

Variants: vibration chK (K=2..5) + current ch1, and vibration ch1 + electric chJ (J=2..6; ch4-6
are phase voltages in V). Writes channels.csv for analyze_channels.py.

Usage (from the repo root, PYTHONPATH=., KERAS_BACKEND=torch like the server):
    python channel_variants.py <dataset_dir> <codigo_dir> <out_dir>
(~16 min, reads ~68 GB).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from app.plugins.ml38_wine_machinery_fault_detection import preprocessing as pp
from app.plugins.ml38_wine_machinery_fault_detection.inference import predict_matrix
from app.plugins.ml38_wine_machinery_fault_detection.model_loader import load_bundle_from


def read(path):
    return pd.read_csv(path, dtype=float, engine="pyarrow")


def main(dataset_dir, codigo_dir, out_dir):
    bundle = load_bundle_from(lambda name: f"{codigo_dir}/models/artifacts/{name}")
    manifest = pd.read_csv(f"{codigo_dir}/data/processed/input_manifest.csv")
    out, start = [], time.time()
    for group_id, group in manifest.groupby("group_id", sort=True):
        first = group.iloc[0]
        base = f"{first.motor}/{first.speed}/{first.fault}"
        vib_stem = first.vibration_file[:-len("ch1.csv")]
        ele_stem = first.electrical_file[:-len("ch1.csv")]
        vib1 = read(f"{dataset_dir}/Vibration/{base}/{first.vibration_file}")
        ele1 = read(f"{dataset_dir}/Electric/{base}/{first.electrical_file}")
        variants = (
            [(f"vib_ch{k}", f"{dataset_dir}/Vibration/{base}/{vib_stem}ch{k}.csv", "vib")
             for k in range(2, 6)]
            + [(f"ele_ch{j}", f"{dataset_dir}/Electric/{base}/{ele_stem}ch{j}.csv", "ele")
               for j in range(2, 7)]
        )
        for name, path, kind in variants:
            if not os.path.exists(path):
                continue
            alt = read(path)
            feats, ids = [], []
            for _, row in group.iterrows():
                col = int(row.window_index) + 1
                if col >= alt.shape[1]:
                    continue
                vib = (alt if kind == "vib" else vib1).iloc[:, col].values
                ele = (alt if kind == "ele" else ele1).iloc[:, col].values
                feats.append(pp.build_feature_vector(vib, ele, row.motor, int(row.speed)))
                ids.append(int(row.sample_id))
            if feats:
                for sample_id, pred in zip(ids, predict_matrix(bundle, np.array(feats))):
                    out.append({"sample_id": sample_id, "variant": name,
                                "pred_family": pred["family"], "pred_severity": pred["severity"],
                                "conf": pred["severity_confidence"]})
        print(f"group {group_id} ({base}) t={time.time() - start:.0f}s", flush=True)
    pd.DataFrame(out).to_csv(f"{out_dir}/channels.csv", index=False)
    print("DONE", round(time.time() - start, 1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
