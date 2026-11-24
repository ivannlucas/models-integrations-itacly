"""Rebuild the 877 training windows from the raw NLN-EMP files (vibration ch1 + current ch1).

For every row of input_manifest.csv it computes the 24 features with the AI team's original
preprocess.py and with the plugin port, the same features on signals cut to 4096 / 20000 / 60000 /
120000 samples (KI-05), and the largest vibration peak within ±15 % of the code's rotation
frequency (KI-07). Results are written as parquet/CSV for analyze_rebuild.py.

Usage (from the repo root, PYTHONPATH=.):
    python rebuild_from_raw.py <dataset_dir> <codigo_dir> <out_dir>
<dataset_dir> is the folder that holds Vibration/ and Electric/ (~6 min, reads 16 GB).
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

from app.plugins.ml38_wine_machinery_fault_detection import preprocessing as pp
from app.plugins.ml38_wine_machinery_fault_detection.constants import FEATURE_NAMES

TRUNCATION_LENGTHS = [4096, 20000, 60000, 120000]


def load_original_preprocess(codigo_dir):
    """The AI team's src/data_processing/preprocess.py, loaded by path (no 'src' package clash)."""
    spec = importlib.util.spec_from_file_location(
        "orig_pre", f"{codigo_dir}/src/data_processing/preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_vector(orig, vib, ele, motor, speed):
    """Feature vector exactly as build_dataset.build_processed_dataset concatenates it."""
    rpm = orig.get_rpm(motor, speed)
    return np.concatenate([orig.extract_time_features(vib), orig.extract_spectral_features(vib),
                           orig.extract_physical_vibration_features(vib, rpm),
                           orig.extract_time_features(ele), orig.extract_spectral_features(ele)])


def measured_1x(vib, rpm):
    """Frequency of the largest vibration peak within ±15 % of the nominal rotation frequency."""
    spectrum = np.abs(rfft(vib))
    freqs = rfftfreq(len(vib), 1 / 20000)
    f0 = rpm / 60
    mask = (freqs > 0.85 * f0) & (freqs < 1.15 * f0)
    return float(freqs[mask][np.argmax(spectrum[mask])])


def main(dataset_dir, codigo_dir, out_dir):
    orig = load_original_preprocess(codigo_dir)
    manifest = pd.read_csv(f"{codigo_dir}/data/processed/input_manifest.csv")
    rows_orig, rows_plugin, extra = [], [], []
    truncated = {length: [] for length in TRUNCATION_LENGTHS}
    start = time.time()
    for group_id, group in manifest.groupby("group_id", sort=True):
        first = group.iloc[0]
        base = f"{first.motor}/{first.speed}/{first.fault}"
        # Same reader as the original ETL (pandas C engine, dtype=float) for bit-level parity.
        vib_df = pd.read_csv(f"{dataset_dir}/Vibration/{base}/{first.vibration_file}", dtype=float)
        ele_df = pd.read_csv(f"{dataset_dir}/Electric/{base}/{first.electrical_file}", dtype=float)
        n_windows = min(vib_df.shape[1] - 1, ele_df.shape[1] - 1)
        for _, row in group.iterrows():
            col = int(row.window_index) + 1
            vib, ele = vib_df.iloc[:, col].values, ele_df.iloc[:, col].values
            rows_orig.append(original_vector(orig, vib, ele, row.motor, str(row.speed)))
            rows_plugin.append(pp.build_feature_vector(vib, ele, row.motor, int(row.speed)))
            for length in TRUNCATION_LENGTHS:
                truncated[length].append(
                    pp.build_feature_vector(vib[:length], ele[:length], row.motor, int(row.speed)))
            rpm = orig.get_rpm(row.motor, str(row.speed))
            extra.append({"sample_id": int(row.sample_id), "group_id": int(group_id),
                          "n_win_file": n_windows, "len_vib": len(vib), "len_ele": len(ele),
                          "rpm_code": rpm, "f1x_code": rpm / 60,
                          "f1x_measured": measured_1x(vib, rpm)})
        print(f"group {group_id} ({base}) windows={len(group)}/{n_windows} "
              f"t={time.time() - start:.0f}s", flush=True)

    pd.DataFrame(rows_orig, columns=FEATURE_NAMES).to_parquet(f"{out_dir}/feat_orig.parquet")
    pd.DataFrame(rows_plugin, columns=FEATURE_NAMES).to_parquet(f"{out_dir}/feat_plugin.parquet")
    for length, rows in truncated.items():
        pd.DataFrame(rows, columns=FEATURE_NAMES).to_parquet(
            f"{out_dir}/feat_trunc_{length}.parquet")
    pd.DataFrame(extra).to_csv(f"{out_dir}/extra.csv", index=False)
    print("DONE", round(time.time() - start, 1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
