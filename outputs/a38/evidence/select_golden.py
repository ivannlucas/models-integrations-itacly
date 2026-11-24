"""Deterministic golden-case selection for a38 (no synthetic inputs).

Rule: for each of the 7 families, walk the folds in the order 1,3,5,2,4 and take the FIRST window
(lowest sample_id) of that family in the fold, until 3 cases per family. Inputs = the 24 features
of data/processed/feature_dataset.parquet; expected = data/predictions/predictions.csv (delivered
final model) for the same sample_id. Prints the YAML block pasted into inbox/a38/manifest.yaml.

Usage: python -I select_golden.py <inbox/a38/codigo>
"""
import sys

import numpy as np
import pandas as pd

C = sys.argv[1]
df = pd.read_parquet(f"{C}/data/processed/feature_dataset.parquet")
fa = pd.read_csv(f"{C}/data/splits/fold_assignments.csv")
pr = pd.read_csv(f"{C}/data/predictions/predictions.csv")
man = pd.read_csv(f"{C}/data/processed/input_manifest.csv")
assert (fa.sample_id.values == np.arange(len(df))).all()
assert (fa.group_id.values == df.group_id.values).all()
feats = [c for c in df.columns if c not in ("family", "severity", "group_id")]

picked = []
for fam in sorted(df.family.unique()):
    n = 0
    for fold in (1, 3, 5, 2, 4):
        sub = fa[(fa.family == fam) & (fa.fold == fold)].sort_values("sample_id")
        if sub.empty:
            continue
        picked.append(int(sub.sample_id.iloc[0]))
        n += 1
        if n == 3:
            break

for i, sid in enumerate(picked, 1):
    r, p, m = df.iloc[sid], pr.iloc[sid], man.iloc[sid]
    assert p.sample_id == sid
    sev = None if pd.isna(p.pred_severity) else int(p.pred_severity)
    conf = None if pd.isna(p.severity_confidence) else float(p.severity_confidence)
    if p.pred_family == "healthy":
        label = "sin fallo"
    else:
        label = "leve" if sev == 0 else "significativo"
    fold = int(fa.fold.iloc[sid])
    print(f"  - id: caso_{i:03d}")
    print(f"    source: dataset_split_oof_fold  # data/splits/fold_assignments.csv fold={fold}; "
          f"expected = data/predictions/predictions.csv")
    print(f"    trace: {{sample_id: {sid}, group_id: {int(r.group_id)}, fold: {fold}, "
          f"motor: {m.motor}, speed: {int(m.speed)}, fault: \"{m.fault}\", "
          f"window_index: {int(m.window_index)}, true_family: {r.family}, "
          f"true_severity: {int(r.severity)}}}")
    print("    input:")
    for f in feats:
        print(f"      {f}: {repr(float(r[f]))}")
    print("    expected:")
    print(f"      family: {p.pred_family}")
    print(f"      severity: {'null' if sev is None else sev}")
    print(f"      severity_label: \"{label}\"")
    print(f"      severity_confidence: {'null' if conf is None else repr(conf)}")
