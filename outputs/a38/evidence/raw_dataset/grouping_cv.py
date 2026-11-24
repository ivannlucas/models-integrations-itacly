"""Family OOF with the delivered grouping vs grouping by physical fault (motor, fault).

The delivered GroupKFold groups by (motor, speed, fault), so the same physical fault at another
speed of Motor-2 stays in training. Both groupings are evaluated with the delivered RandomForest
settings over 5 shuffled StratifiedGroupKFold(5) partitions, plus the delivered GroupKFold(5).

Usage (from the repo root, PYTHONPATH=.): python grouping_cv.py <codigo_dir>   (~4 min)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from app.plugins.ml38_wine_machinery_fault_detection.constants import FEATURE_NAMES


def oof_scores(x, y, groups, splitter):
    """Out-of-fold (accuracy, macro-F1) of scaler + RF(800, leaf 2, balanced, seed 42)."""
    pred = np.empty(len(y), dtype=object)
    for train, test in splitter.split(x, y, groups):
        scaler = StandardScaler().fit(x[train])
        rf = RandomForestClassifier(n_estimators=800, min_samples_leaf=2, class_weight="balanced",
                                    random_state=42, n_jobs=4)
        rf.fit(scaler.transform(x[train]), y[train])
        pred[test] = rf.predict(scaler.transform(x[test]))
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def main(codigo_dir):
    df = pd.read_parquet(f"{codigo_dir}/data/processed/feature_dataset.parquet")
    manifest = pd.read_csv(f"{codigo_dir}/data/processed/input_manifest.csv")
    x, y = df[FEATURE_NAMES].to_numpy(float), df.family.to_numpy()
    groupings = {
        "original (motor, velocidad, fallo)": df.group_id.to_numpy(),
        "fallo físico (motor, fallo)": (manifest.motor + "|" + manifest.fault).to_numpy(),
    }
    acc, macro = oof_scores(x, y, groupings["original (motor, velocidad, fallo)"], GroupKFold(5))
    print(f"GroupKFold(5) entregado: acc {acc:.4f}  macro-F1 {macro:.4f}")
    for name, groups in groupings.items():
        res = np.array([oof_scores(x, y, groups, StratifiedGroupKFold(5, shuffle=True,
                                                                      random_state=k))
                        for k in range(5)])
        print(f"{name}, 5 particiones aleatorias: acc {res[:, 0].mean():.4f}±"
              f"{res[:, 0].std():.4f}  macro-F1 {res[:, 1].mean():.4f}±{res[:, 1].std():.4f}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
