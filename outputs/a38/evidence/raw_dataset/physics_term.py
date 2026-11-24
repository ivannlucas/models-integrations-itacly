"""Physics-guided severity loss: does the rule hold in the data, and does the term change the CV?

1. Rule check: the penalty relu(low_band_norm − p_significant) pushes the probability of
   "significant" up where the normalised vib_band_low is high. Per family: Spearman correlation
   between severity level and vib_band_low, and mean normalised low band of each binary class.
2. Ablation: GroupKFold CV of each family's severity MLP with λ = 0 vs the delivered λ = 0.3,
   3 seeds each, with the plugin's own training code (training._family_severity_cv).

Usage (from the repo root, PYTHONPATH=., KERAS_BACKEND=torch): python physics_term.py <codigo_dir>
"""
import functools
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import app.plugins.ml38_wine_machinery_fault_detection.training as training
from app.plugins.ml38_wine_machinery_fault_detection.constants import FEATURE_NAMES


def rule_check(df):
    data = df[(df.family != "healthy") & (df.severity > 0)].copy()
    data["significant"] = (data.severity > 1).astype(int)
    rows = []
    for family, group in data.groupby("family"):
        low = group.vib_band_low
        low_norm = (low - low.min()) / (low.max() - low.min() + 1e-8)
        rows.append({
            "familia": family, "n": len(group),
            "rho_severidad_banda_baja": round(spearmanr(group.severity, low).correlation, 3),
            "banda_baja_norm_leve": round(low_norm[group.significant == 0].mean(), 3),
            "banda_baja_norm_significativo": round(low_norm[group.significant == 1].mean(), 3),
            "p_mann_whitney": round(mannwhitneyu(low[group.significant == 0],
                                                 low[group.significant == 1]).pvalue, 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def ablation(df):
    x = df[FEATURE_NAMES].to_numpy(float)
    family, severity, groups = df.family.to_numpy(), df.severity.to_numpy(), df.group_id.to_numpy()
    original_loss = training.physics_guided_loss
    rows = []
    try:
        for lam in (0.0, 0.3):
            training.physics_guided_loss = functools.partial(original_loss, lam)
            for fam in sorted(set(family) - {"healthy"}):
                mask = (family == fam) & (severity > 0)
                y = np.where(severity[mask] == 1, 0, 1)
                n_splits = min(5, len(np.unique(groups[mask])))
                for seed in (1, 2, 3):
                    cv = training._family_severity_cv(  # pylint: disable=protected-access
                        x[mask], y, groups[mask], n_splits, 1000 * seed)
                    rows.append({"lambda": lam, "familia": fam, "seed": seed, **cv})
    finally:
        training.physics_guided_loss = original_loss
    result = pd.DataFrame(rows).groupby(["familia", "lambda"])[["accuracy_mean", "macro_f1_mean"]]
    print(result.agg(["mean", "std"]).round(3).to_string())


def main(codigo_dir):
    df = pd.read_parquet(f"{codigo_dir}/data/processed/feature_dataset.parquet")
    print("1. ¿Se cumple la regla física en los datos?")
    rule_check(df)
    print("\n2. Ablación del término físico (CV de severidad, 3 semillas)")
    ablation(df)


if __name__ == "__main__":
    main(sys.argv[1])
