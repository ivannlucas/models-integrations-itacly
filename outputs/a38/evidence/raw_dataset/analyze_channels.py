"""Summarise channels.csv (channel_variants.py): family accuracy per channel variant (KI-06).

The ch1 baseline is in-sample (the served model was trained on those exact windows): the other
channels are the same tests seen through another sensor, not an independent test set.

Usage: python analyze_channels.py <codigo_dir> <out_dir_of_channel_variants>
"""
import sys

import pandas as pd


def main(codigo_dir, out_dir):
    ref = pd.read_csv(f"{codigo_dir}/data/predictions/predictions.csv")
    channels = pd.read_csv(f"{out_dir}/channels.csv").merge(
        ref[["sample_id", "true_family"]], on="sample_id")
    channels["ok"] = channels.pred_family == channels.true_family
    print("Referencia vib ch1 + corriente ch1 (in-sample): acierto 1.000")
    summary = channels.groupby("variant").agg(
        n=("ok", "size"), acierto=("ok", "mean"),
        pred_healthy=("pred_family", lambda s: (s == "healthy").mean()))
    print(summary.round(3).to_string())
    print("\nAcierto por familia real:")
    print(channels.pivot_table(index="true_family", columns="variant", values="ok",
                               aggfunc="mean").round(2).to_string())
    faults = channels[channels.true_family != "healthy"]
    print("\nVentanas con fallo predichas como healthy (fallos no detectados):")
    print(faults.groupby("variant").pred_family.apply(lambda s: (s == "healthy").mean())
          .round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
