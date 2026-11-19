"""Helper run in its own process by test_ml39_inference.py — never collected by pytest
(leading underscore). Must run isolated: ml1's model_loader (and a37's, once merged) sets
KERAS_BACKEND=torch at import time for the standalone ``keras`` package, which TensorFlow 2.16+
reuses as ``tf.keras``. Once another plugin has done that in the same process, ml39's
``tf.GradientTape``-based Integrated Gradients silently breaks (``RuntimeError: Can't call
numpy() on Tensor that requires grad`` — a torch tensor leaking into a tf.GradientTape).
ml39 is never deployed in the same pod/process as another model in production (Kubernetes sets
MODEL=<id> per pod), but the shared pytest process and the local multi-model dev server are
exposed to this — see inbox/a39/manifest.yaml / verification report known_issues.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")

from app.plugins.ml39_wine_predictive_maintenance_xai.plugin import (  # noqa: E402  pylint: disable=wrong-import-position
    Ml39WinePredictiveMaintenanceXaiPlugin,
)

_PHYSICAL_SENSORS = (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4", "VS1",
)


def _readings_for_cycle(x_raw: np.ndarray, cycle_id: int) -> list:
    cycle = x_raw[cycle_id - 1]
    readings = []
    for t in range(cycle.shape[0]):
        row = {"t": float(t) / 10.0}
        for j, name in enumerate(_PHYSICAL_SENSORS):
            row[name] = float(cycle[t, j])
        readings.append(row)
    return readings


def main() -> None:
    """Explain each requested cycle and print its result as one JSON line."""
    cycle_ids = json.loads(sys.argv[1])
    x_raw = np.load("inbox/a37/codigo/data/processed/tensors/X.npy")

    plugin = Ml39WinePredictiveMaintenanceXaiPlugin()
    plugin.load()

    for cycle_id in cycle_ids:
        resp = plugin.predict_inline(features={"readings": _readings_for_cycle(x_raw, cycle_id)})
        print(json.dumps({
            "cycle_id": cycle_id,
            "predicted_class": resp.predicted_class,
            "ig_importance": resp.ig_importance,
        }))


if __name__ == "__main__":
    main()
