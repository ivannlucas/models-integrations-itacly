"""Delivered a38 artifacts vs data/predictions/predictions.csv under a given Keras backend.

Usage: python -I artifact_backend_parity.py <torch|tensorflow> <artifacts_dir> <parquet> <predictions_csv>
Result (2026-11-17, sklearn 1.5.2 + keras 3.15.1): 877/877 families; |Δ confidence| ≤ 3.6e-7
(torch) / 1.2e-7 (tensorflow).
"""
import os
import sys
import warnings

os.environ["KERAS_BACKEND"] = sys.argv[1]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

ART = sys.argv[2]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rf = joblib.load(f"{ART}/family_rf.pkl")
    sc = joblib.load(f"{ART}/family_scaler.pkl")
    sev_models = joblib.load(f"{ART}/severity_models.pkl")
    sev_scalers = joblib.load(f"{ART}/severity_scalers.pkl")
print("sklearn", sklearn.__version__, "keras", keras.__version__, keras.backend.backend())

df = pd.read_parquet(sys.argv[3])
X = df.drop(columns=["family", "severity", "group_id"]).values
ref = pd.read_csv(sys.argv[4])
pred_family = rf.predict(sc.transform(X))
print("family match vs predictions.csv:", (pred_family == ref.pred_family.values).mean())

conf = []
for i, fam in enumerate(pred_family):
    if fam == "healthy" or fam not in sev_models:
        conf.append(np.nan)
        continue
    x_sev = sev_scalers[fam].transform(X[i:i + 1]).astype(np.float32)
    z = keras.ops.convert_to_numpy(sev_models[fam](x_sev, training=False))
    conf.append(float(1 / (1 + np.exp(-z.flatten()[0]))))
conf = np.array(conf)
rc = ref.severity_confidence.values
mask = ~np.isnan(rc)
print("nan pattern match:", (np.isnan(conf) == np.isnan(rc)).all(),
      "max abs diff conf:", np.nanmax(np.abs(conf[mask] - rc[mask])))
