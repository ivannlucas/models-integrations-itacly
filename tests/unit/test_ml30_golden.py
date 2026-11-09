"""ml30 golden cases (inbox/a30/manifest.yaml) through the real plugin and served artifacts.

The expected scores come from the AI team's own predictor on data/splits/test.csv. Case
obs_OBS0000776 has an empty sensor_ph: it checks the plugin lets the fitted preprocessor impute
the training median, as the original does (it used to fill 0.0 and score that real incident 0).
Skipped when the local artifacts are not available.

Inline, the data-contract scale check sees a single row, so a real but extreme value (z > 10)
is rejected: golden obs_OBS0001565 (1199 h since the previous lot) gets DataContractError.
Accepted on purpose (inline prediction is disabled in the platform); see known_issues in the
manifest. Every golden case is still checked through predict_batch.
"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from app.domain.services.exceptions import DataContractError
from app.plugins.ml30_meat_traceability_detection.constants import ARTIFACT_FOLDER_NAME
from app.plugins.ml30_meat_traceability_detection.plugin import Ml30MeatTraceabilityDetectionPlugin

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "inbox" / "a30" / "manifest.yaml"
_GOLDEN = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))["golden_cases"]
TOLERANCE = 1e-6  # expected scores are rounded to 6 decimals
REJECTED_INLINE = {"obs_OBS0001565"}  # single-row scale check (see module docstring)


@pytest.fixture(name="served", scope="module")
def fixture_served():
    """The ml30 plugin loaded with the local served artifacts, or skip."""
    if not (_ROOT / "artifacts" / ARTIFACT_FOLDER_NAME / "model_state.pt").exists():
        pytest.skip("Artefactos de ml30 no disponibles en local")
    plugin = Ml30MeatTraceabilityDetectionPlugin()
    plugin.load()
    return plugin


@pytest.mark.parametrize("case", _GOLDEN, ids=[case["id"] for case in _GOLDEN])
def test_golden_inline(served, case):
    """predict_inline reproduces the original prediction and score of each golden case."""
    if case["id"] in REJECTED_INLINE:
        with pytest.raises(DataContractError, match="Otra escala"):
            served.predict_inline(features=case["input"])
        return
    response = served.predict_inline(features=case["input"])
    assert response.pred_traceability_incident == case["expected"]["pred_traceability_incident"]
    assert response.pred_score == pytest.approx(case["expected"]["pred_score"], abs=TOLERANCE)


def test_golden_batch(served, tmp_path):
    """predict_batch over all golden rows (an empty cell stays empty in the CSV) matches too."""
    csv = tmp_path / "golden.csv"
    pd.DataFrame([case["input"] for case in _GOLDEN]).to_csv(csv, index=False)
    rows = served.predict_batch(data_path=str(csv)).predictions
    for case, row in zip(_GOLDEN, rows):
        assert row["pred_traceability_incident"] == case["expected"]["pred_traceability_incident"]
        assert row["pred_score"] == pytest.approx(case["expected"]["pred_score"], abs=TOLERANCE)
