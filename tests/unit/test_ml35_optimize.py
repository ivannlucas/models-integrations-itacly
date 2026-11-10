"""ml35 ``optimize`` mode (pygad GA) run by the real plugin code, not the conftest fake.

The endpoint test in test_ml35_dairy_ann_cleaning_cost.py goes through the fake plugin, so it
never caught that ``_run_optimize`` overwrote its ``ctx`` tuple with the scenario features
(ValueError: too many values to unpack) nor that pygad was missing from requirements.txt.
"""
# pylint: disable=protected-access  # tests call the private GA helper _run_optimize with a ctx
from pathlib import Path

import joblib
import pytest
import torch

from app.plugins.ml35_dairy_ann_cleaning_cost import plugin as ml35
from app.plugins.ml35_dairy_ann_cleaning_cost.constants import FEATURES
from app.plugins.ml35_dairy_ann_cleaning_cost.model_loader import PasteurizationANN

pytest.importorskip("pygad")

_DELIVERED_ARTIFACTS = (
    Path(__file__).resolve().parents[2] / "inbox" / "a35" / "codigo" / "models" / "artifacts"
)

# AI team's own GA output (random_seed=42) for the delivered model:
# inbox/a35/codigo/data/predictions/optimization_results_single_mode.csv (row 1) and
# optimization_results_massive_mode.csv (rows 1-2).
GOLDEN = [
    ({"temp_entrada_leche": 14.5, "temp_ambiente": 31.0, "horas_desde_limpieza": 12.0,
      "presion_diferencial_bar": 2.1},
     {"opt_temp_leche": 72.02595024383646, "opt_temp_agua": 91.92316834197237,
      "opt_flujo": 2510.123167692437,
      "consumo_estandar": 26723.205078125, "consumo_optimizado": 17079.13671875,
      "ahorro_pct": 36.088741343639995,
      "pu_logrado": 23.143797939474133}),
    ({"temp_entrada_leche": 4.94, "temp_ambiente": 22.89, "horas_desde_limpieza": 19.6,
      "presion_diferencial_bar": 1.98},
     {"opt_temp_leche": 72.09072404652726, "opt_temp_agua": 86.33851578884543,
      "opt_flujo": 2510.123167692437,
      "consumo_estandar": 29038.9765625, "consumo_optimizado": 18906.814453125,
      "ahorro_pct": 34.89159505180134,
      "pu_logrado": 23.642207755293345}),
    ({"temp_entrada_leche": 6.38, "temp_ambiente": 19.12, "horas_desde_limpieza": 21.6,
      "presion_diferencial_bar": 2.22},
     {"opt_temp_leche": 72.01029479845633, "opt_temp_agua": 89.05626102310191,
      "opt_flujo": 2510.123167692437,
      "consumo_estandar": 29797.3046875, "consumo_optimizado": 19113.9296875,
      "ahorro_pct": 35.8534945091248,
      "pu_logrado": 23.024920364061387}),
]


class _ConstantModel(torch.nn.Module):
    """Stand-in ANN: always predicts 0 in scaled space."""

    def forward(self, x):
        """Return a zero prediction per input row."""
        return torch.zeros((x.shape[0], 1))


class _IdentityScaler:
    """Stand-in scaler: identity forward, +1000 on the inverse transform."""

    def transform(self, x):
        """Return the input unchanged (as a numpy array if it is a DataFrame)."""
        return x.to_numpy() if hasattr(x, "to_numpy") else x

    def inverse_transform(self, x):
        """Shift values by +1000 so the de-scaled prediction is easy to check."""
        return x + 1000.0


def test_optimize_uses_the_model_it_is_given():
    """The (model, scaler_X, scaler_y) passed in ctx — base or retrained — drives the GA."""
    response = ml35.Ml35DairyAnnCleaningCostPlugin()._run_optimize(
        GOLDEN[0][0], (_ConstantModel(), _IdentityScaler(), _IdentityScaler()),
    )
    assert response.consumo_estandar == pytest.approx(1000.0)
    assert response.consumo_optimizado == pytest.approx(1000.0)
    assert response.pu_logrado >= ml35.PU_MIN


@pytest.fixture(name="delivered_ctx", scope="module")
def fixture_delivered_ctx():
    """Load the AI team's delivered ANN and scalers, or skip if they are not available."""
    if not _DELIVERED_ARTIFACTS.exists():
        pytest.skip(f"Artefactos entregados no disponibles: {_DELIVERED_ARTIFACTS}")
    model = PasteurizationANN(input_size=len(FEATURES))
    model.load_state_dict(
        torch.load(_DELIVERED_ARTIFACTS / "model_ann.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return (
        model,
        joblib.load(_DELIVERED_ARTIFACTS / "scaler_X.pkl"),
        joblib.load(_DELIVERED_ARTIFACTS / "scaler_y.pkl"),
    )


@pytest.mark.parametrize("scenario,expected", GOLDEN)
def test_optimize_reproduces_ai_team_results(delivered_ctx, scenario, expected):
    """The GA on the delivered model reproduces the AI team's own optimization output."""
    response = ml35.Ml35DairyAnnCleaningCostPlugin()._run_optimize(
        scenario, delivered_ctx
    ).model_dump()
    for field, value in expected.items():
        assert response[field] == pytest.approx(value, rel=1e-4), field
