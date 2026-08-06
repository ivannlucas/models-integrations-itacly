from __future__ import annotations

import pandas as pd

from src.feature_engineering import build_platform_features
from src.trigger import run_purchase_trigger


def test_trigger_flag_is_one_when_projected_inventory_drops_below_safety_stock() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "date": "2025-01-05",
                "raw_material_id": "RM_A",
                "current_inventory_tons": 10.0,
                "expected_requirement_tons": 21.0,
                "lead_time_days": 7.0,
                "safety_coverage_days": 7.0,
                "expected_yield_rate": 0.9,
                "expected_waste_rate": 0.02,
                "unit_purchase_cost": 3.0,
                "shelf_life_days": 14.0,
                "destination_profile": "cooked_standard_line",
            }
        ]
    )
    triggered = run_purchase_trigger(build_platform_features(dataframe))
    assert int(triggered.loc[0, "purchase_trigger_flag"]) == 1


def test_trigger_flag_is_zero_when_inventory_is_sufficient() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "date": "2025-01-05",
                "raw_material_id": "RM_A",
                "current_inventory_tons": 80.0,
                "expected_requirement_tons": 12.0,
                "lead_time_days": 3.0,
                "safety_coverage_days": 7.0,
                "expected_yield_rate": 0.9,
                "expected_waste_rate": 0.02,
                "unit_purchase_cost": 3.0,
                "shelf_life_days": 14.0,
                "destination_profile": "cooked_standard_line",
            }
        ]
    )
    triggered = run_purchase_trigger(build_platform_features(dataframe))
    assert int(triggered.loc[0, "purchase_trigger_flag"]) == 0
