from __future__ import annotations

import pandas as pd

from src.feature_engineering import build_platform_features
from src.optimizer import run_quantity_optimizer
from src.trigger import run_purchase_trigger


def test_order_quantity_is_zero_when_trigger_is_negative() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "date": "2025-01-05",
                "raw_material_id": "RM_A",
                "current_inventory_tons": 95.0,
                "expected_requirement_tons": 15.0,
                "lead_time_days": 2.0,
                "safety_coverage_days": 7.0,
                "expected_yield_rate": 0.92,
                "expected_waste_rate": 0.01,
                "unit_purchase_cost": 3.1,
                "shelf_life_days": 21.0,
                "destination_profile": "cooked_standard_line",
            }
        ]
    )
    feature_frame = build_platform_features(dataframe)
    trigger_frame = run_purchase_trigger(feature_frame)
    optimized = run_quantity_optimizer(trigger_frame)
    assert int(optimized.loc[0, "purchase_trigger_flag"]) == 0
    assert float(optimized.loc[0, "order_quantity_tons"]) == 0.0
