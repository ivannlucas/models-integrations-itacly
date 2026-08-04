from __future__ import annotations

import pandas as pd

from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_predict_smoke_generates_predictions_latest() -> None:
    ensure_repro_smoke_pipeline()
    prediction_path = REPO_ROOT / "data/predictions/predictions_latest__mixed_context.csv"
    assert prediction_path.exists()

    df = pd.read_csv(prediction_path)
    expected_columns = {
        "synthetic_procurement_need_pred",
        "purchase_trigger_proba",
        "purchase_trigger_flag",
        "quantity_optimizer_recommendation_tons",
        "order_quantity_tons",
    }
    assert expected_columns.issubset(df.columns)
    assert (df.loc[df["purchase_trigger_flag"] == 0, "order_quantity_tons"] == 0.0).all()
