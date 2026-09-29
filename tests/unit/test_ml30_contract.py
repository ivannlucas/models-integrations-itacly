"""Unit tests for ml30's data-contract check against the fitted preprocessor."""
import logging

import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import DataContractError
from app.plugins.ml30_meat_traceability_detection.constants import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from app.plugins.ml30_meat_traceability_detection.contract import (
    check_data_contract,
    enforce_data_contract,
)
from app.plugins.ml30_meat_traceability_detection.tabular_preprocessor import TabularPreprocessor

_LOGGER = logging.getLogger(__name__)


def _valid_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.normal(0.5, 0.1, n) for col in NUMERIC_FEATURES}
    data.update({col: rng.choice(["a", "b", "c"], n) for col in CATEGORICAL_FEATURES})
    return pd.DataFrame(data)[FEATURE_COLUMNS]


@pytest.fixture(name="preprocessor")
def _preprocessor():
    return TabularPreprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(_valid_frame(200))


def test_valid_data_has_no_errors_or_warnings(preprocessor):
    df = _valid_frame()
    assert check_data_contract(df, preprocessor, df.columns) == ([], [])


def test_column_with_only_unknown_values_is_an_error(preprocessor):
    df = _valid_frame()
    df["stage"] = "boning"
    errors, _ = check_data_contract(df, preprocessor, df.columns)
    assert errors and "stage" in errors[0]


def test_ratio_sent_as_percentage_is_an_error(preprocessor):
    df = _valid_frame()
    df["yield_pct_from_parent"] = df["yield_pct_from_parent"] * 100
    errors, _ = check_data_contract(df, preprocessor, df.columns)
    assert errors and "yield_pct_from_parent" in errors[0]


def test_partially_unknown_values_only_warn(preprocessor):
    df = _valid_frame()
    df.loc[:4, "plant_line"] = "L2"
    errors, warnings = check_data_contract(df, preprocessor, df.columns)
    assert errors == []
    assert any("plant_line" in w for w in warnings)


def test_missing_values_are_not_unknown_categories(preprocessor):
    df = _valid_frame()
    df.loc[:9, "prev_stage"] = "nan"
    assert check_data_contract(df, preprocessor, df.columns) == ([], [])


def test_absent_column_warns_instead_of_failing(preprocessor):
    df = _valid_frame()
    present = [c for c in df.columns if c != "cold_room_id"]
    errors, warnings = check_data_contract(df, preprocessor, present)
    assert errors == []
    assert any("cold_room_id" in w for w in warnings)


def test_enforce_raises_short_detail(preprocessor):
    df = _valid_frame()
    for col in CATEGORICAL_FEATURES:
        df[col] = "desconocido"
    df["yield_pct_from_parent"] = df["yield_pct_from_parent"] * 100
    with pytest.raises(DataContractError) as exc:
        enforce_data_contract(df, preprocessor, df.columns, _LOGGER)
    assert len(str(exc.value)) <= 300
