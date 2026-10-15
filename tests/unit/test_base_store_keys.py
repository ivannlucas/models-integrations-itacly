"""Production coupling: the container/publisher store and each plugin's loader store must share
the active-version key (_ACTIVE_VERSIONS is keyed by the store's local dir), otherwise a swap or
rollback would not change what the loader reads."""
import importlib

import pytest

from app.infrastructure.artifact_store import ArtifactStore

PLUGINS = [
    ("ml13_wine_price_fluctuation_prediction", "Ml13WinePriceFluctuationPredictionPlugin"),
    ("ml14_wine_phyto_price_forecast", "Ml14WinePhytoPriceForecastPlugin"),
    ("ml15_wine_ipi_price_forecast", "Ml15WineIpiPriceForecastPlugin"),
    ("ml16_meat_raw_material_price_alert", "Ml16MeatRawMaterialPriceAlertPlugin"),
    ("ml17_meat_market_price_analysis", "Ml17MeatMarketPriceAnalysisPlugin"),
    ("ml18_meat_spatial_price_forecast", "Ml18MeatSpatialPriceForecastPlugin"),
    ("ml21_cereals_price_spatial", "Ml21CerealsPriceSpatialPlugin"),
    ("ml23_lactic_market_price_forecast", "Ml23LacticMarketPriceForecastPlugin"),
]


def _import(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # heavy deps (tensorflow, torch...) may be missing locally
        pytest.skip(f"{name} not importable here: {exc}")


@pytest.mark.parametrize("package,class_name", PLUGINS)
def test_loader_store_shares_the_publisher_key(package, class_name):
    base = f"app.plugins.{package}"
    constants = _import(f"{base}.constants")
    loader = _import(f"{base}.model_loader")
    plugin_cls = getattr(_import(f"{base}.plugin"), class_name)
    assert plugin_cls.BASE_ARTIFACT_FOLDER == constants.ARTIFACT_FOLDER_NAME
    # The real store, not the autouse temp one: that is what the container builds in production.
    assert ArtifactStore(constants.ARTIFACT_FOLDER_NAME)._key() == loader._store._key()  # pylint: disable=protected-access
