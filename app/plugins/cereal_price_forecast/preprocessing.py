import pandas as pd

from app.plugins.cereal_price_forecast.predict_dto import PredictInlineRequest

_FIELD_RENAME: dict[str, str] = {
    "Fertilizers_index": "Fertilizers index",
    "Seeds_index": "Seeds index",
    "fertilizers_x_product_Durum_wheat": "fertilizers_x_product_Durum wheat",
    "fertilizers_x_product_Feed_barley": "fertilizers_x_product_Feed barley",
    "fertilizers_x_product_Feed_maize": "fertilizers_x_product_Feed maize",
    "fertilizers_x_product_Feed_wheat": "fertilizers_x_product_Feed wheat",
    "fertilizers_x_product_Malting_barley": "fertilizers_x_product_Malting barley",
    "fertilizers_x_product_Milling_wheat": "fertilizers_x_product_Milling wheat",
    "seeds_x_product_Durum_wheat": "seeds_x_product_Durum wheat",
    "seeds_x_product_Feed_barley": "seeds_x_product_Feed barley",
    "seeds_x_product_Feed_maize": "seeds_x_product_Feed maize",
    "seeds_x_product_Feed_wheat": "seeds_x_product_Feed wheat",
    "seeds_x_product_Malting_barley": "seeds_x_product_Malting barley",
    "seeds_x_product_Milling_wheat": "seeds_x_product_Milling wheat",
}

_METADATA_FIELDS = frozenset(["product_name", "market_name", "week_begin_date"])


def prepare_features(
    request: PredictInlineRequest,
    feature_cols: list[str],
    feature_medians: dict[str, float],
) -> pd.DataFrame:
    """Build the feature DataFrame expected by the LightGBM model."""
    raw = request.model_dump()

    for field in _METADATA_FIELDS:
        raw.pop(field, None)

    for old_key, new_key in _FIELD_RENAME.items():
        if old_key in raw:
            raw[new_key] = raw.pop(old_key)

    df = pd.DataFrame([raw])

    missing = [c for c in feature_cols if c not in df.columns]
    for col in missing:
        df[col] = float("nan")

    df = df[feature_cols]

    for col in df.columns:
        if df[col].isna().any():
            if col in feature_medians:
                df[col] = df[col].fillna(feature_medians[col])
            else:
                col_median = df[col].median()
                df[col] = df[col].fillna(col_median if not pd.isna(col_median) else 0.0)
    df = df.fillna(0.0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df
