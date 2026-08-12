"""Feature engineering exports for platform and reproducibility routes."""

from src.feature_engineering.build_features import build_platform_features


def run_feature_engineering(config, logger):
    from .pipeline import run_feature_engineering as _run_feature_engineering

    return _run_feature_engineering(config, logger)


__all__ = ["build_platform_features", "run_feature_engineering"]
