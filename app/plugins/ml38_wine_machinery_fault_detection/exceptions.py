"""Plugin-local domain errors for ml38 (mapped to HTTP 422 via registry.extra_predict_exceptions).

Kept inside the plugin instead of app/domain/services/exceptions.py because this integration was
scoped not to modify app/domain (same choice as ml6, ml24 and ml26); they can be promoted to the
domain module in a later refactor.
"""


class UnsupportedOperatingConfigurationError(ValueError):
    """Raised when raw inference asks for a motor/speed pair that is not one of the documented
    NLN-EMP configurations (memoria §3.1 Tabla 2, §8.2, §10: they must not be chosen arbitrarily,
    because they set the rotation frequency of the 1X/2X features)."""


class FeatureContractError(ValueError):
    """Raised when a batch file or raw signal file breaks the input contract: missing feature
    columns, unreadable file, or a window_index that the CSV does not have."""
