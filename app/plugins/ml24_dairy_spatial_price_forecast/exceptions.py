"""Plugin-local domain errors for ml24 (mapped to HTTP 422 via registry.extra_predict_exceptions).

Kept inside the plugin instead of app/domain/services/exceptions.py because this integration was
scoped to not modify app/domain (same approach as ml26); they can be promoted to the domain
module in a later refactor.
"""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8


class SeriesContractError(ValueError):
    """Raised when the long-format series does not meet the inference contract of the memoria
    (section 8.4): missing columns, missing anchor/context products, or the anchor product without
    the target metric 'PRECIO MEDIO kg o litros'."""


class UnknownCcaaError(ValueError):
    """Raised when a requested CCAA code/name is not one of the 17 CCAA the model serves
    (T.ESPANA/ESP is excluded because the official panels use include_spain=false)."""


class InsufficientPriceHistoryError(ValueError):
    """Raised when no requested CCAA has the 13 consecutive monthly anchor prices (ending at its
    last period) that the lag_12 features need. CCAA below the minimum are omitted with a warning,
    as the memoria (section 8.4) promises, instead of being predicted from median-imputed values."""
