"""Plugin-local input validation error for ml26 (mapped to HTTP 422 via
registry.extra_predict_exceptions).

Kept inside the plugin instead of app/domain/services/exceptions.py because this integration was
scoped to not modify app/domain; it can be promoted to the domain module in a later refactor.
"""


class InvalidWineryInputError(ValueError):
    """Raised when lot/readings data violate the input contract (missing columns, unknown
    lot_id, unsupported wine_type/stage, malformed feature_window)."""
