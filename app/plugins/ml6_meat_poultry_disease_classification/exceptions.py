"""Plugin-local domain errors for ml6 (mapped to HTTP 422 via registry.extra_predict_exceptions).

Kept inside the plugin instead of app/domain/services/exceptions.py because this integration was
scoped to not modify app/domain (same approach as ml24 and ml26); they can be promoted to the
domain module in a later refactor.
"""


class UnknownImageModalityError(ValueError):
    """Raised when the request does not say which ensemble to use, or names an unknown one.

    The two ensembles take different image modalities (thermal body images vs. RGB fecal
    images) and are never interchangeable, so there is no safe default: ``model_key`` must be
    ``"thermal"`` or ``"rgb"``.
    """
