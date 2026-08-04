"""Custom exceptions for model runtime service."""


class ModelNotLoadedError(RuntimeError):
    """Raised when prediction is attempted before the model is loaded."""


class TrainingNotSupportedError(NotImplementedError):
    """Raised when the /train endpoint is called (always 501)."""


class InsufficientDataError(ValueError):
    """Raised when the input time series is too short to compute features (wine price)."""


class UnsupportedProductError(ValueError):
    """Raised when the requested product has no trained model (cereal forecast)."""


class InsufficientRowsError(ValueError):
    """Raised when not enough rows survive dropna() after lag feature construction (meat forecast)."""


class InvalidImageError(ValueError):
    """Raised when the uploaded file cannot be decoded as a valid image."""


class InvalidVideoError(ValueError):
    """Raised when the video file cannot be opened or is unreadable."""


class InsufficientFramesError(ValueError):
    """Raised when fewer than clip_length frames are provided for inline inference."""


class NoValidSimulationPointError(ValueError):
    """Raised when no simulation point satisfies the operational constraints (wine sulphite)."""


class PuConstraintViolationError(ValueError):
    """Raised when the requested setpoints violate the PU ≥ 13 food-safety constraint (ml35)."""


class InsufficientTelemetryHistoryError(ValueError):
    """Raised when fewer than seq_len valid rows of telemetry history are provided (ml46)."""



class InsufficientCycleHistoryError(ValueError):
    """Raised when a cycle (run_id) has fewer minutes of history than the lag features need (ml40)."""


class UnknownDiagnosisSystemError(ValueError):
    """Raised when the input columns match neither refrigeracion nor aireado contracts (ml40)."""

class ThermalSafetyViolationError(ValueError):
    """Raised when the GA cannot find a feasible solution meeting T_out >= 72.3 °C (ml34)."""


class InsufficientSequenceHistoryError(ValueError):
    """Raised when no sample_id supplies the 48 consecutive hourly observations a window needs (ml9).

    The delivered pipeline runs with pad_short_sequences=false, so a series shorter than
    window_size produces no window at all and the model would silently return zero predictions.
    This turns that silence into an explicit 422 (see inbox/a09/manifest.yaml
    constraints.historial_minimo).
    """


class InfeasibleOptimizationError(ValueError):
    """Raised when the LP crop-allocation problem is infeasible/unbounded (ml31).

    Combinations of constraints that are too strict (e.g. a very high min_benefit
    together with a narrow ±surface band) yield CBC status != OPTIMAL. The plugin
    translates this to a domain error (HTTP 422) instead of returning an empty plan.
    """

