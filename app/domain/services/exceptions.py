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
