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
