"""Runtime bookkeeping shared by the model plugins (loaded check and prediction counters)."""
from __future__ import annotations

from datetime import datetime, timezone

from app.application.dto.stats_dto import RuntimeStats
from app.domain.services.exceptions import ModelNotLoadedError

NOT_LOADED_MESSAGE = "El modelo no está cargado."


class PluginRuntimeMixin:
    """Mixin for ``ModelPluginPort`` plugins: ``_require_loaded``, ``_record``, runtime stats.

    It relies on the plugin's own ``is_loaded()``. The counters start at their class defaults,
    so plugins don't have to initialise them in ``__init__``; they are read through the
    ``predict_count`` / ``last_predict_at`` properties.
    """

    _predict_count: int = 0
    _last_predict_at: str | None = None

    @property
    def predict_count(self) -> int:
        """Predictions served by this plugin instance since it was created."""
        return self._predict_count

    @property
    def last_predict_at(self) -> str | None:
        """UTC ISO-8601 time of the last served prediction (None before the first one)."""
        return self._last_predict_at

    def _require_loaded(self) -> None:
        """Raise ``ModelNotLoadedError`` if the plugin has not loaded its model."""
        if not self.is_loaded():
            raise ModelNotLoadedError(NOT_LOADED_MESSAGE)

    def _record(self) -> None:
        """Count one served prediction and remember when it happened (UTC, ISO 8601)."""
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _runtime_stats(self) -> RuntimeStats:
        """Runtime counters for ``stats()`` (no latency is measured by default)."""
        return RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None)

    # ── optional latency tracking (plugins that measure each prediction) ─────────────────────

    _total_latency_ms: float = 0.0

    def _record_latency(self, latency_ms: float) -> None:
        """Count one served prediction (``_record``) that took ``latency_ms`` milliseconds."""
        self._total_latency_ms += latency_ms
        self._record()

    def _avg_latency_ms(self, ndigits: int | None = None) -> float | None:
        """Mean latency of the served predictions (None before the first one).

        ``ndigits`` rounds it as the plugin's /stats always did (None: not rounded).
        """
        if not self._predict_count:
            return None
        avg = self._total_latency_ms / self._predict_count
        return avg if ndigits is None else round(avg, ndigits)
