"""Runtime bookkeeping shared by the model plugins (loaded check and prediction counters)."""
from __future__ import annotations

from datetime import datetime, timezone

from app.application.dto.stats_dto import RuntimeStats
from app.domain.services.exceptions import ModelNotLoadedError

NOT_LOADED_MESSAGE = "El modelo no está cargado."


class PluginRuntimeMixin:  # pylint: disable=too-few-public-methods  # only private helpers
    """Mixin for ``ModelPluginPort`` plugins: ``_require_loaded``, ``_record``, runtime stats.

    It relies on the plugin's own ``is_loaded()``. The counters start at their class defaults,
    so plugins don't have to initialise them in ``__init__``.
    """

    _predict_count: int = 0
    _last_predict_at: str | None = None

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
