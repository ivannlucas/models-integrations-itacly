"""Per-row error capture shared by the plugins' batch predictions."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@contextmanager
def row_error_capture(
    rows: list[dict[str, Any]], error_row: dict[str, Any], *, what: str
) -> Iterator[None]:
    """Turn a failure while predicting one row into an error row instead of failing the batch.

    Usage, inside the batch loop::

        with row_error_capture(predictions, {"row": int(idx)}, what=f"fila {idx}"):
            predictions.append(predict_one(row))

    If the block raises, ``{**error_row, "error": str(exc)}`` is appended to ``rows`` (same
    shape the plugins used to build by hand), a warning is logged, and the loop goes on. Any
    exception counts: a batch from the platform must return the good rows even when one row
    hits an unexpected case (bad value, model error…).
    """
    try:
        yield
    except Exception as exc:  # pylint: disable=broad-exception-caught  # one bad row never fails the batch
        logger.warning("Error en %s: %s", what, exc)
        rows.append({**error_row, "error": str(exc)})
