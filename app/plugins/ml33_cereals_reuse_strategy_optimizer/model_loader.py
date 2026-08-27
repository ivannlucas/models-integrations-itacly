"""No-artifact loader for ml33 — the deployed engine is a self-contained MILP solver.

Kept for interface parity with other plugins (the plugin-integration skill expects a
``model_loader.py``), even though there is nothing to load from ``ArtifactStore``/S3: the
engine (``app.plugins.ml33_cereals_reuse_strategy_optimizer.optimizer``) has no serialized
weights, no reference data files and no random seed — it is pure, deterministic code
(see constants.py / manifest.yaml ``artifacts: {folder: null}``).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_artifacts() -> bool:
    """No-op: nothing to load from disk/S3.

    Returns True so ``plugin.load()``/``is_loaded()`` stay uniform with plugins that do
    load artifacts through ``ArtifactStore``.
    """

    logger.info("ml33 reuse-strategy optimizer has no artifacts to load (self-contained MILP solver).")
    return True
