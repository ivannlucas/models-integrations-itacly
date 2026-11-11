"""Writable Numba cache dir for the plugins whose dependencies JIT-compile with ``cache=True``.

librosa (ml41) decorates functions such as ``librosa.core.notation.__o_fold`` with
``@numba.jit(cache=True)``. Numba then needs somewhere to write the compiled cache and tries, in
order: a ``__pycache__`` next to the source file (``site-packages``, owned by root in the image)
and the user's ``~/.cache`` (the image runs as ``luceit``, created with ``--no-create-home``).
With neither writable, the first librosa call fails with::

    cannot cache function '__o_fold': no locator available for file '.../librosa/core/notation.py'

which broke ml41 predict (inline and batch) and train in the deployed container. The fix is to
point ``NUMBA_CACHE_DIR`` at a writable temp dir. The Dockerfile already sets it; this helper is
the in-process fallback for environments that do not (another image, a local run as a
restricted user). Numba reads the env var once when ``numba`` is first imported, and another
plugin's dependency (e.g. shap) may have imported it already, so the live config is patched too.
Idempotent; call it before any librosa/numba use. An explicit ``NUMBA_CACHE_DIR`` always wins.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def numba_cache_dir() -> Path:
    """Default writable Numba cache dir, outside site-packages and the user's home."""
    return Path(tempfile.gettempdir()) / "numba_cache"


def configure_numba_cache() -> str:
    """Make sure Numba has a writable cache dir; return the dir in use."""
    cache_dir = os.environ.get("NUMBA_CACHE_DIR") or str(numba_cache_dir())
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    numba = sys.modules.get("numba")
    if numba is not None:
        # Already imported: its config module has read the env var already.
        numba.config.CACHE_DIR = cache_dir
    return cache_dir
