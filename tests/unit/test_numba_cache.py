"""Tests for app.domain.services.numba_cache (writable Numba cache dir for librosa/ml41)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numba
import pytest

from app.domain.services.numba_cache import configure_numba_cache, numba_cache_dir

# pylint: disable=no-member  # numba.config builds CACHE_DIR & co. dynamically from the env

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(name="restore_numba")
def fixture_restore_numba(monkeypatch):
    """Keep the process-wide Numba config untouched after each test."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", numba.config.CACHE_DIR)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)


@pytest.mark.usefixtures("restore_numba")
def test_defaults_to_temp_dir_and_patches_live_config():
    """Without NUMBA_CACHE_DIR it uses the temp dir, and patches an already-imported numba."""
    cache_dir = configure_numba_cache()

    assert cache_dir == str(numba_cache_dir())
    assert os.environ["NUMBA_CACHE_DIR"] == cache_dir
    assert numba.config.CACHE_DIR == cache_dir
    assert Path(cache_dir).is_dir()


@pytest.mark.usefixtures("restore_numba")
def test_explicit_env_var_wins(monkeypatch, tmp_path):
    """An explicit NUMBA_CACHE_DIR (the Dockerfile's) is kept as is."""
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(explicit))

    assert configure_numba_cache() == str(explicit)
    assert numba.config.CACHE_DIR == str(explicit)
    assert explicit.is_dir()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root POSIX user")
def test_cache_true_jit_works_without_writable_source_dir_or_home(tmp_path):
    """Reproduce the container failure (root-owned source dir, no home) and check the fix.

    Runs in a subprocess so numba is imported fresh, like in the service.
    """
    src_dir = tmp_path / "ro"
    src_dir.mkdir()
    (src_dir / "jitmod.py").write_text(textwrap.dedent("""
        from numba import jit

        @jit(nopython=True, cache=True)
        def add_one(x):
            return x + 1
    """))
    src_dir.chmod(0o555)
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    env.update(HOME="/nonexistent", TMPDIR=str(tmp_path))

    def run(code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code], cwd=_REPO_ROOT, env=env,
            capture_output=True, text=True, check=False,
        )

    import_jitmod = f"import sys; sys.path.insert(0, {str(src_dir)!r}); import jitmod;"
    try:
        broken = run(import_jitmod + " print(jitmod.add_one(1))")
        fixed = run(
            "from app.domain.services.numba_cache import configure_numba_cache;"
            " configure_numba_cache(); " + import_jitmod + " print(jitmod.add_one(1))"
        )
    finally:
        src_dir.chmod(0o755)

    assert broken.returncode != 0
    assert "no locator available" in broken.stderr
    assert fixed.returncode == 0, fixed.stderr
    assert fixed.stdout.strip() == "2"
