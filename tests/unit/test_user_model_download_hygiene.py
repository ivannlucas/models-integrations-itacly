"""Regression tests for how plugins handle user models downloaded from MLflow.

1. No predict*/stats method may assign attributes on ``self`` (the shared plugin instance).
   Swapping a user's model into ``self`` for the duration of a request leaks it to concurrent
   requests (see test_user_model_isolation.py). This structural check also covers plugins
   added in the future, which the behavioural isolation tests only cover once listed.
2. Every ``download_*_from_mlflow`` helper must delete its temp dir when the run has no usable
   model. The None it returns becomes a UserModelUnavailableError (422), so without the
   cleanup every retry with an empty run leaves an orphan dir (possibly with partial
   downloads) and the pod's disk slowly fills up.
"""
import ast
import importlib
import inspect
import pathlib
import tempfile

import pytest

from app.domain.services.exceptions import UserModelUnavailableError

PLUGINS_DIR = pathlib.Path(__file__).resolve().parents[2] / "app" / "plugins"
# Runtime counters are per-instance bookkeeping, not model state.
ALLOWED_SELF_ATTRS = ("_predict_count", "_last_predict", "_total_latency", "_n_", "_runtime", "_stats")


def _self_assignments(fn: ast.FunctionDef) -> list[str]:
    found = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for sub in ast.walk(target):
                if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self" and not sub.attr.startswith(ALLOWED_SELF_ATTRS)):
                    found.append(f"self.{sub.attr} (línea {node.lineno})")
    return found


@pytest.mark.parametrize("plugin_file", sorted(PLUGINS_DIR.glob("*/plugin.py")), ids=lambda p: p.parent.name)
def test_predict_and_stats_never_assign_model_state_on_self(plugin_file):
    tree = ast.parse(plugin_file.read_text(encoding="utf-8"))
    offenders = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            if fn.name.startswith("predict") or fn.name == "stats":
                offenders += [f"{fn.name}: {a}" for a in _self_assignments(fn)]
    assert not offenders, (
        "predict/stats must resolve the user's model into locals, never into self "
        f"(shared across concurrent requests): {offenders}"
    )


def _download_helpers():
    for path in sorted(PLUGINS_DIR.glob("*/mlflow_utils.py")):
        module = importlib.import_module(f"app.plugins.{path.parent.name}.mlflow_utils")
        if not hasattr(module, "BaseMLflowTracker"):
            continue  # non-trainable plugins (ml28/ml31/ml33) never touch MLflow
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("download_") and fn.__module__ == module.__name__:
                yield pytest.param(module, name, id=f"{path.parent.name}.{name}")


@pytest.mark.parametrize("empty_artifact", [False, True], ids=["no_artifact", "incomplete_artifact"])
@pytest.mark.parametrize("module,func_name", list(_download_helpers()))
def test_download_cleans_temp_dir_when_run_has_no_model(module, func_name, empty_artifact, monkeypatch):
    created = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    class EmptyRunTracker:
        def __init__(self, run_id=""):
            self.run_id = run_id

        def download_artifacts(self, dest_dir, artifact_path=""):
            # "" → MLflow returned nothing; dest_dir → the run exists but lacks the model files.
            return dest_dir if empty_artifact else ""

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)
    monkeypatch.setattr(module, "BaseMLflowTracker", EmptyRunTracker)

    with pytest.raises(UserModelUnavailableError):
        getattr(module, func_name)("run-without-model")

    assert created, f"{func_name} did not create a temp dir — test no longer exercises it"
    leaked = [p for p in created if pathlib.Path(p).exists()]
    assert not leaked, f"{func_name} left temp dirs behind: {leaked}"
