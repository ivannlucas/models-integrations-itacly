# Reentreno que actualiza el modelo base — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que cada `/train` correcto de ml13, 14, 15, 16, 17, 18, 21 y 23 publique el modelo reentrenado como nueva versión del modelo base (`versions/<vN>/` + `current.json`) y el pod lo sirva al momento, con rollback.

**Architecture:** `ArtifactStore` aprende versiones (puntero `current.json`, versión activa por carpeta compartida en el proceso). Un helper común `persist_retrained_bundle` sube el bundle al run de MLflow (opcional, fallo = aviso) y lo publica como versión. `ModelContainer` envuelve el `/train` de los plugins que declaran `BASE_ARTIFACT_FOLDER`: lock, entrena, crea una instancia nueva del plugin, la carga y la sustituye; si no carga, restaura el puntero. `POST /rollback` mueve el puntero a la versión anterior y recarga.

**Tech Stack:** Python 3.10, FastAPI, pydantic v2, boto3 (S3), MLflow client, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-reentreno-actualiza-base-design.md`

## Global Constraints

- Rama `feature/retrain-updates-base-model`, en la carpeta principal del repo (sin worktrees).
- **Sin commits**: el usuario hace los commits. El último paso de cada tarea deja el árbol listo y muestra `git diff --ignore-cr-at-eol --stat`.
- El árbol de trabajo está en CRLF con índice LF: al editar ficheros existentes conservar sus fines de línea; revisar con `git diff --ignore-cr-at-eol`.
- Tests: `python3 -m pytest -p no:flask <ruta> -q` (el `-p no:flask` es obligatorio en este WSL).
- Solo los 8 modelos promueven su base: `ml13-wine-price-fluctuation-prediction`, `ml14-wine-phyto-price-forecast`, `ml15-wine-ipi-price-forecast`, `ml16-meat-raw-material-price-alert`, `ml17-meat-market-price-analysis`, `ml18-meat-spatial-price-forecast`, `ml21-cereals-price-spatial`, `ml23-lactic-market-price-forecast`. El resto de modelos no cambia de comportamiento.
- `v0` = ficheros actuales en la raíz de `artifacts/fixed/<carpeta>/`; nunca se mueven ni se borran. Nada se borra nunca.
- `current.json` se escribe **el último** al publicar. Número de versión nueva = máximo existente en `versions/` + 1.
- Un fallo de MLflow en `/train` no aborta: `upload_warning` y la base se publica igual. Un fallo de publicación → `ModelPersistenceError` (502), la base no cambia.
- `mlflow_run_id` es opcional (`str = ""`) en el `TrainRequest` de los 8 modelos.
- Mensajes al usuario en español, como el resto del repo.
- **Antes de la Tarea 1**, una sola vez, guardar los fallos preexistentes de la suite: `python3 -m pytest -p no:flask tests/unit -q -rf 2>&1 | grep '^FAILED' | sort > /tmp/claude-1000/-mnt-c-Users-PC-Desktop-Repositorios-models-integrations-itacly/0cabd567-c354-4897-9d14-75abf4a53287/scratchpad/baseline-failures.txt` (la Tarea 14 compara contra ella; sin `git stash`, el árbol tiene cambios CRLF).

## Review Focus

1. **`v0` en S3 con versiones ya publicadas**: listar `artifacts/fixed/<carpeta>/` devuelve también `versions/**` y `current.json`. Descargar la v0 no debe copiarlos a la raíz local ni heredarlos en una versión nueva. Test en la Tarea 2 (`test_v0_download_ignores_versions_tree_and_pointer`).
2. **Excepción en `train()` después de publicar**: si `train()` publica y luego falla al construir la respuesta, el puntero no puede quedarse apuntando a una versión que el pod no sirve. Test en la Tarea 4 (`test_train_failing_after_publish_restores_pointer`).
3. **Métricas con `NaN`** (ml21 rellena `auc` de regresión con `nan`): `version.json` tiene que ser JSON válido. Test en la Tarea 3 (`test_non_finite_metrics_become_null`).
4. **Rollback cuando la versión anterior ya no carga**: el puntero vuelve a la vigente y el pod sigue sirviendo. Test en la Tarea 4 (`test_rollback_to_a_version_that_does_not_load_keeps_current`).
5. **Tests que entrenan de verdad sin aislar el store**: cualquier `train()` real de los tests escribiría en `artifacts/` del repo. Lo evita la fixture autouse de la Tarea 3, y su test comprueba que el store de pruebas no apunta a `artifacts/`.

---

## Estructura de ficheros

| Fichero | Responsabilidad |
|---|---|
| `app/domain/services/exceptions.py` (mod) | `BaseVersionNotFoundError`, `NoPreviousBaseVersionError`, `BaseReloadError` |
| `app/application/dto/base_version_dto.py` (nuevo) | `VersionInfo`, `RollbackRequest`, `RollbackResponse` |
| `app/application/dto/stats_dto.py` (mod) | `StatsResponse.base_version` |
| `app/infrastructure/artifact_store.py` (mod) | Versiones: puntero, versión activa, publicar, activar, leer `version.json` |
| `app/infrastructure/base_model_publisher.py` (nuevo) | `persist_retrained_bundle` + `_store_for` (punto de inyección en tests) |
| `app/infrastructure/http/dependencies/container.py` (mod) | Promoción, recarga con lock, rollback |
| `app/application/use_cases/get_stats_use_case.py` (mod) | Añade `base_version` a `/stats` |
| `app/infrastructure/http/router_factory.py` (mod) | `POST /rollback` |
| `app/plugins/<8 modelos>/{plugin,train_dto}.py` (mod) | `BASE_ARTIFACT_FOLDER`, `train()` vía helper, DTOs |
| `app/plugins/ml14_wine_phyto_price_forecast/model_loader.py` (mod) | Base con `scalers.json` |
| `tests/unit/fake_s3.py` (nuevo) | Cliente S3 en memoria |
| `tests/unit/tiny_promoting_plugin.py` (nuevo) | Plugin mínimo que promueve, para contenedor y router |
| `tests/conftest.py` (mod) | Fixture autouse `isolated_base_store` |
| `CLAUDE.md`, `.claude/skills/plugin-integration/SKILL.md`, `README.md` (mod) | Regla y endpoints |

---

### Task 1: Excepciones y DTOs de versión

**Files:**
- Modify: `app/domain/services/exceptions.py` (añadir al final)
- Create: `app/application/dto/base_version_dto.py`
- Modify: `app/application/dto/stats_dto.py`
- Test: `tests/unit/test_base_version_dto.py`

**Interfaces:**
- Produces: `BaseVersionNotFoundError(LookupError)`, `NoPreviousBaseVersionError(RuntimeError)`, `BaseReloadError(RuntimeError)`; `VersionInfo(version: str, parent: str | None, created_at: str | None, mlflow_run_id: str | None, data_path: str | None, metrics: dict[str, Any])`; `RollbackRequest(version: str | None)`; `RollbackResponse(base_version: VersionInfo, previous_base_version: VersionInfo)`; `StatsResponse.base_version: VersionInfo | None = None`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_base_version_dto.py`:

```python
"""DTOs and exceptions of the versioned base model."""
from app.application.dto.base_version_dto import RollbackRequest, RollbackResponse, VersionInfo
from app.application.dto.stats_dto import RuntimeStats, StatsResponse
from app.domain.services.exceptions import (
    BaseReloadError,
    BaseVersionNotFoundError,
    NoPreviousBaseVersionError,
)


def test_version_info_defaults_describe_v0():
    info = VersionInfo(version="v0")
    assert info.parent is None and info.mlflow_run_id is None and info.metrics == {}


def test_rollback_dtos():
    assert RollbackRequest().version is None
    resp = RollbackResponse(base_version={"version": "v1", "parent": "v0"},
                            previous_base_version={"version": "v2", "parent": "v1"})
    assert resp.base_version.version == "v1" and resp.previous_base_version.parent == "v1"


def test_stats_response_base_version_is_optional():
    base = dict(model_name="m", version="1", description="d", task_type="t", framework="f",
                inputs=[], outputs=[], metrics={}, runtime_stats=RuntimeStats(total_predictions=0))
    assert StatsResponse(**base).base_version is None
    assert StatsResponse(**base, base_version={"version": "v3"}).base_version.version == "v3"


def test_exception_hierarchy():
    assert issubclass(BaseVersionNotFoundError, LookupError)
    assert not issubclass(BaseVersionNotFoundError, ValueError)  # /train maps ValueError to 400
    assert issubclass(NoPreviousBaseVersionError, RuntimeError)
    assert issubclass(BaseReloadError, RuntimeError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -p no:flask tests/unit/test_base_version_dto.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.application.dto.base_version_dto'`

- [ ] **Step 3: Implement**

Añadir al final de `app/domain/services/exceptions.py`:

```python
class BaseVersionNotFoundError(LookupError):
    """The requested base model version does not exist in the artifact store (→ 404)."""


class NoPreviousBaseVersionError(RuntimeError):
    """Rollback asked with no target and the current base is v0, which has no parent (→ 409)."""


class BaseReloadError(RuntimeError):
    """The newly activated base version could not be loaded; the pointer was restored (→ 500)."""
```

`app/application/dto/base_version_dto.py`:

```python
"""DTOs for the versioned base model (versions/<vN>/ + current.json in the artifact store)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VersionInfo(BaseModel):
    """One published base version, as stored in its version.json (v0 = the delivered artifact)."""

    model_config = ConfigDict(protected_namespaces=())
    version: str = Field(..., description="Versión del modelo base: v0 (entregado), v1, v2…")
    parent: str | None = Field(default=None, description="Versión de la que hereda (None en v0)")
    created_at: str | None = Field(default=None, description="Fecha de publicación (ISO 8601, UTC)")
    mlflow_run_id: str | None = Field(default=None, description="Run de MLflow del reentreno, si lo hubo")
    data_path: str | None = Field(default=None, description="Datos con los que se reentrenó")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Métricas de test del reentreno")


class RollbackRequest(BaseModel):
    """Body of POST /rollback. Without version, the base goes back to the current one's parent."""

    version: str | None = Field(default=None, description="Versión a activar (por defecto, la anterior)")


class RollbackResponse(BaseModel):
    """Result of POST /rollback."""

    base_version: VersionInfo
    previous_base_version: VersionInfo
```

En `app/application/dto/stats_dto.py`, añadir el import tras `from pydantic import BaseModel, ConfigDict`:

```python
from app.application.dto.base_version_dto import VersionInfo
```

y el campo al final de `StatsResponse`:

```python
    runtime_stats: RuntimeStats
    base_version: Optional[VersionInfo] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest -p no:flask tests/unit/test_base_version_dto.py -q`
Expected: 4 passed

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat` (el usuario hace el commit)

---

### Task 2: `ArtifactStore` con versiones

**Files:**
- Modify: `app/infrastructure/artifact_store.py`
- Create: `tests/unit/fake_s3.py`
- Test: `tests/unit/test_artifact_store_versions.py`

**Interfaces:**
- Consumes: `BaseVersionNotFoundError` (Tarea 1).
- Produces (en `app.infrastructure.artifact_store`): constantes `BASE_VERSION = "v0"`, `POINTER_FILENAME = "current.json"`, `VERSION_INFO_FILENAME = "version.json"`, `VERSIONS_DIRNAME = "versions"`; dict de módulo `_ACTIVE_VERSIONS: dict[str, str]`; `ArtifactStore(model_name: str, root: Path | None = None, use_s3: bool | None = None)` con `local_dir` (carpeta de la versión activa), `path()`, `download_all_if_needed()` (como hoy, sobre la versión activa), `active_version() -> str`, `refresh() -> str`, `publish_version(bundle_dir: str | Path, info: dict) -> dict`, `set_active_version(version: str) -> None`, `version_exists(version: str) -> bool`, `version_info(version: str | None = None) -> dict`.
- `tests/unit/fake_s3.py`: `FakeS3` con `put(key, data: bytes)`, `get(key) -> bytes`, `keys() -> list[str]`, `fail_upload_on: str | None`.

- [ ] **Step 1: Write the fake S3 client**

`tests/unit/fake_s3.py`:

```python
"""In-memory stand-in for the boto3 S3 client calls ArtifactStore makes (single bucket)."""
from __future__ import annotations

import io
from pathlib import Path

from botocore.exceptions import ClientError


class FakeS3:
    """Supports list_objects_v2 pagination, download_file, upload_file, get_object, put_object."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_upload_on: str | None = None   # upload_file raises for keys ending with this

    # ── helpers for tests ──
    def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def keys(self) -> list[str]:
        return sorted(self.objects)

    # ── boto3 surface ──
    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        fake = self

        class _Paginator:
            def paginate(self, Bucket: str, Prefix: str):  # noqa: N803 — boto3 names
                contents = [{"Key": k, "Size": len(v)} for k, v in sorted(fake.objects.items())
                            if k.startswith(Prefix)]
                return [{"Contents": contents}] if contents else [{}]

        return _Paginator()

    def download_file(self, bucket: str, key: str, path: str) -> None:
        Path(path).write_bytes(self.objects[key])

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        if self.fail_upload_on and key.endswith(self.fail_upload_on):
            raise ConnectionError(f"s3 caído subiendo {key}")
        self.objects[key] = Path(path).read_bytes()

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket: str, Key: str, Body) -> dict:  # noqa: N803
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_artifact_store_versions.py`:

```python
"""Versioned base model in ArtifactStore: current.json + versions/<vN>/, never overwritten."""
import json

import pytest

import app.infrastructure.artifact_store as store_mod
from app.domain.services.exceptions import BaseVersionNotFoundError
from app.infrastructure.artifact_store import ArtifactStore
from tests.unit.fake_s3 import FakeS3

ROOT = "artifacts/fixed/m/"


@pytest.fixture(autouse=True)
def _fresh_active_versions(monkeypatch):
    monkeypatch.setattr(store_mod, "_ACTIVE_VERSIONS", {})


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(store_mod, "_build_s3_client", lambda: fake)
    monkeypatch.setenv("STORAGE_BUCKET", "xai")
    return fake


@pytest.fixture
def bundle(tmp_path):
    def _make(**files: bytes):
        d = tmp_path / f"bundle_{len(list(tmp_path.glob('bundle_*')))}"
        d.mkdir()
        for name, data in files.items():
            (d / name).write_bytes(data)
        return d
    return _make


def test_without_pointer_the_base_is_v0_at_the_root(s3, tmp_path):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    assert store.refresh() == "v0"
    assert store.local_dir == tmp_path / "cache" / "m"
    assert store.path("model.pkl").read_bytes() == b"old"
    assert store.version_info() == {"version": "v0", "parent": None, "created_at": None,
                                     "mlflow_run_id": None, "data_path": None, "metrics": {}}


def test_v0_download_ignores_versions_tree_and_pointer(s3, tmp_path):
    s3.put(ROOT + "model.pkl", b"old")
    s3.put(ROOT + "versions/v1/model.pkl", b"new")
    s3.put(ROOT + "versions/v1/version.json", b'{"version": "v1", "parent": "v0"}')
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.download_all_if_needed()   # active version is v0: no current.json was read
    local = tmp_path / "cache" / "m"
    assert (local / "model.pkl").read_bytes() == b"old"
    assert not (local / "versions").exists() and not (local / "current.json").exists()


def test_publish_inherits_the_parent_and_moves_the_pointer_last(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    s3.put(ROOT + "panel.csv", b"ctx")
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.refresh()

    info = store.publish_version(bundle(**{"model.pkl": b"new"}),
                                 {"mlflow_run_id": "run-1", "data_path": "s3://d.csv", "metrics": {"mae": 1.5}})

    assert info["version"] == "v1" and info["parent"] == "v0" and info["mlflow_run_id"] == "run-1"
    assert info["metrics"] == {"mae": 1.5} and info["created_at"]
    assert s3.get(ROOT + "versions/v1/model.pkl") == b"new"
    assert s3.get(ROOT + "versions/v1/panel.csv") == b"ctx"
    assert json.loads(s3.get(ROOT + "versions/v1/version.json"))["parent"] == "v0"
    assert json.loads(s3.get(ROOT + "current.json")) == {"version": "v1"}
    assert s3.get(ROOT + "model.pkl") == b"old"          # v0 untouched
    assert store.active_version() == "v1"
    assert store.path("model.pkl").read_bytes() == b"new"

    info2 = store.publish_version(bundle(**{"extra.json": b"{}"}), {"metrics": {}})
    assert info2["version"] == "v2" and info2["parent"] == "v1"
    assert s3.get(ROOT + "versions/v2/model.pkl") == b"new"   # inherited from v1
    assert ROOT + "versions/v2/version.json" in s3.keys()


def test_version_json_of_the_parent_is_not_inherited_as_an_artifact(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.publish_version(bundle(**{"model.pkl": b"a"}), {})
    store.publish_version(bundle(**{"model.pkl": b"b"}), {})
    assert json.loads(s3.get(ROOT + "versions/v2/version.json"))["version"] == "v2"


def test_failed_upload_keeps_the_pointer_and_the_next_publish_skips_the_number(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    s3.fail_upload_on = "version.json"            # every artifact uploads, version.json (last) fails
    with pytest.raises(ConnectionError):
        store.publish_version(bundle(**{"model.pkl": b"new"}), {})
    assert ROOT + "current.json" not in s3.keys()
    assert store.active_version() == "v0"
    assert ROOT + "versions/v1/model.pkl" in s3.keys()   # the partial folder stays

    s3.fail_upload_on = None
    assert store.publish_version(bundle(**{"model.pkl": b"new"}), {})["version"] == "v2"


def test_refresh_reads_the_pointer_and_stores_of_the_same_folder_share_it(s3, tmp_path):
    s3.put(ROOT + "current.json", b'{"version": "v1"}')
    s3.put(ROOT + "versions/v1/model.pkl", b"new")
    s3.put(ROOT + "versions/v1/version.json", b'{"version": "v1", "parent": "v0"}')
    first, second = ArtifactStore("m", root=tmp_path / "c"), ArtifactStore("m", root=tmp_path / "c")
    assert first.refresh() == "v1"
    assert second.active_version() == "v1"
    assert second.path("model.pkl").read_bytes() == b"new"
    assert second.local_dir == tmp_path / "c" / "m" / "versions" / "v1"
    assert ArtifactStore("m", root=tmp_path / "other").active_version() == "v0"


def test_set_active_version_and_version_info(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.publish_version(bundle(**{"model.pkl": b"a"}), {})
    store.publish_version(bundle(**{"model.pkl": b"b"}), {})

    store.set_active_version("v1")
    assert json.loads(s3.get(ROOT + "current.json")) == {"version": "v1"}
    assert store.version_info()["version"] == "v1"
    assert store.version_info("v2")["parent"] == "v1"
    store.set_active_version("v0")
    assert store.path("model.pkl").read_bytes() == b"old"
    with pytest.raises(BaseVersionNotFoundError, match="v9"):
        store.set_active_version("v9")
    with pytest.raises(BaseVersionNotFoundError):
        store.version_info("v9")
    assert store.version_exists("v0") and store.version_exists("v2") and not store.version_exists("v9")


def test_version_info_is_read_from_s3_when_the_version_is_not_cached(s3, tmp_path):
    s3.put(ROOT + "versions/v4/version.json", b'{"version": "v4", "parent": "v3", "metrics": {"f1": 0.7}}')
    store = ArtifactStore("m", root=tmp_path / "cache")
    assert store.version_info("v4")["metrics"] == {"f1": 0.7}


def test_local_mode_without_bucket(monkeypatch, tmp_path, bundle):
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    base = tmp_path / "m"
    base.mkdir()
    (base / "model.pkl").write_bytes(b"old")
    store = ArtifactStore("m", root=tmp_path)
    info = store.publish_version(bundle(**{"model.pkl": b"new"}), {})
    assert info["version"] == "v1"
    assert (base / "versions" / "v1" / "model.pkl").read_bytes() == b"new"
    assert json.loads((base / "current.json").read_text()) == {"version": "v1"}
    assert (base / "model.pkl").read_bytes() == b"old"

    store_mod._ACTIVE_VERSIONS.clear()  # pylint: disable=protected-access — simulates a restart
    assert ArtifactStore("m", root=tmp_path).refresh() == "v1"


def test_use_s3_false_ignores_the_bucket(s3, tmp_path, bundle):
    store = ArtifactStore("m", root=tmp_path, use_s3=False)
    store.publish_version(bundle(**{"model.pkl": b"new"}), {})
    assert s3.keys() == []
    assert (tmp_path / "m" / "versions" / "v1" / "model.pkl").exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_artifact_store_versions.py -q`
Expected: FAIL (`TypeError: ArtifactStore.__init__() got an unexpected keyword argument 'root'`)

- [ ] **Step 4: Implement**

En `app/infrastructure/artifact_store.py`:

1. Docstring del módulo: añadir al final, antes del `"""` de cierre:

```text
Versioned base model (only models whose train() publishes, see base_model_publisher.py)
--------------------------------------------------------------------------------------
s3://<STORAGE_BUCKET>/artifacts/fixed/<model_name>/current.json   {"version": "vN"}
s3://<STORAGE_BUCKET>/artifacts/fixed/<model_name>/versions/<vN>/  full bundle + version.json

Without current.json the base is "v0": the files at the root, exactly as before. A version is
published as a new folder and current.json is written last, so a failed publish never changes
the served model; nothing is ever overwritten or deleted.
```

2. Imports: añadir `import json`, `import re`, `import shutil`, `from datetime import datetime, timezone`, `from botocore.exceptions import ClientError` y `from app.domain.services.exceptions import BaseVersionNotFoundError`.

3. Tras `ARTIFACTS_ROOT = _REPO_ROOT / "artifacts"`:

```python
BASE_VERSION = "v0"
POINTER_FILENAME = "current.json"
VERSION_INFO_FILENAME = "version.json"
VERSIONS_DIRNAME = "versions"
_VERSION_RE = re.compile(r"^v(\d+)$")
# Top-level entries of the model folder that are versioning metadata, never v0 artifacts.
_RESERVED_TOP_LEVEL = {POINTER_FILENAME, VERSIONS_DIRNAME}

# Base version in use per model folder, shared by every ArtifactStore of that folder in this
# process (ml15 reads its store from two modules; the container and the publisher build their
# own). Keyed by the local base folder so stores with different roots (tests) never mix.
_ACTIVE_VERSIONS: dict[str, str] = {}
```

4. Sustituir la clase `ArtifactStore` entera por:

```python
class ArtifactStore:
    """Resolves and downloads artifact files for a single model.

    Usage::

        _store = ArtifactStore("ml25_wine_sulphites")
        path = _store.path("quality_rf.pkl")   # returns a resolved Path

    Paths resolve against the active base version (v0 unless the model publishes versions).
    """

    def __init__(self, model_name: str, root: Path | None = None, use_s3: bool | None = None) -> None:
        """Initialize the store for the given model name.

        root overrides ARTIFACTS_ROOT and use_s3=False ignores STORAGE_BUCKET (both for tests).
        """
        self._model_name = model_name
        self._local_dir = (root or ARTIFACTS_ROOT) / model_name
        self._use_s3 = use_s3

    # ── versions ──────────────────────────────────────────────────────────────

    def _bucket(self) -> str | None:
        if self._use_s3 is False:
            return None
        return os.environ.get("STORAGE_BUCKET") or None

    def _key(self) -> str:
        return str(self._local_dir)

    def active_version(self) -> str:
        """Return the base version this process serves for the model (v0 by default)."""
        return _ACTIVE_VERSIONS.get(self._key(), BASE_VERSION)

    def _set_active(self, version: str) -> None:
        _ACTIVE_VERSIONS[self._key()] = version

    def _version_dir(self, version: str) -> Path:
        if version == BASE_VERSION:
            return self._local_dir
        return self._local_dir / VERSIONS_DIRNAME / version

    def _remote_root(self) -> str:
        return f"artifacts/fixed/{self._model_name}/"

    def _remote_prefix(self, version: str) -> str:
        if version == BASE_VERSION:
            return self._remote_root()
        return f"{self._remote_root()}{VERSIONS_DIRNAME}/{version}/"

    def _read_json(self, relative: str) -> dict | None:
        """Read <model folder>/<relative> (S3 when configured, else local); None if absent."""
        bucket = self._bucket()
        if bucket:
            try:
                obj = _build_s3_client().get_object(Bucket=bucket, Key=self._remote_root() + relative)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                    return None
                raise
            return json.loads(obj["Body"].read().decode("utf-8"))
        local = self._local_dir / relative
        if not local.exists():
            return None
        return json.loads(local.read_text(encoding="utf-8"))

    def _write_pointer(self, version: str) -> None:
        body = json.dumps({"version": version})
        bucket = self._bucket()
        if bucket:
            _build_s3_client().put_object(Bucket=bucket, Key=self._remote_root() + POINTER_FILENAME,
                                          Body=body.encode("utf-8"))
        else:
            self._local_dir.mkdir(parents=True, exist_ok=True)
            (self._local_dir / POINTER_FILENAME).write_text(body, encoding="utf-8")

    def refresh(self) -> str:
        """Read current.json and make its version the active one (v0 if there is none)."""
        pointer = self._read_json(POINTER_FILENAME) or {}
        version = pointer.get("version", BASE_VERSION)
        self._set_active(version)
        logger.info("Model '%s' base version: %s", self._model_name, version)
        return version

    def _max_version(self) -> int:
        numbers = [0]
        bucket = self._bucket()
        if bucket:
            prefix = f"{self._remote_root()}{VERSIONS_DIRNAME}/"
            paginator = _build_s3_client().get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    match = _VERSION_RE.match(obj["Key"][len(prefix):].split("/", 1)[0])
                    if match:
                        numbers.append(int(match.group(1)))
        else:
            versions_dir = self._local_dir / VERSIONS_DIRNAME
            if versions_dir.exists():
                for child in versions_dir.iterdir():
                    match = _VERSION_RE.match(child.name)
                    if child.is_dir() and match:
                        numbers.append(int(match.group(1)))
        return max(numbers)

    def version_info(self, version: str | None = None) -> dict:
        """Return the version.json of *version* (the active one by default).

        Raises BaseVersionNotFoundError if that version was never published.
        """
        version = version or self.active_version()
        if version == BASE_VERSION:
            return {"version": BASE_VERSION, "parent": None, "created_at": None,
                    "mlflow_run_id": None, "data_path": None, "metrics": {}}
        local = self._version_dir(version) / VERSION_INFO_FILENAME
        if local.exists():
            return json.loads(local.read_text(encoding="utf-8"))
        data = self._read_json(f"{VERSIONS_DIRNAME}/{version}/{VERSION_INFO_FILENAME}")
        if data is None:
            raise BaseVersionNotFoundError(
                f"La versión '{version}' del modelo base de '{self._model_name}' no existe."
            )
        return data

    def version_exists(self, version: str) -> bool:
        """True for v0 and for every version with a published version.json."""
        try:
            self.version_info(version)
        except BaseVersionNotFoundError:
            return False
        return True

    def set_active_version(self, version: str) -> None:
        """Point current.json at an already published *version* and serve it from now on."""
        if not self.version_exists(version):
            raise BaseVersionNotFoundError(
                f"La versión '{version}' del modelo base de '{self._model_name}' no existe."
            )
        self._write_pointer(version)
        self._set_active(version)

    def publish_version(self, bundle_dir: str | Path, info: dict) -> dict:
        """Publish *bundle_dir* on top of the active version as a new version and activate it.

        The new folder holds the active version's files with the bundle's files over them, plus
        version.json (parent, mlflow_run_id, data_path, metrics). current.json is written last:
        if anything fails before, the served version does not change.
        """
        parent = self.active_version()
        bucket = self._bucket()
        if bucket:
            self._download_all()   # the parent must be complete locally to inherit its files
        new_version = f"v{self._max_version() + 1}"
        parent_dir = self._version_dir(parent)

        def _skip_metadata(directory: str, names: list[str]) -> list[str]:
            if Path(directory) != parent_dir:
                return []
            return [n for n in names if n in _RESERVED_TOP_LEVEL or n == VERSION_INFO_FILENAME]

        staging = Path(tempfile.mkdtemp(prefix=f"{self._model_name}_{new_version}_"))
        try:
            if parent_dir.exists():
                shutil.copytree(parent_dir, staging, dirs_exist_ok=True, ignore=_skip_metadata)
            shutil.copytree(bundle_dir, staging, dirs_exist_ok=True)
            version_info = {
                "version": new_version,
                "parent": parent,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mlflow_run_id": info.get("mlflow_run_id") or None,
                "data_path": info.get("data_path") or None,
                "metrics": info.get("metrics") or {},
            }
            (staging / VERSION_INFO_FILENAME).write_text(
                json.dumps(version_info, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if bucket:
                self._upload_dir(staging, self._remote_prefix(new_version))
            target = self._version_dir(new_version)
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(staging, target)   # local cache (the store itself without S3)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        self._write_pointer(new_version)
        self._set_active(new_version)
        logger.info("Model '%s' base version %s published (parent %s).",
                    self._model_name, new_version, parent)
        return version_info

    def _upload_dir(self, local_dir: Path, remote_prefix: str) -> None:
        """Upload every file of local_dir under remote_prefix, version.json last."""
        bucket = self._bucket()
        s3 = _build_s3_client()
        files = sorted(p for p in local_dir.rglob("*") if p.is_file())
        files.sort(key=lambda p: p == local_dir / VERSION_INFO_FILENAME)
        for file_path in files:
            key = remote_prefix + file_path.relative_to(local_dir).as_posix()
            s3.upload_file(str(file_path), bucket, key)

    # ── artifact paths ────────────────────────────────────────────────────────

    @property
    def local_dir(self) -> Path:
        """Return the local directory of the active base version's artifacts."""
        return self._version_dir(self.active_version())

    def path(self, filename: str) -> Path:
        """Return the local path to *filename*, downloading from S3 if needed.

        Raises FileNotFoundError if the file is absent and S3 is not configured.
        """
        local = self.local_dir / filename
        if not local.exists():
            if self._bucket():
                self._download_all()
            else:
                raise FileNotFoundError(
                    f"Artifact not found: {local}\n"
                    f"Either place the file there or set STORAGE_BUCKET to "
                    f"enable automatic download from S3."
                )
        return local

    def download_all_if_needed(self) -> None:
        """Download every file of the active version from S3, skipping up-to-date files.

        Raises EnvironmentError if STORAGE_BUCKET is not set.
        """
        if not self._bucket():
            raise EnvironmentError(
                "STORAGE_BUCKET is not set. Cannot download artifacts from S3."
            )
        self._download_all()

    # ── S3 download ───────────────────────────────────────────────────────────

    def _download_all(self) -> None:
        """Download all files of the active version from S3, skipping unchanged ones."""
        try:
            from tqdm import tqdm  # type: ignore[import]
        except ImportError:
            tqdm = None

        bucket = self._bucket()
        version = self.active_version()
        s3 = _build_s3_client()
        remote_prefix = self._remote_prefix(version)
        local_dir = self._version_dir(version)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Collect all remote objects under the version prefix
        paginator = s3.get_paginator("list_objects_v2")
        objects = [
            obj
            for page in paginator.paginate(Bucket=bucket, Prefix=remote_prefix)
            for obj in page.get("Contents", [])
        ]
        if version == BASE_VERSION:
            # v0 lives at the model root, next to current.json and versions/: those are not its files.
            objects = [
                obj for obj in objects
                if obj["Key"][len(remote_prefix):].split("/", 1)[0] not in _RESERVED_TOP_LEVEL
            ]
        remote_files = [obj for obj in objects if not _is_directory_marker(obj, objects)]

        if not remote_files:
            logger.warning("No files found in S3 for model: %s (version %s)", self._model_name, version)
            return

        # Filter to only files that need downloading
        pending = []
        for obj in remote_files:
            relative_path = obj["Key"][len(remote_prefix):]
            local_path = local_dir / relative_path
            if _file_needs_download(local_path, obj["Size"]):
                pending.append((obj["Key"], local_path))
            else:
                logger.info("Skipping (unchanged): %s", relative_path)

        if not pending:
            logger.info("All artifact files are up-to-date. No download needed.")
            return

        logger.info(
            "Downloading %d file(s) for model '%s' (version %s) to %s ...\n%s",
            len(pending),
            self._model_name,
            version,
            local_dir.resolve(),
            "\n".join(local_path.name for _, local_path in pending),
        )

        iterator = tqdm(pending, desc="Downloading", unit="file") if tqdm else pending
        for remote_key, local_path in iterator:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, remote_key, str(local_path))
            except Exception as exc:
                logger.error("Failed to download %s: %s", remote_key, exc)
                raise  # bubble up so the app doesn't start with missing artifacts

        logger.info("Artifact download complete for model '%s'.", self._model_name)
```

Nota: el test existente `tests/unit/test_artifact_store.py::test_download_skips_slashless_folder_markers_of_the_s3_proxy` hace `monkeypatch.setattr(store, "_local_dir", …)`. Sigue funcionando porque `_local_dir` sigue siendo la carpeta base y la clave de versión activa se calcula al vuelo a partir de ella.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_artifact_store_versions.py tests/unit/test_artifact_store.py tests/unit/test_infrastructure.py -q`
Expected: todos en verde

- [ ] **Step 6: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 3: Helper `persist_retrained_bundle` y aislamiento en tests

**Files:**
- Create: `app/infrastructure/base_model_publisher.py`
- Modify: `tests/conftest.py` (fixture autouse al final)
- Test: `tests/unit/test_base_model_publisher.py`

**Interfaces:**
- Consumes: `ArtifactStore(...).publish_version(bundle_dir, info) -> dict` (Tarea 2), `ModelPersistenceError`.
- Produces: `_store_for(artifact_folder: str) -> ArtifactStore` (lo usan también el contenedor y los tests); `PersistResult(base_version: dict, upload_warning: str | None)`; `persist_retrained_bundle(bundle_dir: str | Path, *, artifact_folder: str, data_path: str, metrics: dict, mlflow_run_id: str = "", upload_to_mlflow: Callable[[], None] | None = None) -> PersistResult`.
- Fixture `isolated_base_store` (autouse) en `tests/conftest.py`: devuelve el `Path` raíz del store de pruebas; `_store_for(folder)` → `ArtifactStore(folder, root=<tmp>, use_s3=False)`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_base_model_publisher.py`:

```python
"""persist_retrained_bundle: MLflow (optional, failure = warning) + publish as the new base."""
import json
from unittest.mock import MagicMock

import pytest

from app.domain.services.exceptions import ModelPersistenceError
from app.infrastructure import artifact_store, base_model_publisher
from app.infrastructure.base_model_publisher import persist_retrained_bundle


@pytest.fixture
def bundle(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "model.pkl").write_bytes(b"new")
    return str(d)


def test_the_test_store_never_points_at_the_repo_artifacts(isolated_base_store):
    store = base_model_publisher._store_for("m")  # pylint: disable=protected-access
    assert artifact_store.ARTIFACTS_ROOT not in store.local_dir.parents
    assert store.local_dir == isolated_base_store / "m"


def test_publishes_without_mlflow_run(bundle, isolated_base_store):
    upload = MagicMock()
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv",
                                      metrics={"mae": 2.0}, upload_to_mlflow=upload)
    upload.assert_not_called()
    assert result.upload_warning is None
    assert result.base_version["version"] == "v1" and result.base_version["mlflow_run_id"] is None
    assert result.base_version["data_path"] == "d.csv"
    assert (isolated_base_store / "m" / "versions" / "v1" / "model.pkl").read_bytes() == b"new"


def test_with_run_uploads_to_mlflow_and_records_the_run(bundle):
    upload = MagicMock()
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv", metrics={},
                                      mlflow_run_id="run-1", upload_to_mlflow=upload)
    upload.assert_called_once_with()
    assert result.base_version["mlflow_run_id"] == "run-1" and result.upload_warning is None


def test_mlflow_failure_is_a_warning_and_the_base_is_still_published(bundle):
    upload = MagicMock(side_effect=ConnectionError("mlflow caído"))
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv", metrics={},
                                      mlflow_run_id="run-1", upload_to_mlflow=upload)
    assert "mlflow caído" in result.upload_warning and "run-1" in result.upload_warning
    assert result.base_version["version"] == "v1"


def test_publish_failure_is_a_model_persistence_error(bundle, monkeypatch):
    store = MagicMock(publish_version=MagicMock(side_effect=OSError("s3 caído")))
    monkeypatch.setattr(base_model_publisher, "_store_for", lambda folder: store)
    with pytest.raises(ModelPersistenceError, match="s3 caído"):
        persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv", metrics={})


def test_non_finite_metrics_become_null(bundle, isolated_base_store):
    result = persist_retrained_bundle(bundle, artifact_folder="m", data_path="d.csv",
                                      metrics={"auc_h1": float("nan"), "mae": 1.0, "inf": float("inf"),
                                               "n": 3, "label": "x", "nested": {"a": 1}})
    assert result.base_version["metrics"] == {"auc_h1": None, "mae": 1.0, "inf": None, "n": 3, "label": "x"}
    raw = (isolated_base_store / "m" / "versions" / "v1" / "version.json").read_text()
    assert "NaN" not in raw and json.loads(raw)["metrics"]["auc_h1"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -p no:flask tests/unit/test_base_model_publisher.py -q`
Expected: FAIL (`ModuleNotFoundError: app.infrastructure.base_model_publisher` y fixture `isolated_base_store` no encontrada)

- [ ] **Step 3: Implement**

`app/infrastructure/base_model_publisher.py`:

```python
"""Persist a retrained bundle as the model's new base version.

Used by the train() of the models that declare BASE_ARTIFACT_FOLDER (ml13, 14, 15, 16, 17, 18,
21, 23): every successful /train becomes the base served without mlflow_run_id. The bundle is
also uploaded to the caller's MLflow run when one is given — traceability only: if that upload
fails the base is still published and the response carries upload_warning. If publishing fails,
ModelPersistenceError (→ 502) and the base does not change (see ArtifactStore.publish_version).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.domain.services.exceptions import ModelPersistenceError
from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistResult:
    """Outcome of persist_retrained_bundle."""

    base_version: dict
    upload_warning: str | None


def _store_for(artifact_folder: str) -> ArtifactStore:
    """The store a model's base versions are published to (tests swap it for a temp one)."""
    return ArtifactStore(artifact_folder)


def _json_safe(metrics: dict) -> dict[str, Any]:
    """Keep scalar metrics; NaN/inf become None so version.json stays valid JSON."""
    safe: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            safe[key] = None
        elif value is None or isinstance(value, (bool, int, float, str)):
            safe[key] = value
    return safe


def persist_retrained_bundle(
    bundle_dir: str | Path,
    *,
    artifact_folder: str,
    data_path: str,
    metrics: dict,
    mlflow_run_id: str = "",
    upload_to_mlflow: Callable[[], None] | None = None,
) -> PersistResult:
    """Upload *bundle_dir* to the MLflow run (if any) and publish it as the new base version.

    *upload_to_mlflow* is the plugin's own MLflow logging + upload, called only with a run id.
    The bundle must use the base artifact's canonical filenames; files it lacks are inherited
    from the current base version.
    """
    upload_warning = None
    if mlflow_run_id and upload_to_mlflow is not None:
        try:
            upload_to_mlflow()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("MLflow upload failed for run %s: %s", mlflow_run_id, exc)
            upload_warning = (
                f"El modelo no se ha podido guardar en el run de MLflow {mlflow_run_id} "
                f"({type(exc).__name__}: {exc}); sí se ha publicado como modelo base."
            )
    try:
        version = _store_for(artifact_folder).publish_version(bundle_dir, {
            "mlflow_run_id": mlflow_run_id or None,
            "data_path": data_path,
            "metrics": _json_safe(metrics),
        })
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Publishing the new base of '%s' failed: %s", artifact_folder, exc)
        raise ModelPersistenceError(
            f"El modelo reentrenado no se ha podido publicar como modelo base: {exc}"
        ) from exc
    return PersistResult(base_version=version, upload_warning=upload_warning)
```

Al final de `tests/conftest.py`:

```python
# ── Versioned base model: never publish into the repo's artifacts/ ─────────

@pytest.fixture(autouse=True)
def isolated_base_store(monkeypatch, tmp_path_factory):
    """train() of the models that promote their base publishes a new version. In tests it goes
    to a temp folder without S3, never to the repo's artifacts/ or the real bucket. The
    container uses the same _store_for, so promotion/rollback tests see the same store."""
    from app.infrastructure import artifact_store, base_model_publisher

    root = tmp_path_factory.mktemp("base_store")
    monkeypatch.setattr(artifact_store, "_ACTIVE_VERSIONS", {})
    monkeypatch.setattr(
        base_model_publisher, "_store_for",
        lambda folder: artifact_store.ArtifactStore(folder, root=root, use_s3=False),
    )
    return root
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_base_model_publisher.py tests/unit/test_artifact_store_versions.py -q`
Expected: todos en verde

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 4: Contenedor — promoción, recarga con lock y rollback

**Files:**
- Modify: `app/infrastructure/http/dependencies/container.py`
- Modify: `app/application/use_cases/get_stats_use_case.py`
- Create: `tests/unit/tiny_promoting_plugin.py`
- Test: `tests/unit/test_container_base_promotion.py`

**Interfaces:**
- Consumes: `base_model_publisher._store_for`, `persist_retrained_bundle` (Tarea 3); `ArtifactStore.active_version/refresh/set_active_version/version_info` (Tarea 2); excepciones y `VersionInfo` (Tarea 1).
- Produces: `ModelContainer.promotes_base: bool`, `ModelContainer.base_version() -> dict`, `ModelContainer.train(request) -> Any`, `ModelContainer.rollback(version: str | None = None) -> dict` (`{"base_version": dict, "previous_base_version": dict}`); `GetStatsUseCase(plugin, base_version_provider: Callable[[], dict] | None = None)`. Un plugin promueve si su clase define `BASE_ARTIFACT_FOLDER: str`.
- `tests/unit/tiny_promoting_plugin.py`: `TinyPromotingPlugin` (atributos de clase `fail_load: bool`, `fail_after_publish: bool`), `TinyTrainRequest`, `TinyTrainResponse`, `TinyPredictRequest`, `TinyPredictResponse`, `TINY_FOLDER = "tiny_model"`.

- [ ] **Step 1: Write the tiny plugin**

`tests/unit/tiny_promoting_plugin.py`:

```python
"""Minimal plugin that promotes its base on train() — exercises container and router wiring."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel

from app.application.dto.base_version_dto import VersionInfo
from app.application.dto.stats_dto import RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.infrastructure import base_model_publisher
from app.infrastructure.base_model_publisher import persist_retrained_bundle

TINY_FOLDER = "tiny_model"


class TinyTrainRequest(BaseModel):
    data_path: str
    mlflow_run_id: str = ""


class TinyTrainResponse(BaseModel):
    value: str
    base_version: VersionInfo | None = None
    upload_warning: str | None = None


class TinyPredictRequest(BaseModel):
    mode: str = "inline"


class TinyPredictResponse(BaseModel):
    value: str


class TinyPromotingPlugin(ModelPluginPort):
    """Serves the content of value.txt of the active base version ("base" in v0)."""

    BASE_ARTIFACT_FOLDER = TINY_FOLDER
    fail_load = False
    fail_after_publish = False

    def __init__(self) -> None:
        self.value: str | None = None

    def load(self) -> None:
        if TinyPromotingPlugin.fail_load:
            raise RuntimeError("bundle roto")
        local = base_model_publisher._store_for(TINY_FOLDER).local_dir / "value.txt"  # pylint: disable=protected-access
        self.value = local.read_text() if local.exists() else "base"

    def is_loaded(self) -> bool:
        return self.value is not None

    def predict_inline(self, *, features: dict, model_key=None, threshold=None, mlflow_run_id: str = ""):
        return TinyPredictResponse(value=self.value)

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = ""):
        return TinyPredictResponse(value=self.value)

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        return StatsResponse(model_name="tiny", version="1", description="tiny", task_type="t",
                             framework="f", inputs=[], outputs=[], metrics={},
                             runtime_stats=RuntimeStats(total_predictions=0))

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TinyTrainResponse:
        bundle = Path(tempfile.mkdtemp(prefix="tiny_"))
        try:
            (bundle / "value.txt").write_text(data_path)
            result = persist_retrained_bundle(str(bundle), artifact_folder=TINY_FOLDER,
                                              data_path=data_path, metrics={"mae": 1.0},
                                              mlflow_run_id=mlflow_run_id)
        finally:
            shutil.rmtree(bundle, ignore_errors=True)
        if TinyPromotingPlugin.fail_after_publish:
            raise RuntimeError("fallo tras publicar")
        return TinyTrainResponse(value=data_path, base_version=result.base_version,
                                 upload_warning=result.upload_warning)
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_container_base_promotion.py`:

```python
"""ModelContainer promotes, reloads and rolls back the base of plugins with BASE_ARTIFACT_FOLDER."""
import pytest

from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.services.exceptions import (
    BaseReloadError,
    BaseVersionNotFoundError,
    NoPreviousBaseVersionError,
)
from app.infrastructure import base_model_publisher
from app.infrastructure.http.dependencies.container import ModelContainer
from tests.conftest import FakePlugin
from tests.unit.tiny_promoting_plugin import (
    TINY_FOLDER,
    TinyPredictRequest,
    TinyPromotingPlugin,
    TinyTrainRequest,
)


@pytest.fixture(autouse=True)
def _reset_tiny_flags():
    TinyPromotingPlugin.fail_load = False
    TinyPromotingPlugin.fail_after_publish = False
    yield
    TinyPromotingPlugin.fail_load = False
    TinyPromotingPlugin.fail_after_publish = False


@pytest.fixture
def container():
    c = ModelContainer(TinyPromotingPlugin())
    c.init()
    return c


def _store():
    return base_model_publisher._store_for(TINY_FOLDER)  # pylint: disable=protected-access


def _served(c) -> str:
    return c.predict_use_case.execute(TinyPredictRequest()).value


def test_init_reads_the_pointer_and_starts_at_v0(container):
    assert container.promotes_base
    assert container.base_version()["version"] == "v0"
    assert _served(container) == "base"


def test_train_publishes_and_serves_the_new_base_with_a_fresh_instance(container):
    old_plugin = container._plugin  # pylint: disable=protected-access
    resp = container.train_use_case.execute(TinyTrainRequest(data_path="datos-1"))
    assert resp.base_version.version == "v1"
    assert container._plugin is not old_plugin  # pylint: disable=protected-access
    assert old_plugin.value == "base"            # in-flight requests keep their instance
    assert _served(container) == "datos-1"
    assert container.base_version()["version"] == "v1"
    assert container.stats_use_case.execute().base_version.version == "v1"


def test_two_trains_chain_versions(container):
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    resp = container.train_use_case.execute(TinyTrainRequest(data_path="b"))
    assert resp.base_version.version == "v2" and resp.base_version.parent == "v1"
    assert _served(container) == "b"


def test_reload_failure_restores_the_pointer_and_keeps_serving(container):
    TinyPromotingPlugin.fail_load = True
    with pytest.raises(BaseReloadError, match="v1"):
        container.train_use_case.execute(TinyTrainRequest(data_path="roto"))
    assert _store().active_version() == "v0"
    assert _served(container) == "base"


def test_train_failing_after_publish_restores_pointer(container):
    TinyPromotingPlugin.fail_after_publish = True
    with pytest.raises(RuntimeError, match="fallo tras publicar"):
        container.train_use_case.execute(TinyTrainRequest(data_path="x"))
    assert _store().active_version() == "v0"
    assert _served(container) == "base"


def test_rollback_without_version_goes_to_the_parent(container):
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    container.train_use_case.execute(TinyTrainRequest(data_path="b"))
    result = container.rollback()
    assert result["base_version"]["version"] == "v1"
    assert result["previous_base_version"]["version"] == "v2"
    assert _served(container) == "a"
    # the next train inherits from the restored version
    resp = container.train_use_case.execute(TinyTrainRequest(data_path="c"))
    assert resp.base_version.version == "v3" and resp.base_version.parent == "v1"


def test_rollback_to_an_explicit_version_and_back_to_v0(container):
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    container.train_use_case.execute(TinyTrainRequest(data_path="b"))
    assert container.rollback("v0")["base_version"]["version"] == "v0"
    assert _served(container) == "base"


def test_rollback_from_v0_has_no_parent(container):
    with pytest.raises(NoPreviousBaseVersionError):
        container.rollback()


def test_rollback_to_an_unknown_version(container):
    with pytest.raises(BaseVersionNotFoundError, match="v9"):
        container.rollback("v9")
    assert _store().active_version() == "v0"


def test_rollback_to_a_version_that_does_not_load_keeps_current(container):
    container.train_use_case.execute(TinyTrainRequest(data_path="a"))
    TinyPromotingPlugin.fail_load = True
    with pytest.raises(BaseReloadError):
        container.rollback("v0")
    assert _store().active_version() == "v1"
    assert _served(container) == "a"


def test_plugins_without_base_folder_are_unchanged():
    plugin = FakePlugin(model_id="x", inline_factory=lambda *a, **k: None,
                        batch_factory=lambda *a, **k: None)
    c = ModelContainer(plugin)
    assert not c.promotes_base
    assert isinstance(c.train_use_case, TrainModelUseCase)
    c.init()
    assert c.stats_use_case.execute().base_version is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_container_base_promotion.py -q`
Expected: FAIL (`AttributeError: 'ModelContainer' object has no attribute 'promotes_base'`)

- [ ] **Step 4: Implement**

`app/application/use_cases/get_stats_use_case.py` (sustituir entero):

```python
"""Use case layer for retrieving model statistics."""
import logging
from typing import Callable

from app.application.dto.base_version_dto import VersionInfo
from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort


logger = logging.getLogger(__name__)


class GetStatsUseCase:
    """Use case for retrieving model statistics."""
    def __init__(
        self, plugin: ModelPluginPort, base_version_provider: Callable[[], dict] | None = None,
    ) -> None:
        """Initialize the use case with a model plugin.

        base_version_provider is set only for models that version their base (it fills
        StatsResponse.base_version); None leaves the plugin's stats untouched.
        """
        self._plugin = plugin
        self._base_version_provider = base_version_provider

    def execute(self, mlflow_run_id: str = "") -> StatsResponse:
        """Execute the use case and return model statistics from the plugin."""
        if mlflow_run_id:
            logger.info("Retrieving user-trained model stats from MLflow run_id=%s", mlflow_run_id)
        response = self._plugin.stats(mlflow_run_id=mlflow_run_id)
        if self._base_version_provider is not None:
            response = response.model_copy(
                update={"base_version": VersionInfo(**self._base_version_provider())}
            )
        return response
```

`app/infrastructure/http/dependencies/container.py` (sustituir entero):

```python
"""
Dependency injection container for model plugins.
Define aquí el ModelContainer, que se encarga de instanciar y cargar el plugin concreto,
y de proporcionar los casos de uso y el servicio runtime asociados.

Los plugins que declaran BASE_ARTIFACT_FOLDER (ml13, 14, 15, 16, 17, 18, 21, 23) publican cada
reentreno como nueva versión del modelo base. Para ellos el contenedor serializa /train y
/rollback con un lock, y tras activar una versión crea una instancia nueva del plugin, la carga
y la sustituye; si no carga, el puntero vuelve a la versión anterior y se sigue sirviendo la
instancia vieja.
"""
import logging
import threading
from typing import Any

from app.application.use_cases.get_stats_use_case import GetStatsUseCase
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import BaseReloadError, NoPreviousBaseVersionError
from app.domain.services.model_runtime_service import ModelRuntimeService
from app.infrastructure import base_model_publisher

logger = logging.getLogger(__name__)


class _PromotingTrainUseCase:
    """/train of a model that promotes its base: train, publish, reload (ModelContainer.train)."""

    def __init__(self, container: "ModelContainer") -> None:
        self._container = container

    def execute(self, request: Any) -> Any:
        """Delegate to the container, which owns the lock and the plugin swap."""
        return self._container.train(request)


class ModelContainer:
    """Generic DI container for any model plugin.

    Wires up use cases and the runtime service around a concrete ModelPluginPort.
    Each plugin returns its own typed Pydantic response models, so no response
    classes need to be injected here.
    """

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Recibe el plugin concreto y crea los casos de uso."""
        folder = getattr(type(plugin), "BASE_ARTIFACT_FOLDER", None)
        self._base_store = base_model_publisher._store_for(folder) if folder else None  # pylint: disable=protected-access
        self._lock = threading.Lock()
        self._wire(plugin)

    def _wire(self, plugin: ModelPluginPort) -> None:
        """(Re)build the service and use cases around *plugin*. Each attribute is replaced
        whole, so a request reads either the old or the new instance, never a mix."""
        self._plugin = plugin
        self._service = ModelRuntimeService(plugin)
        self.predict_use_case = PredictModelUseCase(plugin)
        self.stats_use_case = GetStatsUseCase(
            plugin, base_version_provider=self.base_version if self._base_store else None,
        )
        self.train_use_case = _PromotingTrainUseCase(self) if self._base_store else TrainModelUseCase(plugin)

    @property
    def promotes_base(self) -> bool:
        """True when every /train publishes a new base version (BASE_ARTIFACT_FOLDER)."""
        return self._base_store is not None

    def init(self) -> None:
        """Carga el plugin (si no se ha cargado ya)."""
        logger.info("Initializing container — loading plugin %s ...", type(self._plugin).__name__)
        if self._base_store is not None:
            self._base_store.refresh()
        self._plugin.load()
        logger.info("Plugin %s loaded successfully.", type(self._plugin).__name__)

    def base_version(self) -> dict:
        """version.json of the base version being served."""
        return self._base_store.version_info()

    def _reload_or_restore(self, previous: str) -> None:
        """Load a fresh plugin on the active version and swap it in; on failure, point the base
        back at *previous* and keep serving the current instance."""
        failed = self._base_store.active_version()
        try:
            fresh = type(self._plugin)()
            fresh.load()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._base_store.set_active_version(previous)
            logger.exception("Base version %s of %s does not load; back to %s",
                             failed, type(self._plugin).__name__, previous)
            raise BaseReloadError(
                f"La versión {failed} del modelo base no se ha podido cargar "
                f"({type(exc).__name__}: {exc}); el modelo base vuelve a {previous}."
            ) from exc
        self._wire(fresh)
        logger.info("Plugin %s now serves base version %s", type(fresh).__name__, failed)

    def train(self, request: Any) -> Any:
        """Train (the plugin publishes the new base) and serve the new version."""
        with self._lock:
            previous = self._base_store.active_version()
            try:
                response = TrainModelUseCase(self._plugin).execute(request)
            except Exception:
                # A train() that failed after publishing must not leave the pointer on a
                # version this pod does not serve.
                if self._base_store.active_version() != previous:
                    self._base_store.set_active_version(previous)
                raise
            self._reload_or_restore(previous)
            return response

    def rollback(self, version: str | None = None) -> dict:
        """Activate *version* (the current one's parent by default) and serve it."""
        with self._lock:
            current = self._base_store.version_info()
            target = version or current.get("parent")
            if not target:
                raise NoPreviousBaseVersionError(
                    "El modelo base vigente es v0: no hay una versión anterior a la que volver."
                )
            self._base_store.set_active_version(target)
            self._reload_or_restore(current["version"])
            return {"base_version": self._base_store.version_info(), "previous_base_version": current}

    @property
    def service(self) -> ModelRuntimeService:
        """Devuelve el servicio runtime asociado al plugin."""
        return self._service
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_container_base_promotion.py tests/unit/test_infrastructure.py -q`
Expected: todos en verde

- [ ] **Step 6: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 5: `POST /rollback` en el router

**Files:**
- Modify: `app/infrastructure/http/router_factory.py`
- Modify: `tests/unit/test_mlflow_persistence_contract.py:19-31`
- Test: `tests/unit/test_router_base_promotion.py`

**Interfaces:**
- Consumes: `ModelContainer.promotes_base/rollback/train_use_case` (Tarea 4), `RollbackRequest/RollbackResponse` (Tarea 1), `TinyPromotingPlugin` y DTOs (Tarea 4).
- Produces: `POST {prefix}/rollback` — 200 `RollbackResponse`, 404, 409, 500, 501 (modelo sin versiones). Constante de tests `PROMOTING_MODEL_IDS` en `tests/unit/test_mlflow_persistence_contract.py`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_router_base_promotion.py`:

```python
"""HTTP surface of the versioned base: /train answers base_version, /rollback, /stats."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.infrastructure.http.dependencies.container import ModelContainer
from app.infrastructure.http.router_factory import make_model_router
from tests.conftest import TEST_REGISTRY
from tests.unit.tiny_promoting_plugin import (
    TinyPredictRequest,
    TinyPredictResponse,
    TinyPromotingPlugin,
    TinyTrainRequest,
    TinyTrainResponse,
)

PREFIX = "/models/tiny"


@pytest.fixture
def tiny_client():
    TinyPromotingPlugin.fail_load = False
    app = FastAPI()
    container = ModelContainer(TinyPromotingPlugin())
    container.init()
    app.state.containers = {"tiny": container}
    app.include_router(make_model_router(
        model_id="tiny", version="1", predict_request_type=TinyPredictRequest,
        predict_response_type=TinyPredictResponse, train_request_type=TinyTrainRequest,
        train_response_type=TinyTrainResponse,
    ), prefix=PREFIX)
    yield TestClient(app)
    TinyPromotingPlugin.fail_load = False


def test_train_without_run_answers_the_new_base_version(tiny_client):
    resp = tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})
    assert resp.status_code == 200
    assert resp.json()["base_version"]["version"] == "v1"
    assert tiny_client.get(f"{PREFIX}/stats").json()["base_version"]["version"] == "v1"


def test_rollback_endpoint(tiny_client):
    tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})
    tiny_client.post(f"{PREFIX}/train", json={"data_path": "d2"})
    resp = tiny_client.post(f"{PREFIX}/rollback")
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_version"]["version"] == "v1" and body["previous_base_version"]["version"] == "v2"
    resp = tiny_client.post(f"{PREFIX}/rollback", json={"version": "v0"})
    assert resp.status_code == 200 and resp.json()["base_version"]["version"] == "v0"


def test_rollback_errors(tiny_client):
    assert tiny_client.post(f"{PREFIX}/rollback").status_code == 409            # v0, no parent
    assert tiny_client.post(f"{PREFIX}/rollback", json={"version": "v7"}).status_code == 404
    tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})
    TinyPromotingPlugin.fail_load = True
    resp = tiny_client.post(f"{PREFIX}/rollback")
    assert resp.status_code == 500 and "v0" in resp.json()["detail"]


def test_train_reload_failure_is_500(tiny_client):
    TinyPromotingPlugin.fail_load = True
    resp = tiny_client.post(f"{PREFIX}/train", json={"data_path": "d1"})
    assert resp.status_code == 500 and "vuelve a v0" in resp.json()["detail"]


def test_models_without_versions_answer_501_on_rollback(client):
    entry = TEST_REGISTRY[0]
    resp = client.post(f"{entry.prefix}/rollback")
    assert resp.status_code == 501
```

En `tests/unit/test_mlflow_persistence_contract.py`, sustituir las líneas de `TRAINABLE` y el docstring del punto 1:

```python
PROMOTING_MODEL_IDS = {
    "ml13-wine-price-fluctuation-prediction", "ml14-wine-phyto-price-forecast",
    "ml15-wine-ipi-price-forecast", "ml16-meat-raw-material-price-alert",
    "ml17-meat-market-price-analysis", "ml18-meat-spatial-price-forecast",
    "ml21-cereals-price-spatial", "ml23-lactic-market-price-forecast",
}
# The 8 models whose /train publishes the new base accept /train without a run (spec D4).
TRAINABLE = [entry for entry in TEST_REGISTRY
             if entry.train_request_type is not None and entry.model_id not in PROMOTING_MODEL_IDS]
```

y en el docstring del módulo, el punto 1 pasa a ser:

```text
1. ``mlflow_run_id`` is mandatory and non-empty on every /train (→ 422), except on the 8 models
   whose /train publishes the new base model (PROMOTING_MODEL_IDS), where it is optional.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_router_base_promotion.py -q`
Expected: FAIL (`/rollback` → 404 Not Found de FastAPI)

- [ ] **Step 3: Implement**

En `app/infrastructure/http/router_factory.py`:

Imports: añadir

```python
from app.application.dto.base_version_dto import RollbackRequest, RollbackResponse
```

y ampliar el import de excepciones:

```python
from app.domain.services.exceptions import (
    ArchiveLimitExceededError,
    BaseReloadError,
    BaseVersionNotFoundError,
    ModelPersistenceError,
    NoPreviousBaseVersionError,
    TrainingNotSupportedError,
    UserModelUnavailableError,
)
```

Antes de `return router`:

```python
    @router.post("/rollback")
    def rollback(request: Request, body: RollbackRequest | None = None) -> RollbackResponse:
        """Serve a previous base version (the current one's parent unless body.version is given)."""
        container = _get_container(request)
        if not getattr(container, "promotes_base", False):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"El modelo '{model_id}' no versiona su modelo base: no admite rollback.",
            )
        try:
            return RollbackResponse(**container.rollback(body.version if body else None))
        except BaseVersionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except NoPreviousBaseVersionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except BaseReloadError as exc:
            logger.error("Rollback of '%s' failed to load: %s", model_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc
```

En el handler de `/train`, añadir antes del `except Exception` final:

```python
        except BaseReloadError as exc:
            logger.error("Retrained base of '%s' does not load: %s", model_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc
```

Y en el docstring de `make_model_router`, cambiar `(health / stats / predict / train)` por `(health / stats / predict / train / rollback)`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_router_base_promotion.py tests/unit/test_mlflow_persistence_contract.py -q`
Expected: todos en verde

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 6: ml17 publica su base al reentrenar

**Files:**
- Modify: `app/plugins/ml17_meat_market_price_analysis/plugin.py` (docstring del módulo, imports, clase, `train()`)
- Modify: `app/plugins/ml17_meat_market_price_analysis/train_dto.py`
- Modify: `tests/unit/test_ml17_meat_market_price_analysis.py:64-` (`test_train_without_mlflow_run_id_returns_422`)
- Test: `tests/unit/test_ml17_base_promotion.py`

**Interfaces:**
- Consumes: `persist_retrained_bundle`, `PersistResult` (Tarea 3), `VersionInfo` (Tarea 1), fixture `isolated_base_store`.
- Produces: `Ml17MeatMarketPriceAnalysisPlugin.BASE_ARTIFACT_FOLDER = "ml17_meat_market_price_analysis"`; `TrainResponse.base_version: VersionInfo | None`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_ml17_base_promotion.py`:

```python
"""ml17: every /train publishes the refit Ridge as the new base (spec 2026-10-14)."""
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from app.plugins.ml17_meat_market_price_analysis import plugin as plugin_mod
from app.plugins.ml17_meat_market_price_analysis.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURE_COLUMNS,
    MODEL_FILENAME,
)


@pytest.fixture
def train_csv(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for col in FEATURE_COLUMNS:
        df[col] = rng.normal(100, 5, len(dates))
    df["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    path = tmp_path / "ml17.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_declares_its_base_folder():
    assert plugin_mod.Ml17MeatMarketPriceAnalysisPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_the_new_base(train_csv, isolated_base_store, monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml17MeatMarketPriceAnalysisPlugin().train(data_path=train_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id is None
    assert resp.base_version.metrics["mae"] == pytest.approx(resp.mae, abs=1e-4)
    tracker.upload_artifacts.assert_not_called()
    model = joblib.load(isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1" / MODEL_FILENAME)
    assert hasattr(model, "predict")


def test_train_with_run_also_uploads_to_mlflow(train_csv, monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml17MeatMarketPriceAnalysisPlugin().train(data_path=train_csv, mlflow_run_id="run-1")
    assert tracker.upload_artifacts.call_args.kwargs["artifact_path"] == "model"
    tracker.log_params.assert_called_once()
    assert resp.base_version.mlflow_run_id == "run-1" and resp.upload_warning is None


def test_mlflow_failure_is_a_warning(train_csv, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml17MeatMarketPriceAnalysisPlugin().train(data_path=train_csv, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v1"
```

En `tests/unit/test_ml17_meat_market_price_analysis.py`, sustituir la función `test_train_without_mlflow_run_id_returns_422` entera por:

```python
def test_train_without_mlflow_run_id_is_accepted(client):
    """mlflow_run_id is optional: every /train publishes the new base model (spec 2026-10-14)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv"})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml17_base_promotion.py tests/unit/test_ml17_meat_market_price_analysis.py -q`
Expected: FAIL (`AttributeError: ... has no attribute 'BASE_ARTIFACT_FOLDER'` y 422 en el test de API)

- [ ] **Step 3: Implement**

`train_dto.py`: añadir `from app.application.dto.base_version_dto import VersionInfo`; quitar el import de `MlflowRunId`; docstring de `TrainRequest` → `"""Retrain request: CSV with the 8 official_v1_4 columns; MLflow run optional."""`; sustituir el campo `mlflow_run_id` por:

```python
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
    )
```

y añadir al final de `TrainResponse`:

```python
    base_version: VersionInfo | None = Field(
        default=None, description="Versión del modelo base publicada por este reentreno"
    )
```

`plugin.py`:
- Docstring del módulo: sustituir ```train()`` refits the Ridge pipeline on user data and persists the result exclusively to MLflow (see train()).` por `` ``train()`` refits the Ridge pipeline and publishes it as the new base model (see train()).``
- Imports: añadir `ARTIFACT_FOLDER_NAME` al import de constants (orden alfabético, antes de `FEATURE_COLUMNS`); quitar `ModelPersistenceError` del import de excepciones (queda `from app.domain.services.exceptions import ModelNotLoadedError`); añadir `from app.infrastructure.base_model_publisher import persist_retrained_bundle`.
- En la clase, justo después del docstring:

```python
    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model
```

- En `train()`: sustituir el docstring a partir de `pipeline fit on X=features[:-1], y=target[1:].` por:

```text
        pipeline fit on X=features[:-1], y=target[1:]. The refit pipeline is a brand-new
        object — self._model is never mutated. It is published as the new base model version
        (persist_retrained_bundle) and, if mlflow_run_id is given, also saved to that run; the
        container then serves the new base.
```

- Sustituir el bloque

```python
        tracker = BaseMLflowTracker(mlflow_run_id)
        tracker.log_params({
            "alpha": RIDGE_ALPHA,
            "imputer_strategy": IMPUTER_STRATEGY,
            "scaler": "StandardScaler",
        })

```

por:

```python
        tracker = BaseMLflowTracker(mlflow_run_id)

```

- Sustituir desde `        upload_warning = None` hasta el final del `except ... raise ModelPersistenceError(...) from exc` por:

```python
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "n_samples": len(X_train),
            **({"mase": mase} if mase is not None else {}),
            **({"r2_train": r2_train} if r2_train is not None else {}),
            **({"directional_accuracy": directional_accuracy} if directional_accuracy is not None else {}),
        }
        bundle_dir = tempfile.mkdtemp(prefix="ml17_train_")
        try:
            joblib.dump(new_model, f"{bundle_dir}/{MODEL_FILENAME}")

            def _upload_to_mlflow() -> None:
                tracker.log_params({
                    "alpha": RIDGE_ALPHA,
                    "imputer_strategy": IMPUTER_STRATEGY,
                    "scaler": "StandardScaler",
                })
                tracker.log_metrics(metrics)
                tracker.upload_artifacts(bundle_dir, artifact_path="model")

            persisted = persist_retrained_bundle(
                bundle_dir, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=metrics, mlflow_run_id=mlflow_run_id, upload_to_mlflow=_upload_to_mlflow,
            )
        finally:
            shutil.rmtree(bundle_dir, ignore_errors=True)
```

- En el `return TrainResponse(...)`: `detail=f"Reentrenamiento completado — modelo base {persisted.base_version['version']}."`, `upload_warning=persisted.upload_warning`, y añadir `base_version=persisted.base_version,`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml17_base_promotion.py tests/unit/test_ml17_meat_market_price_analysis.py tests/unit/test_user_model_isolation.py -q`
Expected: todos en verde

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 7: ml13 publica su base al reentrenar

**Files:**
- Modify: `app/plugins/ml13_wine_price_fluctuation_prediction/plugin.py` (clase, `train()`, imports)
- Modify: `app/plugins/ml13_wine_price_fluctuation_prediction/train_dto.py`
- Modify: `tests/unit/test_ml13_wine_price_fluctuation_prediction.py` (`test_real_train_uploads_only_to_mlflow`, `test_real_train_fails_when_mlflow_upload_fails`)

**Interfaces:**
- Consumes: `persist_retrained_bundle` (Tarea 3), `VersionInfo` (Tarea 1), fixture `isolated_base_store`.
- Produces: `Ml13WinePriceFluctuationPredictionPlugin.BASE_ARTIFACT_FOLDER = "ml13_wine_price_fluctuation_prediction"`; `TrainResponse.base_version`.

- [ ] **Step 1: Write the failing tests**

En `tests/unit/test_ml13_wine_price_fluctuation_prediction.py`, sustituir `test_real_train_uploads_only_to_mlflow` y `test_real_train_fails_when_mlflow_upload_fails` por:

```python
def test_real_train_publishes_the_new_base_and_uploads_to_the_run(real_plugin, train_csv, monkeypatch,
                                                                   isolated_base_store):
    from unittest.mock import MagicMock

    from app.plugins.ml13_wine_price_fluctuation_prediction import model_loader
    from app.plugins.ml13_wine_price_fluctuation_prediction import plugin as plugin_mod
    from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
        ARTIFACT_FOLDER_NAME, FEATURE_SCHEMA_FILENAME, MODEL_CONFIG_FILENAME, MODEL_FILENAME, SCALER_FILENAME,
    )

    repo_dir = model_loader._store.local_dir  # pylint: disable=protected-access
    before = set(repo_dir.rglob("*")) if repo_dir.exists() else set()
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = real_plugin.train(data_path=train_csv, mlflow_run_id="run-1")
    assert resp.n_test_rows == 24 and resp.upload_warning is None
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id == "run-1"
    assert tracker.upload_artifacts.call_args.kwargs["artifact_path"] == "model"
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    for name in (MODEL_FILENAME, SCALER_FILENAME, FEATURE_SCHEMA_FILENAME, MODEL_CONFIG_FILENAME):
        assert (v1 / name).exists()
    after = set(repo_dir.rglob("*")) if repo_dir.exists() else set()
    assert after == before  # nothing written under the repo's artifacts/


def test_real_train_without_run_publishes_the_base(real_plugin, train_csv):
    resp = real_plugin.train(data_path=train_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id is None


def test_real_train_mlflow_failure_is_a_warning(real_plugin, train_csv, monkeypatch):
    from unittest.mock import MagicMock

    from app.plugins.ml13_wine_price_fluctuation_prediction import plugin as plugin_mod

    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = real_plugin.train(data_path=train_csv, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v1"


def test_declares_its_base_folder():
    from app.plugins.ml13_wine_price_fluctuation_prediction.constants import ARTIFACT_FOLDER_NAME
    from app.plugins.ml13_wine_price_fluctuation_prediction.plugin import (
        Ml13WinePriceFluctuationPredictionPlugin,
    )
    assert Ml13WinePriceFluctuationPredictionPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_mlflow_run_id_is_accepted(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/wine_prices.csv"})
    assert resp.status_code == 200
```

Si el fichero tiene un test de API que espera 422 sin `mlflow_run_id` (`grep -n "422" tests/unit/test_ml13_wine_price_fluctuation_prediction.py`), borrarlo: lo sustituye `test_train_without_mlflow_run_id_is_accepted`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml13_wine_price_fluctuation_prediction.py -q`
Expected: FAIL (sin `base_version`, sin `BASE_ARTIFACT_FOLDER`, 422)

- [ ] **Step 3: Implement**

`train_dto.py`: añadir `from app.application.dto.base_version_dto import VersionInfo`; quitar el import de `MlflowRunId`; cambiar `mlflow_run_id: MlflowRunId = Field(...)` por:

```python
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
    )
```

y añadir al final de `TrainResponse`:

```python
    base_version: VersionInfo | None = Field(
        default=None, description="Versión del modelo base publicada por este reentreno"
    )
```

`plugin.py`:
- Imports: añadir `ARTIFACT_FOLDER_NAME` al import de constants; quitar `ModelPersistenceError` del import de excepciones; añadir `from app.infrastructure.base_model_publisher import persist_retrained_bundle`.
- En la clase, tras el docstring: `    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model`
- Docstring de `train()`: sustituir desde `Fresh objects are trained;` hasta el final por:

```text
        hold-out metrics on the last 24 weeks. Fresh objects are trained (self._bundle is never
        mutated). The bundle (canonical filenames) is published as the new base model version
        and, if mlflow_run_id is given, also saved to that run (artifact_path="model"); a
        failed MLflow upload is only an upload_warning, a failed publish is ModelPersistenceError.
```

- Sustituir el bloque desde `        tracker = BaseMLflowTracker(mlflow_run_id)` hasta el `finally: shutil.rmtree(mlflow_tmp, ignore_errors=True)` (incluido) por:

```python
        tracker = BaseMLflowTracker(mlflow_run_id)
        metrics = {
            **{f"test_{k}": v for k, v in test_m.items()},
            **{f"cv_{name}_{k}": v for name, m in cv_m.items() for k, v in m.items()},
            **{f"smart_score_{k}": v for k, v in scores.items()},
            "n_trainval": result["n_trainval"],
            "n_test": result["n_test"],
        }
        bundle_dir = tempfile.mkdtemp(prefix="ml13_train_")
        try:
            joblib.dump(result["model"], f"{bundle_dir}/{MODEL_FILENAME}")
            joblib.dump(result["scaler"], f"{bundle_dir}/{SCALER_FILENAME}")
            with open(f"{bundle_dir}/{FEATURE_SCHEMA_FILENAME}", "w", encoding="utf-8") as fh:
                json.dump(schema, fh, indent=2, ensure_ascii=False)
            with open(f"{bundle_dir}/{MODEL_CONFIG_FILENAME}", "w", encoding="utf-8") as fh:
                json.dump(model_config, fh, indent=2)

            def _upload_to_mlflow() -> None:
                tracker.log_params({
                    "best_model_type": result["model_type"],
                    "return_threshold": RETURN_THRESHOLD,
                    "target_window": TARGET_WINDOW,
                    "test_size": TEST_SIZE,
                    "n_folds": N_FOLDS,
                    "random_seed": RANDOM_SEED,
                    **{f"logreg_{k}": v for k, v in LOGREG_PARAMS.items()},
                    **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()},
                })
                tracker.log_metrics(metrics)
                tracker.upload_artifacts(bundle_dir, artifact_path=MLFLOW_ARTIFACT_PATH)

            persisted = persist_retrained_bundle(
                bundle_dir, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=metrics, mlflow_run_id=mlflow_run_id, upload_to_mlflow=_upload_to_mlflow,
            )
        finally:
            shutil.rmtree(bundle_dir, ignore_errors=True)
```

- En el `return TrainResponse(...)`: `upload_warning=persisted.upload_warning,`, añadir `base_version=persisted.base_version,` y en `detail` añadir al final ` Modelo base {persisted.base_version['version']}.` (la f-string ya existente pasa a ser `f"... {result['model_type']}). Modelo base {persisted.base_version['version']}."`).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml13_wine_price_fluctuation_prediction.py tests/unit/test_user_model_isolation.py -q`
Expected: todos en verde

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 8: ml16 publica su base al reentrenar

**Files:**
- Modify: `app/plugins/ml16_meat_raw_material_price_alert/plugin.py`
- Modify: `app/plugins/ml16_meat_raw_material_price_alert/train_dto.py`
- Modify: `tests/unit/test_ml16_meat_raw_material_price_alert.py` (`test_train_requires_mlflow_run_id`)
- Test: `tests/unit/test_ml16_base_promotion.py`

**Interfaces:**
- Consumes: `persist_retrained_bundle` (Tarea 3), `VersionInfo` (Tarea 1), fixture `isolated_base_store`.
- Produces: `Ml16MeatRawMaterialPriceAlertPlugin.BASE_ARTIFACT_FOLDER = "ml16_meat_raw_material_price_alert"`; `TrainResponse.base_version`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_ml16_base_promotion.py`:

```python
"""ml16: every /train publishes the retrained bundle as the new base (spec 2026-10-14)."""
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.plugins.ml16_meat_raw_material_price_alert import plugin as plugin_mod
from app.plugins.ml16_meat_raw_material_price_alert import training
from app.plugins.ml16_meat_raw_material_price_alert.constants import (
    ARTIFACT_FOLDER_NAME,
    BAGGING_FILENAMES,
    MODEL_FILENAMES,
    RAW_REQUIRED_COLUMNS,
    SCALER_FILENAMES,
    TARGETS,
    TRAIN_CONFIG_FILENAME,
)

_METRICS = {"threshold": 0.5, "accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "auc": 0.75}


@pytest.fixture
def instance(monkeypatch):
    monkeypatch.setattr(training, "train_models", lambda df: {
        "models": {t: f"model-{t}" for t in TARGETS},
        "scalers": {t: f"scaler-{t}" for t in TARGETS},
        "bagging_models": {t: [f"bag-{t}"] for t in TARGETS},
        "thresholds": {t: 0.5 for t in TARGETS},
        "input_cols_per_target": {t: ["indice_animales"] for t in TARGETS},
        "metrics": {t: dict(_METRICS) for t in TARGETS},
        "n_train": 40, "n_test": 12,
    })
    plugin = plugin_mod.Ml16MeatRawMaterialPriceAlertPlugin()
    plugin._models = {t: object() for t in TARGETS}       # pylint: disable=protected-access
    plugin._scalers = {t: object() for t in TARGETS}      # pylint: disable=protected-access
    plugin._train_config = {"horizon": 4}                 # pylint: disable=protected-access
    return plugin


@pytest.fixture
def train_csv(tmp_path):
    cols = [c for c in RAW_REQUIRED_COLUMNS if c != "month"] + list(TARGETS)
    df = pd.DataFrame({c: [0.0] * 3 for c in cols})
    df["fecha"] = ["2024-01-01", "2024-02-01", "2024-03-01"]
    path = tmp_path / "ml16.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_declares_its_base_folder():
    assert plugin_mod.Ml16MeatRawMaterialPriceAlertPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_the_new_base(instance, train_csv, isolated_base_store):
    resp = instance.train(data_path=train_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    assert resp.base_version.metrics["target_animales_f1"] == 0.65
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    for target in TARGETS:
        for names in (MODEL_FILENAMES, SCALER_FILENAMES, BAGGING_FILENAMES):
            assert (v1 / names[target]).exists()
    assert json.loads((v1 / TRAIN_CONFIG_FILENAME).read_text())["horizon"] == 4


def test_train_with_run_uploads_and_failure_is_a_warning(instance, train_csv, monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=train_csv, mlflow_run_id="run-1")
    assert tracker.upload_artifacts.call_args.kwargs["artifact_path"] == "model"
    assert resp.base_version.mlflow_run_id == "run-1"

    tracker.upload_artifacts.side_effect = ConnectionError("mlflow caído")
    resp = instance.train(data_path=train_csv, mlflow_run_id="run-2")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v2"
```

En `tests/unit/test_ml16_meat_raw_material_price_alert.py`, sustituir `test_train_requires_mlflow_run_id` entera por:

```python
def test_train_without_mlflow_run_id_is_accepted(client):
    """mlflow_run_id is optional: every /train publishes the new base model (spec 2026-10-14)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/dataset_base.csv"})
    assert resp.status_code == 200
```

(Si el fichero no define `PREFIX`, usar la ruta literal que usa `test_train`: `"/models/ml16-meat-raw-material-price-alert/train"`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml16_base_promotion.py tests/unit/test_ml16_meat_raw_material_price_alert.py -q`
Expected: FAIL (`ModelPersistenceError: Sin run de MLflow` / 422)

- [ ] **Step 3: Implement**

`train_dto.py`: añadir `from pydantic import Field` si no está y `from app.application.dto.base_version_dto import VersionInfo`; quitar el import de `MlflowRunId`; `mlflow_run_id: MlflowRunId` pasa a:

```python
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
    )
```

y al final de `TrainResponse`: `    base_version: VersionInfo | None = None`.

`plugin.py`:
- Imports: añadir `ARTIFACT_FOLDER_NAME` al import de constants; quitar `ModelPersistenceError` del import de excepciones; añadir `from app.infrastructure.base_model_publisher import persist_retrained_bundle`.
- Clase: `    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model`
- Docstring de `train()`: sustituir desde `Trains into fresh model objects` hasta el final por:

```text
        bagging). Trains into fresh model objects — self._models etc. are never mutated. The
        bundle (canonical filenames) is published as the new base model version and, if
        mlflow_run_id is given, also saved to that run (artifact_path="model").
```

- Sustituir el bloque desde `        upload_warning = None` hasta el `raise ModelPersistenceError(...) from exc` del `except` (incluido) por:

```python
        flat_metrics = {
            f"{target}_{k}": v for target, m in result["metrics"].items() for k, v in m.items()
        }
        tracker = BaseMLflowTracker(mlflow_run_id)
        bundle_dir = tempfile.mkdtemp(prefix="ml16_train_")
        try:
            for target in TARGETS:
                joblib.dump(result["models"][target], f"{bundle_dir}/{MODEL_FILENAMES[target]}")
                joblib.dump(result["scalers"][target], f"{bundle_dir}/{SCALER_FILENAMES[target]}")
                joblib.dump(result["bagging_models"][target], f"{bundle_dir}/{BAGGING_FILENAMES[target]}")
            with open(f"{bundle_dir}/{TRAIN_CONFIG_FILENAME}", "w", encoding="utf-8") as fh:
                json.dump(train_config, fh, indent=2, ensure_ascii=False)

            def _upload_to_mlflow() -> None:
                tracker.log_params({
                    "seed": training.DEFAULT_SEED,
                    "lookback": training.DEFAULT_LOOKBACK,
                    "n_bagging": training.DEFAULT_N_BAGGING,
                })
                tracker.log_metrics(flat_metrics)
                tracker.upload_artifacts(bundle_dir, artifact_path="model")

            persisted = persist_retrained_bundle(
                bundle_dir, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=flat_metrics, mlflow_run_id=mlflow_run_id, upload_to_mlflow=_upload_to_mlflow,
            )
        finally:
            shutil.rmtree(bundle_dir, ignore_errors=True)
```

- En el `return TrainResponse(...)`: `upload_warning=persisted.upload_warning,`, añadir `base_version=persisted.base_version,`, y `detail=f"Reentrenamiento completado (XGBoost + LogisticRegression, procedimiento original). Modelo base {persisted.base_version['version']}."`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml16_base_promotion.py tests/unit/test_ml16_meat_raw_material_price_alert.py tests/unit/test_ml16_preprocessing.py tests/unit/test_user_model_isolation.py -q`
Expected: todos en verde

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 9: ml23 publica su base al reentrenar

**Files:**
- Modify: `app/plugins/ml23_lactic_market_price_forecast/plugin.py`
- Modify: `app/plugins/ml23_lactic_market_price_forecast/train_dto.py`
- Modify: `tests/unit/test_ml23_lactic_market_price_forecast.py` (`test_train_without_mlflow_run_id_returns_422`)
- Test: `tests/unit/test_ml23_base_promotion.py`

**Interfaces:**
- Consumes: `persist_retrained_bundle` (Tarea 3), `VersionInfo` (Tarea 1), `upload_artifacts_to_mlflow(artifact_dir, mlflow_run_id, metrics)` (ya existe en `mlflow_utils.py`).
- Produces: `Ml23LacticMarketPriceForecastPlugin.BASE_ARTIFACT_FOLDER = "ml23_lactic_market_price_forecast"`; `TrainResponse.base_version`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_ml23_base_promotion.py`:

```python
"""ml23: every /train publishes the refit GRU as the new base (spec 2026-10-14)."""
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from app.plugins.ml23_lactic_market_price_forecast import mlflow_utils  # noqa: E402
from app.plugins.ml23_lactic_market_price_forecast import plugin as plugin_mod  # noqa: E402
from app.plugins.ml23_lactic_market_price_forecast.constants import (  # noqa: E402
    ARTIFACT_FOLDER_NAME,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)


@pytest.fixture
def train_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for producto, canal in (("LECHE ENTERA", "HIPER"), ("LECHE DESNATADA", "SUPER")):
        for i, fecha in enumerate(pd.date_range("2015-01-01", periods=60, freq="MS")):
            rows.append({"fecha": fecha.strftime("%Y-%m-%d"), "producto": producto, "canal": canal,
                         "target_precio_medio": 0.8 + 0.002 * i + rng.normal(0, 0.01),
                         "ipc_general_lag1": 100 + 0.1 * i, "mes": fecha.month})
    path = tmp_path / "ml23.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(plugin_mod, "TRAIN_EPOCHS", 2)
    monkeypatch.setattr(plugin_mod, "TRAIN_N_SEEDS", 1)


def test_declares_its_base_folder():
    assert plugin_mod.Ml23LacticMarketPriceForecastPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_the_new_base(train_csv, fast, isolated_base_store):
    resp = plugin_mod.Ml23LacticMarketPriceForecastPlugin().train(data_path=train_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    assert (v1 / MODEL_FILENAME).exists() and (v1 / SCALER_FILENAME).exists()
    assert json.loads((v1 / MANIFEST_FILENAME).read_text())["selected_model"] == "GRU"


def test_train_with_run_and_mlflow_failure_is_a_warning(train_csv, fast, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml23LacticMarketPriceForecastPlugin().train(data_path=train_csv, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id == "run-1"
```

En `tests/unit/test_ml23_lactic_market_price_forecast.py`, sustituir `test_train_without_mlflow_run_id_returns_422` entera por:

```python
def test_train_without_mlflow_run_id_is_accepted(client):
    """mlflow_run_id is optional: every /train publishes the new base model (spec 2026-10-14)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv"})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml23_base_promotion.py tests/unit/test_ml23_lactic_market_price_forecast.py -q`
Expected: FAIL (`ModelPersistenceError: Sin run de MLflow` / 422). Si el CSV sintético no basta para el split (error de forma en `prepare_rnn_split_with_test`), alargar `periods` a 84: el contrato del test es "entrena y publica", no la cifra de meses.

- [ ] **Step 3: Implement**

`train_dto.py`: añadir `from app.application.dto.base_version_dto import VersionInfo`; quitar el import de `MlflowRunId`; `mlflow_run_id: MlflowRunId = Field(...)` pasa a:

```python
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
    )
```

y al final de `TrainResponse`:

```python
    base_version: VersionInfo | None = Field(
        default=None, description="Versión del modelo base publicada por este reentreno"
    )
```

`plugin.py`:
- Imports: añadir `ARTIFACT_FOLDER_NAME` al import de constants; quitar `ModelPersistenceError` del import de excepciones; añadir `from app.infrastructure.base_model_publisher import persist_retrained_bundle`.
- Clase: `    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model`
- Docstring de `train()`: sustituir `The refit checkpoint is written only to a temp dir and uploaded to MLflow; the served artifacts/.../gru_model.pt is never overwritten.` por `The refit bundle is published as the new base model version (never overwriting a previous one) and, if mlflow_run_id is given, also uploaded to that run.`
- Quitar la línea `        upload_warning = None` del principio de `train()`.
- Sustituir el bloque desde `            if not mlflow_run_id:` hasta el `raise ModelPersistenceError(...) from exc` del `except` (incluido) por:

```python
            metrics = {
                "mae": test_metrics["MAE"], "rmse": test_metrics["RMSE"],
                "mape_pct": test_metrics["MAPE_pct"], "r2": test_metrics["R2"],
                "direction_acc_pct": test_metrics["direction_acc_pct"],
                "n_train": n_train, "n_test": n_test,
            }
            persisted = persist_retrained_bundle(
                str(upload_dir), artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=metrics, mlflow_run_id=mlflow_run_id,
                upload_to_mlflow=lambda: upload_artifacts_to_mlflow(str(upload_dir), mlflow_run_id, metrics=metrics),
            )
```

- En el `return TrainResponse(...)`: `detail=f"Reentrenamiento completado — modelo base {persisted.base_version['version']}."`, `upload_warning=persisted.upload_warning,` y añadir `base_version=persisted.base_version,`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml23_base_promotion.py tests/unit/test_ml23_lactic_market_price_forecast.py tests/unit/test_user_model_isolation.py -q`
Expected: todos en verde

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 10: ml15 publica su base y, opcionalmente, sus históricos de referencia

**Files:**
- Modify: `app/plugins/ml15_wine_ipi_price_forecast/plugin.py`
- Modify: `app/plugins/ml15_wine_ipi_price_forecast/train_dto.py`
- Modify: `tests/unit/test_ml15_training.py` (`test_train_uploads_only_to_mlflow`, `test_train_fails_when_mlflow_upload_fails`)
- Modify: `tests/unit/test_ml15_wine_ipi_price_forecast.py` (`test_train_without_mlflow_run_id_returns_422`)
- Modify: `tests/unit/test_mlflow_persistence_contract.py` (`test_ml15_train_fails_when_mlflow_upload_fails`)

**Interfaces:**
- Consumes: `persist_retrained_bundle` (Tarea 3), `VersionInfo` (Tarea 1), `history.load_monthly_csv(path) -> pd.DataFrame`, `history.load_reference_data() -> (prices_df, financial_df)`, `REFERENCE_PRICES_FILENAME`, `REFERENCE_FINANCIAL_FILENAME`.
- Produces: `Ml15WineIpiPriceForecastPlugin.BASE_ARTIFACT_FOLDER = "ml15_wine_ipi_price_forecast"`; `TrainRequest.reference_prices_path: str = ""`, `TrainRequest.reference_financial_path: str = ""`; `train(self, *, data_path, mlflow_run_id="", reference_prices_path="", reference_financial_path="")`; `TrainResponse.base_version`. El caso de uso ya reenvía a `train()` todo campo del request que su firma declare.

- [ ] **Step 1: Write the failing tests**

En `tests/unit/test_ml15_training.py`, sustituir `test_train_uploads_only_to_mlflow` y `test_train_fails_when_mlflow_upload_fails` por:

```python
def test_train_publishes_the_new_base_and_uploads_to_the_run(instance, monkeypatch, tmp_path, isolated_base_store):
    csv = tmp_path / "panel.csv"
    _panel().to_csv(csv, index=False)
    store = tmp_path / "mlflow"
    tracker = MagicMock()
    tracker.upload_artifacts.side_effect = lambda d, artifact_path: shutil.copytree(d, store / artifact_path)
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))

    resp = instance.train(data_path=str(csv), mlflow_run_id="run-1")
    assert resp.n_train_rows == 29 and resp.n_test_rows == 7 and resp.train_end_date == "2020-11-01"
    payload = joblib.load(store / "model" / MODEL_FILENAME)
    assert payload["feature_columns"] == list(FEATURE_COLUMNS) and payload["train_end_date"] == "2020-11-01"
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id == "run-1"
    v1 = isolated_base_store / "ml15_wine_ipi_price_forecast" / "versions" / "v1"
    assert joblib.load(v1 / MODEL_FILENAME)["train_end_date"] == "2020-11-01"
    assert "históricos exógenos" in resp.upload_warning   # no reference CSVs were given


def test_train_without_run_publishes_the_base(instance, tmp_path):
    csv = tmp_path / "panel.csv"
    _panel().to_csv(csv, index=False)
    resp = instance.train(data_path=str(csv), mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id is None


def test_mlflow_failure_is_a_warning(instance, monkeypatch, tmp_path):
    csv = tmp_path / "panel.csv"
    _panel().to_csv(csv, index=False)
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=str(csv), mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v1"


def _reference_paths():
    from app.plugins.ml15_wine_ipi_price_forecast import model_loader
    return model_loader.reference_data_paths()


def test_reference_csvs_travel_with_the_new_base(instance, tmp_path, isolated_base_store):
    try:
        current_prices, current_financial = _reference_paths()
    except FileNotFoundError:
        pytest.skip("ml15 reference CSVs not present locally (artifacts/ml15_wine_ipi_price_forecast)")
    csv = tmp_path / "panel.csv"
    _panel().to_csv(csv, index=False)
    prices, financial = tmp_path / "prices.csv", tmp_path / "financial.csv"
    shutil.copyfile(current_prices, prices)
    shutil.copyfile(current_financial, financial)

    resp = instance.train(data_path=str(csv), reference_prices_path=str(prices),
                          reference_financial_path=str(financial))
    v1 = isolated_base_store / "ml15_wine_ipi_price_forecast" / "versions" / "v1"
    assert (v1 / REFERENCE_PRICES_FILENAME).exists() and (v1 / REFERENCE_FINANCIAL_FILENAME).exists()
    assert resp.upload_warning is None


def test_reference_csv_with_wrong_columns_is_rejected_before_training(instance, tmp_path, monkeypatch):
    try:
        _reference_paths()
    except FileNotFoundError:
        pytest.skip("ml15 reference CSVs not present locally")
    monkeypatch.setattr(plugin_mod.training, "train_model",
                        MagicMock(side_effect=AssertionError("must not train")))
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "x": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="reference_prices_path"):
        instance.train(data_path=str(tmp_path / "panel.csv"), reference_prices_path=str(bad))


def test_only_one_reference_csv_is_rejected(instance, tmp_path):
    with pytest.raises(ValueError, match="juntos"):
        instance.train(data_path=str(tmp_path / "panel.csv"), reference_prices_path="/tmp/p.csv")
```

Añadir a los imports de `tests/unit/test_ml15_training.py` (en el bloque de constants ya importado) `REFERENCE_FINANCIAL_FILENAME` y `REFERENCE_PRICES_FILENAME`, y `import pandas as pd` si no está.

En `tests/unit/test_mlflow_persistence_contract.py`, sustituir `test_ml15_train_fails_when_mlflow_upload_fails` por:

```python
def test_ml15_train_mlflow_failure_is_a_warning_and_the_base_is_published(ml15, monkeypatch):
    plugin, _, instance, data = ml15

    class UnreachableTracker:
        def __init__(self, run_id):
            self.run_id = run_id

        def log_params(self, *_args):
            pass

        def log_metrics(self, *_args):
            pass

        def upload_artifacts(self, *_args, **_kwargs):
            raise ConnectionError("MLflow unreachable")

    monkeypatch.setattr(plugin, "BaseMLflowTracker", UnreachableTracker)
    resp = instance.train(data_path=data, mlflow_run_id="run-1")
    assert "MLflow unreachable" in resp.upload_warning
    assert resp.base_version.version == "v1"
```

En `tests/unit/test_ml15_wine_ipi_price_forecast.py`, sustituir `test_train_without_mlflow_run_id_returns_422` entera por:

```python
def test_train_without_mlflow_run_id_is_accepted(client):
    """mlflow_run_id is optional: every /train publishes the new base model (spec 2026-10-14)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/a15_train.csv"})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml15_training.py tests/unit/test_ml15_wine_ipi_price_forecast.py tests/unit/test_mlflow_persistence_contract.py -q`
Expected: FAIL (`unexpected keyword argument 'reference_prices_path'`, sin `base_version`, 422)

- [ ] **Step 3: Implement**

`train_dto.py`: añadir `from app.application.dto.base_version_dto import VersionInfo`; quitar el import de `MlflowRunId`; `mlflow_run_id: MlflowRunId = Field(...)` pasa a:

```python
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
    )
    reference_prices_path: str = Field(
        default="",
        description=(
            "Opcional, junto con reference_financial_path: CSV mensual con el mismo formato que "
            "global_phytosanitary_prices_b2020.csv. Si se indica, pasa a ser el histórico de "
            "referencia del modelo base nuevo (lags del IPI y exógenas al predecir solo con el IPI)."
        ),
    )
    reference_financial_path: str = Field(
        default="",
        description=(
            "Opcional, junto con reference_prices_path: CSV mensual con el mismo formato que "
            "financial_proxies_monthly.csv (cobre, Brent, EUR/USD, IYM)."
        ),
    )
```

y al final de `TrainResponse`: `    base_version: VersionInfo | None = None`.

`plugin.py`:
- Imports: añadir `ARTIFACT_FOLDER_NAME`, `REFERENCE_FINANCIAL_FILENAME` y `REFERENCE_PRICES_FILENAME` al import de constants; quitar `ModelPersistenceError` del import de excepciones (dejar el resto); añadir `from app.infrastructure.base_model_publisher import persist_retrained_bundle`.
- Clase: `    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model`
- Añadir este método estático justo antes de `train()`:

```python
    @staticmethod
    def _validated_reference_csvs(prices_path: str, financial_path: str) -> list[tuple[str, str]]:
        """Check the optional reference CSVs against the current ones' columns, before training.

        Returns [(local copy, canonical filename)] to add to the new base bundle; [] if none
        were given. The local copies are temp files the caller removes.
        """
        if not prices_path and not financial_path:
            return []
        if not (prices_path and financial_path):
            raise ValueError(
                "reference_prices_path y reference_financial_path se indican juntos: el histórico de "
                "referencia del modelo base se renueva entero."
            )
        # Compare raw file against raw file: load_reference_data() derives extra columns.
        current_prices_path, current_financial_path = model_loader.reference_data_paths()
        current_prices = history.load_monthly_csv(current_prices_path)
        current_financial = history.load_monthly_csv(current_financial_path)
        checked: list[tuple[str, str]] = []
        for field, path, current, filename in (
            ("reference_prices_path", prices_path, current_prices, REFERENCE_PRICES_FILENAME),
            ("reference_financial_path", financial_path, current_financial, REFERENCE_FINANCIAL_FILENAME),
        ):
            fd, copy_path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            with local_file_path(path) as local_path:
                shutil.copyfile(local_path, copy_path)
            try:
                frame = history.load_monthly_csv(copy_path)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                os.unlink(copy_path)
                raise ValueError(f"{field}: no se puede leer como histórico mensual ({exc}).") from exc
            missing = [c for c in current.columns if c not in frame.columns]
            if missing:
                os.unlink(copy_path)
                raise ValueError(f"{field}: faltan las columnas del histórico vigente {missing}.")
            checked.append((copy_path, filename))
        return checked
```

  y añadir `import os` a los imports del módulo.

- `train()`: firma nueva

```python
    def train(
        self, *, data_path: str, mlflow_run_id: str = "",
        reference_prices_path: str = "", reference_financial_path: str = "",
    ) -> TrainResponse:
```

  Docstring: sustituir desde `The retrained artifact` hasta el final por:

```text
        panel it reproduces horizon_6_ridge.pkl and final_evaluation.json exactly. The retrained
        payload is published as the new base model version (and saved to the MLflow run if
        mlflow_run_id is given). reference_prices_path/reference_financial_path, if given, are
        validated before training and replace the bundled reference histories in the new
        version; otherwise they are inherited and upload_warning says so.
```

  Tras `self._require_loaded()`, añadir:

```python
        references = self._validated_reference_csvs(reference_prices_path, reference_financial_path)
```

  Sustituir el bloque desde `        upload_warning = None` hasta el final del `else:` de MLflow (el `finally` con `shutil.rmtree(mlflow_tmp, ...)` incluido) por:

```python
        tracker = BaseMLflowTracker(mlflow_run_id)
        bundle_dir = tempfile.mkdtemp(prefix="ml15_train_")
        try:
            joblib.dump(new_payload, f"{bundle_dir}/{MODEL_FILENAME}")
            for copy_path, filename in references:
                shutil.copyfile(copy_path, f"{bundle_dir}/{filename}")

            def _upload_to_mlflow() -> None:
                tracker.log_params({"model_kind": "ridge", "alpha": ALPHA})
                tracker.log_metrics({k: v for k, v in metrics.items() if v is not None})
                tracker.upload_artifacts(bundle_dir, artifact_path="model")

            persisted = persist_retrained_bundle(
                bundle_dir, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=metrics, mlflow_run_id=mlflow_run_id, upload_to_mlflow=_upload_to_mlflow,
            )
        finally:
            shutil.rmtree(bundle_dir, ignore_errors=True)
            for copy_path, _ in references:
                os.unlink(copy_path)

        warnings = [persisted.upload_warning] if persisted.upload_warning else []
        if not references:
            warnings.append(
                "No se han renovado los históricos exógenos de referencia (reference_prices_path / "
                "reference_financial_path): el modelo base nuevo hereda los de la versión anterior."
            )
        upload_warning = " ".join(warnings) or None
```

  (El `if not mlflow_run_id: raise ModelPersistenceError(...)` desaparece con el bloque.)

  En el `return TrainResponse(...)`: `detail=f"Reentrenamiento completado (Ridge, procedimiento original). Modelo base {persisted.base_version['version']}."`, mantener `upload_warning=upload_warning,` y añadir `base_version=persisted.base_version,`.

  Nota: `test_only_one_reference_csv_is_rejected` falla antes de leer el CSV de entrenamiento, por eso su `data_path` no existe.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml15_training.py tests/unit/test_ml15_wine_ipi_price_forecast.py tests/unit/test_ml15_history.py tests/unit/test_ml15_preprocessing.py tests/unit/test_mlflow_persistence_contract.py tests/unit/test_user_model_isolation.py -q`
Expected: todos en verde (los dos tests de CSV de referencia se saltan si no hay artefactos locales de ml15; decirlo en el informe).

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 11: ml14 publica su base; la base acepta `scalers.json`

**Files:**
- Modify: `app/plugins/ml14_wine_phyto_price_forecast/plugin.py`
- Modify: `app/plugins/ml14_wine_phyto_price_forecast/model_loader.py` (`load_artifact_bundle`)
- Modify: `app/plugins/ml14_wine_phyto_price_forecast/train_dto.py`
- Modify: `tests/unit/test_ml14_training.py` (`test_train_uploads_only_to_mlflow_and_the_model_is_served_by_run`, `test_train_fails_when_mlflow_upload_fails`)
- Modify: `tests/unit/test_ml14_wine_phyto_price_forecast.py` (`test_train_without_mlflow_run_id_returns_422`)

**Interfaces:**
- Consumes: `persist_retrained_bundle` (Tarea 3), `VersionInfo` (Tarea 1), `model_loader.save_user_bundle(...)`, `model_loader.load_user_bundle(directory)`, `upload_artifacts_to_mlflow(artifact_dir, mlflow_run_id, metrics, params)`.
- Produces: `Ml14WinePhytoPriceForecastPlugin.BASE_ARTIFACT_FOLDER = "ml14_wine_phyto_price_forecast"`; `load_artifact_bundle()` usa `scalers.json` si la versión activa lo tiene; `TrainResponse.base_version`.

- [ ] **Step 1: Write the failing tests**

En `tests/unit/test_ml14_training.py`, sustituir `test_train_uploads_only_to_mlflow_and_the_model_is_served_by_run` por una versión que conserve todas sus aserciones actuales (las del run y la del `predict_inline` con `mlflow_run_id="run-1"`) y añada al final:

```python
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id == "run-1"
    v1 = isolated_base_store / "ml14_wine_phyto_price_forecast" / "versions" / "v1"
    assert (v1 / "gru_model.pt").exists() and (v1 / "scalers.json").exists()
```

añadiendo `isolated_base_store` a sus parámetros. Sustituir `test_train_fails_when_mlflow_upload_fails` por:

```python
def test_mlflow_failure_is_a_warning_and_the_base_is_published(instance, raw_csv, fast, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=raw_csv, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v1"


def test_train_without_run_publishes_the_base(instance, raw_csv, fast):
    resp = instance.train(data_path=raw_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id is None


def test_a_promoted_base_loads_with_its_own_scalers(instance, raw_csv, fast, monkeypatch, isolated_base_store):
    from app.infrastructure.artifact_store import ArtifactStore
    from app.plugins.ml14_wine_phyto_price_forecast import model_loader

    resp = instance.train(data_path=raw_csv, mlflow_run_id="")
    store = ArtifactStore("ml14_wine_phyto_price_forecast", root=isolated_base_store, use_s3=False)
    assert store.active_version() == resp.base_version.version   # shared with the publisher
    monkeypatch.setattr(model_loader, "_store", store)
    monkeypatch.setattr(model_loader, "_build_scalers_and_features",
                        MagicMock(side_effect=AssertionError("v1 must not refit scalers")))
    bundle = model_loader.load_artifact_bundle()
    assert bundle["feature_columns"] == _features()
```

En `tests/unit/test_ml14_wine_phyto_price_forecast.py`, sustituir `test_train_without_mlflow_run_id_returns_422` entera por:

```python
def test_train_without_mlflow_run_id_is_accepted(client):
    """mlflow_run_id is optional: every /train publishes the new base model (spec 2026-10-14)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv"})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml14_training.py tests/unit/test_ml14_wine_phyto_price_forecast.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

`model_loader.py`, al principio de `load_artifact_bundle()` (tras el docstring) y ampliando el docstring:

```python
def load_artifact_bundle() -> dict:
    """Load the base GRU of the active version.

    A version published by /train carries its own scalers (scalers.json, see save_user_bundle)
    and is loaded as such. The delivered v0 has none: its input/target scalers are refit from
    the bundled reference dataset, as the AI team's predict did.

    _store.path(filename) only reaches out to S3 for a given file if it isn't already present
    locally (and only if STORAGE_BUCKET is set) — matches the pattern used by other forecast
    plugins in this repo (e.g. ml16, ml23).
    """
    try:
        scalers_path = _store.path(USER_SCALERS_FILENAME)
    except FileNotFoundError:
        scalers_path = None
    if scalers_path is not None and scalers_path.exists():
        return load_user_bundle(_store.local_dir)

    feature_columns, input_scaler, target_scaler = _build_scalers_and_features()
    ...  # resto de la función, sin cambios
```

(Si `USER_SCALERS_FILENAME` no está importado en `model_loader.py`, añadirlo al import de constants; `save_user_bundle` ya lo usa, así que debería estar.)

`train_dto.py`: añadir `from app.application.dto.base_version_dto import VersionInfo`; quitar el import de `MlflowRunId`; `mlflow_run_id: MlflowRunId = Field(...)` pasa a:

```python
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
    )
```

En `TrainResponse`, añadir al final:

```python
    base_version: VersionInfo | None = Field(
        default=None, description="Versión del modelo base publicada por este reentreno"
    )
```

y si su campo `mlflow_run_id: str` es obligatorio (línea ~51), pasarlo a `mlflow_run_id: str = ""`.

`plugin.py`:
- Imports: añadir `ARTIFACT_FOLDER_NAME` al import de constants; quitar `ModelPersistenceError`; añadir `from app.infrastructure.base_model_publisher import persist_retrained_bundle`.
- Clase: `    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model`
- Docstring de `train()`: sustituir `The served model is never touched; the result lives only in its MLflow run (ModelPersistenceError → 502 if the upload fails).` por `The served model is never mutated: the bundle (gru_model.pt + scalers.json) is published as the new base model version and, if mlflow_run_id is given, also saved to that run.`
- Sustituir el bloque `try: ... except Exception as exc: ... raise ModelPersistenceError(...) from exc` (dentro del `artifact_tmp`) por:

```python
        params = {"model": MODEL_NAME, "horizon_weeks": HORIZON_WEEKS, "seq_len": SEQ_LEN,
                  "epochs_max": TRAIN_EPOCHS, "batch_size": TRAIN_BATCH_SIZE,
                  "test_ratio": TEST_RATIO, "n_seeds": TRAIN_N_SEEDS,
                  "canonical_seed": TRAIN_SEED_BASE, "target_mode": "drift_residual_h"}
        try:
            model_loader.save_user_bundle(Path(artifact_tmp), result.model, result.feature_columns,
                                          result.input_scaler, result.target_scaler)
            persisted = persist_retrained_bundle(
                artifact_tmp, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=result.metrics, mlflow_run_id=mlflow_run_id,
                upload_to_mlflow=lambda: upload_artifacts_to_mlflow(
                    artifact_tmp, mlflow_run_id, metrics=result.metrics, params=params),
            )
        finally:
            shutil.rmtree(artifact_tmp, ignore_errors=True)
```

  (el `finally` que ya existe se conserva una sola vez).
- `return TrainResponse(...)`: `upload_warning=persisted.upload_warning,` y añadir `base_version=persisted.base_version,`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml14_training.py tests/unit/test_ml14_wine_phyto_price_forecast.py tests/unit/test_ml14_preprocessing.py tests/unit/test_user_model_isolation.py -q`
Expected: todos en verde

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 12: ml18 publica su base al reentrenar

**Files:**
- Modify: `app/plugins/ml18_meat_spatial_price_forecast/plugin.py`
- Modify: `app/plugins/ml18_meat_spatial_price_forecast/train_dto.py`
- Modify: `tests/unit/test_ml18_training.py` (`test_train_uploads_only_to_mlflow_and_the_model_is_served_by_run`, `test_train_fails_when_mlflow_upload_fails`)
- Modify: `tests/unit/test_ml18_meat_spatial_price_forecast.py` (`test_train_without_mlflow_run_id_returns_422`)
- Test: `tests/unit/test_ml18_base_promotion.py` (no necesita tensorflow)

**Interfaces:**
- Consumes: `persist_retrained_bundle` (Tarea 3), `VersionInfo` (Tarea 1), `training.save_training_artifacts(dir, result)`, `upload_artifacts_to_mlflow(artifact_dir, mlflow_run_id, metrics, params, history)`.
- Produces: `Ml18MeatSpatialPriceForecastPlugin.BASE_ARTIFACT_FOLDER = "ml18_meat_spatial_price_forecast"`; `TrainResponse.base_version`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_ml18_base_promotion.py` (con `train_gru` y `save_training_artifacts` sustituidos, para no depender de tensorflow):

```python
"""ml18: every /train publishes the retrained GRU as the new base (spec 2026-10-14)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.plugins.ml18_meat_spatial_price_forecast import mlflow_utils
from app.plugins.ml18_meat_spatial_price_forecast import plugin as plugin_mod
from app.plugins.ml18_meat_spatial_price_forecast import training
from app.plugins.ml18_meat_spatial_price_forecast.constants import ARTIFACT_FOLDER_NAME, MODEL_FILENAME


@pytest.fixture
def fake_training(monkeypatch):
    monkeypatch.setattr(training, "train_gru", lambda path, on_epoch=None: SimpleNamespace(metrics={"mae": 0.1}))
    monkeypatch.setattr(training, "save_training_artifacts",
                        lambda directory, result: (directory / MODEL_FILENAME).write_bytes(b"gru"))


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "panel.csv"
    path.write_text("Fecha;CCAA\n")
    return str(path)


def test_declares_its_base_folder():
    assert plugin_mod.Ml18MeatSpatialPriceForecastPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_the_new_base(fake_training, csv_path, isolated_base_store):
    resp = plugin_mod.Ml18MeatSpatialPriceForecastPlugin().train(data_path=csv_path, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    assert (isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1" / MODEL_FILENAME).read_bytes() == b"gru"


def test_mlflow_failure_is_a_warning(fake_training, csv_path, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml18MeatSpatialPriceForecastPlugin().train(data_path=csv_path, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.mlflow_run_id == "run-1"
```

Si `TrainResponse` de ml18 exige más campos de métricas que `mae` (`**result.metrics`), poner en `SimpleNamespace(metrics=…)` exactamente los campos obligatorios de `app/plugins/ml18_meat_spatial_price_forecast/train_dto.py::TrainResponse` (leerlo antes de ejecutar el test).

En `tests/unit/test_ml18_training.py`: en `test_train_uploads_only_to_mlflow_and_the_model_is_served_by_run`, añadir el parámetro `isolated_base_store` y al final `assert resp.base_version.version == "v1"`. Sustituir `test_train_fails_when_mlflow_upload_fails` por:

```python
def test_mlflow_failure_is_a_warning(csv_path, fast, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml18MeatSpatialPriceForecastPlugin().train(data_path=csv_path, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v1"
```

En `tests/unit/test_ml18_meat_spatial_price_forecast.py`, sustituir `test_train_without_mlflow_run_id_returns_422` entera por:

```python
def test_train_without_mlflow_run_id_is_accepted(client):
    """mlflow_run_id is optional: every /train publishes the new base model (spec 2026-10-14)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv"})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml18_base_promotion.py tests/unit/test_ml18_meat_spatial_price_forecast.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

`train_dto.py`: mismos cambios que en ml14 — import de `VersionInfo`, quitar `MlflowRunId`, `mlflow_run_id: str = Field(default="", description="Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él (trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base.")`, `TrainResponse.mlflow_run_id: str = ""` y al final de `TrainResponse`:

```python
    base_version: VersionInfo | None = Field(
        default=None, description="Versión del modelo base publicada por este reentreno"
    )
```

`plugin.py`:
- Imports: añadir `ARTIFACT_FOLDER_NAME` al import de constants; quitar `ModelPersistenceError`; añadir `from app.infrastructure.base_model_publisher import persist_retrained_bundle`.
- Clase: `    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model`
- Docstring de `train()`: sustituir `A new model is trained: the served one is never touched. The result lives only in its MLflow run; if it cannot be uploaded, ModelPersistenceError (→ 502).` por `A new model is trained (the served one is never mutated) and published as the new base model version; if mlflow_run_id is given it is also saved to that run.`
- Sustituir el bloque `try: ... except Exception as exc: ... raise ModelPersistenceError(...) from exc` por:

```python
        params = {"model": "GRU", "lookback": LOOKBACK, "epochs_max": TRAIN_EPOCHS,
                  "batch_size": TRAIN_BATCH_SIZE, "train_ratio": TRAIN_RATIO,
                  "val_ratio": VAL_RATIO, "seed": TRAIN_SEED}
        try:
            training.save_training_artifacts(Path(artifact_tmp), result)
            persisted = persist_retrained_bundle(
                artifact_tmp, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=result.metrics, mlflow_run_id=mlflow_run_id,
                upload_to_mlflow=lambda: upload_artifacts_to_mlflow(
                    artifact_tmp, mlflow_run_id, metrics=result.metrics, params=params, history=history),
            )
        finally:
            shutil.rmtree(artifact_tmp, ignore_errors=True)
```

- `return TrainResponse(...)`: `upload_warning=persisted.upload_warning,` y añadir `base_version=persisted.base_version,`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml18_base_promotion.py tests/unit/test_ml18_meat_spatial_price_forecast.py tests/unit/test_ml18_training.py tests/unit/test_user_model_isolation.py -q`
Expected: verde (`test_ml18_training.py` se salta entero sin tensorflow en este WSL; decirlo en el informe)

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 13: ml21 publica su base y el CSV de entrenamiento como panel

**Files:**
- Modify: `app/plugins/ml21_cereals_price_spatial/plugin.py` (clase, imports, `train()`)
- Modify: `app/plugins/ml21_cereals_price_spatial/train_dto.py`
- Modify: `tests/unit/test_ml21_cereals_price_spatial.py` (tests de API de `/train` que manden o exijan `mlflow_run_id`)
- Test: `tests/unit/test_ml21_base_promotion.py`

**Interfaces:**
- Consumes: `persist_retrained_bundle` (Tarea 3), `VersionInfo` (Tarea 1), `DATASET_FILENAME = "dataset_entrenamiento_final.csv"`, `self._panel_df` (panel vigente cargado).
- Produces: `Ml21CerealsPriceSpatialPlugin.BASE_ARTIFACT_FOLDER = "ml21_cereals_price_spatial"`; el bundle publicado incluye `dataset_entrenamiento_final.csv` = el CSV de entrenamiento tal cual; `TrainResponse.base_version`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_ml21_base_promotion.py`:

```python
"""ml21: /train publishes the 6 models + metadata + the training CSV as panel (spec 2026-10-14)."""
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import sklearn.model_selection
from app.plugins.ml21_cereals_price_spatial import plugin as plugin_mod
from app.plugins.ml21_cereals_price_spatial import preprocessing
from app.plugins.ml21_cereals_price_spatial.constants import ARTIFACT_FOLDER_NAME, DATASET_FILENAME

PANEL_COLUMNS = ["date", "provincia", "cereal_predominante", "precio_provincial_lag_1", "x1"]


class _FastSearch:
    """GridSearchCV stand-in: fits the estimator once with few trees (same code path, seconds)."""

    def __init__(self, estimator, param_grid, **_kwargs):
        self.estimator = estimator
        self.best_params_ = {}
        self.best_score_ = 0.0

    def fit(self, X, y):
        if "n_estimators" in self.estimator.get_params():
            self.estimator.set_params(n_estimators=5)
        self.best_estimator_ = self.estimator.fit(X, y)
        return self


@pytest.fixture
def fast_training(monkeypatch):
    rng = np.random.default_rng(0)

    def fake_prepare(df, h, task):
        X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60)})
        y = pd.Series(rng.normal(size=60)) if task == "regresion" else pd.Series([0, 1] * 30)
        return X.iloc[:48], X.iloc[48:], y.iloc[:48], y.iloc[48:]

    monkeypatch.setattr(preprocessing, "prepare_train_test", fake_prepare)
    monkeypatch.setattr(sklearn.model_selection, "GridSearchCV", _FastSearch)


@pytest.fixture
def instance():
    plugin = plugin_mod.Ml21CerealsPriceSpatialPlugin()
    plugin._panel_df = pd.DataFrame(columns=PANEL_COLUMNS)  # pylint: disable=protected-access
    return plugin


def _csv(tmp_path, columns=PANEL_COLUMNS):
    df = pd.DataFrame({c: [1.0, 2.0] for c in columns})
    if "date" in columns:
        df["date"] = ["2020-01-01", "2021-02-01"]
    for col in ("provincia", "cereal_predominante"):
        if col in columns:
            df[col] = ["Burgos", "Soria"]
    path = tmp_path / "panel.csv"
    df.to_csv(path, index=False)
    return path


def test_declares_its_base_folder():
    assert plugin_mod.Ml21CerealsPriceSpatialPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_publishes_models_and_the_training_csv_as_panel(instance, fast_training, tmp_path,
                                                               isolated_base_store):
    csv = _csv(tmp_path)
    resp = instance.train(data_path=str(csv), mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    raw_info = (v1 / "version.json").read_text()
    assert "NaN" not in raw_info and json.loads(raw_info)["metrics"]["mae_h1"] is not None
    assert (v1 / DATASET_FILENAME).read_bytes() == csv.read_bytes()
    metadata = json.loads((v1 / "model_metadata.json").read_text())
    assert set(metadata["selected_models"]) == {"H1", "H2", "H3"}
    for h in (1, 2, 3):
        assert (v1 / f"datagia_best_h{h}_reg.joblib").exists() and (v1 / f"datagia_best_h{h}_clf.joblib").exists()


def test_csv_without_the_panel_columns_is_rejected_before_training(instance, tmp_path, monkeypatch):
    monkeypatch.setattr(preprocessing, "prepare_train_test",
                        MagicMock(side_effect=AssertionError("must not train")))
    csv = _csv(tmp_path, columns=["date", "provincia", "cereal_predominante"])
    with pytest.raises(ValueError, match="precio_provincial_lag_1"):
        instance.train(data_path=str(csv), mlflow_run_id="")


def test_mlflow_failure_is_a_warning(instance, fast_training, tmp_path, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=str(_csv(tmp_path)), mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.mlflow_run_id == "run-1"
```

En `tests/unit/test_ml21_cereals_price_spatial.py`: añadir

```python
def test_train_without_mlflow_run_id_is_accepted(client):
    """mlflow_run_id is optional: every /train publishes the new base model (spec 2026-10-14)."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x.csv"})
    assert resp.status_code == 200
```

y borrar cualquier test de ese fichero que espere 422 por falta de `mlflow_run_id` (comprobar con `grep -n "422" tests/unit/test_ml21_cereals_price_spatial.py`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml21_base_promotion.py tests/unit/test_ml21_cereals_price_spatial.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

`train_dto.py`: añadir `from app.application.dto.base_version_dto import VersionInfo`; quitar `MlflowRunId`; `mlflow_run_id: MlflowRunId` pasa a

```python
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base y "
            "este CSV su panel (dataset_entrenamiento_final.csv)."
        ),
    )
```

y al final de `TrainResponse`: `    base_version: VersionInfo | None = None`.

`plugin.py`:
- Imports: añadir `ARTIFACT_FOLDER_NAME` y `DATASET_FILENAME` al import de constants si no están; añadir `from app.infrastructure.base_model_publisher import persist_retrained_bundle`. Si tras el cambio `ModelPersistenceError` ya no se usa en el módulo, quitarlo del import.
- Clase: `    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model`
- `train()`: firma `def train(self, *, data_path: str = "", mlflow_run_id: str = "") -> Any:`. Docstring, añadir al final:

```text
        The 6 models, model_metadata.json and the training CSV itself (as the new
        dataset_entrenamiento_final.csv panel, so predict_inline finds the new months) are
        published as the new base model version; if mlflow_run_id is given they are also
        saved to that run. The CSV must have every column of the current panel (400 if not).
```

- Mover `tmp_dir = tempfile.mkdtemp(prefix="ml21_train_")` a justo antes del `with local_file_path(data_path) as local_path:` y sustituir ese `with` por:

```python
        tmp_dir = tempfile.mkdtemp(prefix="ml21_train_")
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
                shutil.copyfile(local_path, os.path.join(tmp_dir, DATASET_FILENAME))
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        if self._panel_df is not None:
            missing = [c for c in self._panel_df.columns if c not in df.columns]
            if missing:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise ValueError(
                    f"El CSV de entrenamiento pasa a ser el panel del modelo base y le faltan "
                    f"columnas del panel vigente: {missing}"
                )
```

  y en el `if "date" not in df.columns: raise ValueError(...)` siguiente, añadir `shutil.rmtree(tmp_dir, ignore_errors=True)` antes del `raise`. Borrar la línea original `tmp_dir = tempfile.mkdtemp(prefix="ml21_train_")` que estaba tras el bloque del `tracker`.

- Sustituir desde `            upload_warning = None` hasta el `raise ModelPersistenceError(...) from exc` del `except` (incluido) por:

```python
            def _upload_to_mlflow() -> None:
                tracker.log_params({
                    "model_type": "extra_trees+xgb+logistic",
                    "horizons": "1,2,3",
                    "cut_date": "2021-01-01",
                    "random_state": "42",
                })
                tracker.log_metrics(
                    {k: v for k, v in all_metrics.items() if not (isinstance(v, float) and np.isnan(v))}
                )
                tracker.upload_artifacts(tmp_dir, artifact_path="model")

            persisted = persist_retrained_bundle(
                tmp_dir, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=all_metrics, mlflow_run_id=mlflow_run_id,
                upload_to_mlflow=_upload_to_mlflow if tracker else None,
            )
```

- Sustituir el comentario `# The retrained models live only in their MLflow run ... no reload here.` por `# The container swaps in a fresh plugin on the new base; train() never reloads itself.`
- `return TrainResponse(...)`: `detail=f"Training completado — 6 modelos entrenados (3H × reg+clf). Modelo base {persisted.base_version['version']}."`, `upload_warning=persisted.upload_warning,` y añadir `base_version=persisted.base_version,`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -p no:flask tests/unit/test_ml21_base_promotion.py tests/unit/test_ml21_cereals_price_spatial.py tests/unit/test_user_model_isolation.py -q`
Expected: todos en verde, incluido el test de `main` que comprueba que ningún `train()` recarga el modelo servido (`grep -rn "reload" tests/unit/test_user_model_download_hygiene.py` para localizarlo y ejecutarlo).

- [ ] **Step 5: Dejar listo**

Run: `git diff --ignore-cr-at-eol --stat`

---

### Task 14: Reglas del repo, README y suite completa

**Files:**
- Modify: `CLAUDE.md` (sección "Reglas que nunca se saltan")
- Modify: `.claude/skills/plugin-integration/SKILL.md` (donde diga que el artefacto base nunca se sobrescribe — `grep -n "nunca se sobrescribe\|never overwrit\|base nunca" .claude/skills/plugin-integration/SKILL.md`)
- Modify: `README.md` (sección de endpoints / reentrenamiento — `grep -n "/train" README.md`)

- [ ] **Step 1: CLAUDE.md**

Añadir a "Reglas que nunca se saltan":

```markdown
- El artefacto base nunca se sobrescribe **en el sitio**. En los modelos con
  `BASE_ARTIFACT_FOLDER` (ml13, 14, 15, 16, 17, 18, 21 y 23) cada `/train` publica una versión
  nueva en `artifacts/fixed/<carpeta>/versions/<vN>/` y mueve `current.json`; `POST /rollback`
  vuelve atrás. En el resto, el modelo reentrenado vive solo en su run de MLflow. Diseño:
  `docs/superpowers/specs/2026-10-14-reentreno-actualiza-base-design.md`.
```

- [ ] **Step 2: Skill `plugin-integration`**

En cada punto donde la skill prohíba tocar la base, añadir la excepción con el mismo texto que en CLAUDE.md, y en la sección del patrón de reentrenamiento un párrafo:

```markdown
**Modelos que promueven su base.** Si el modelo debe reemplazar su base en cada reentreno
(decisión de producto, hoy ml13/14/15/16/17/18/21/23): declarar en la clase
`BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME`, `mlflow_run_id: str = ""` en el `TrainRequest`,
`base_version: VersionInfo | None` en el `TrainResponse`, y persistir con
`app.infrastructure.base_model_publisher.persist_retrained_bundle(bundle_dir, …,
upload_to_mlflow=…)` usando los nombres de fichero canónicos de la base. `train()` nunca recarga
el plugin: lo hace el contenedor con una instancia nueva.
```

- [ ] **Step 3: README**

Añadir en la sección de endpoints:

```markdown
### Modelo base versionado (ml13, 14, 15, 16, 17, 18, 21, 23)

- `POST /models/<id>/train` — `mlflow_run_id` opcional. Cada reentreno correcto se publica como
  nueva versión del modelo base (`artifacts/fixed/<carpeta>/versions/<vN>/`, puntero
  `current.json`) y se sirve al momento. La respuesta incluye `base_version`. Si la subida al run
  de MLflow falla, la base se publica igual y `upload_warning` lo indica.
- `POST /models/<id>/rollback` — cuerpo opcional `{"version": "v2"}`; sin cuerpo vuelve a la
  versión anterior. 404 si la versión no existe, 409 si la vigente es v0, 501 en los demás modelos.
- `GET /models/<id>/stats` — incluye `base_version` (versión, padre, run, fecha, métricas).
- ml15 acepta además `reference_prices_path` y `reference_financial_path` (juntos) para renovar
  sus históricos exógenos de referencia; ml21 publica el CSV de entrenamiento como su panel.
```

- [ ] **Step 4: Suite completa**

La línea base se guardó antes de la Tarea 1 (ver Global Constraints).

Run: `python3 -m pytest -p no:flask tests/unit -q -rf 2>&1 | grep '^FAILED' | sort > /tmp/claude-1000/-mnt-c-Users-PC-Desktop-Repositorios-models-integrations-itacly/0cabd567-c354-4897-9d14-75abf4a53287/scratchpad/after-failures.txt; comm -13 /tmp/claude-1000/-mnt-c-Users-PC-Desktop-Repositorios-models-integrations-itacly/0cabd567-c354-4897-9d14-75abf4a53287/scratchpad/baseline-failures.txt /tmp/claude-1000/-mnt-c-Users-PC-Desktop-Repositorios-models-integrations-itacly/0cabd567-c354-4897-9d14-75abf4a53287/scratchpad/after-failures.txt`
Expected: salida vacía (ningún fallo nuevo respecto a la línea base). Cualquier fallo nuevo se investiga y se arregla antes de cerrar; los fallos de la línea base se citan como preexistentes en el informe.

- [ ] **Step 5: Lint**

Run: `python3 -m flake8 app/infrastructure/artifact_store.py app/infrastructure/base_model_publisher.py app/infrastructure/http/dependencies/container.py app/infrastructure/http/router_factory.py app/application/use_cases/get_stats_use_case.py app/application/dto/base_version_dto.py app/plugins/ml1[345678]_*/plugin.py app/plugins/ml2[13]_*/plugin.py app/plugins/ml14_wine_phyto_price_forecast/model_loader.py`
Expected: sin errores en las líneas que cambia esta rama (`git diff --ignore-cr-at-eol -U0` muestra cuáles son); los avisos en líneas no tocadas son preexistentes.

- [ ] **Step 6: Dejar listo**

Run: `git status --short | grep -v '^ M' ; git diff --ignore-cr-at-eol --stat`
Informar al usuario de los ficheros cambiados y nuevos; él hace el commit.
