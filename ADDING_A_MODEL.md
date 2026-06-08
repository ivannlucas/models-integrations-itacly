# Cómo añadir un nuevo modelo

Guía para desarrolladores. Cubre todo lo necesario desde cero hasta tener el modelo
listo para que el equipo de infraestructura lo despliegue.

---

## Convención de nomenclatura

Todos los plugins siguen el esquema **`ml<ticket>_<sector>_<model_name>`** en snake_case:

```
ml25_wine_sulphites
│    │     └── model_name  → nombre descriptivo del modelo
│    └──────── sector      → dominio o área de negocio
└───────────── ml<ticket>  → ID del ticket de Jira (ML-25 → ml25)
```

| Elemento | Formato | Ejemplo |
|---|---|---|
| Carpeta del plugin | `app/plugins/ml<ticket>_<sector>_<model_name>/` | `app/plugins/ml25_wine_sulphites/` |
| Carpeta de artefactos | `artifacts/ml<ticket>_<sector>_<model_name>/` | `artifacts/ml25_wine_sulphites/` |
| `model_id` en registry | kebab-case sin prefijo `ml` | `wine-sulphites` |
| Rama de trabajo | `feature/ML<ticket>-<descripcion>` | `feature/ML25-wine-sulphites` |

> El `model_id` es el identificador público de la API (`/models/<model_id>/`).
> No lleva el prefijo `ml<ticket>` para mantener las URLs estables si el modelo se reimplementa.

---

## Cómo funciona el sistema

Una única imagen Docker contiene el código de todos los modelos. Cada modelo tiene
su propio despliegue en Kubernetes, pero todos comparten esa misma imagen. La variable
de entorno `MODEL` indica al servidor qué plugin cargar al arrancar:

```
Imagen Docker (todos los plugins)
        │
        ├── Despliegue K8s  MODEL=wine-sulphite   → /models/wine-sulphite/predict
        ├── Despliegue K8s  MODEL=wine-price       → /models/wine-price/predict
        └── Despliegue K8s  MODEL=mi-nuevo-modelo  → /models/mi-nuevo-modelo/predict
```

Sin `MODEL` definida (desarrollo local) el servidor carga todos los modelos del registry.

---

## Ficheros que tú creas

```
app/plugins/mi_modelo/        ← carpeta del plugin (guion bajo)
├── __init__.py
├── constants.py
├── predict_dto.py
├── train_dto.py              ← solo si el plugin implementa train()
├── model_loader.py
├── preprocessing.py
├── postprocessing.py
└── plugin.py

artifacts/mi-modelo/          ← artefactos del modelo (guion)
├── model.pkl
└── metadata.json

tests/unit/test_mi_modelo.py  ← tests del endpoint
```

### Fichero existente que modificas

```
app/registry.py               ← añades una entrada ModelEntry
tests/conftest.py             ← añades las factories del fake
```

---

## Qué hace cada fichero del plugin

| Fichero | Propósito |
|---|---|
| `__init__.py` | Vacío. Marca la carpeta como paquete Python. |
| `constants.py` | `MODEL_ID`, lista de features, nombres de clases, umbral. |
| `predict_dto.py` | Clases Pydantic que definen qué recibe y devuelve `/predict`. |
| `train_dto.py` | Clases Pydantic para `/train`. Solo necesario si el plugin implementa `train()`. |
| `model_loader.py` | Carga `model.pkl` y `metadata.json` usando `ArtifactStore`. |
| `preprocessing.py` | Transforma el dict de entrada en array numpy para el modelo. |
| `postprocessing.py` | Transforma la salida del modelo en el dict de respuesta. |
| `plugin.py` | Clase principal. Implementa `ModelPluginPort`. Une todo lo anterior. |

> **Modelo de referencia:** `app/plugins/wine_sulphite/` es el plugin más completo
> del repo. Úsalo como plantilla.

---

## Paso a paso

### 1. `constants.py`

```python
MODEL_ID = "mi-modelo"           # kebab-case; debe coincidir con el directorio de artifacts

FEATURE_COLUMNS = ["feat_a", "feat_b", "feat_c"]
CLASS_NAMES     = ["clase_0", "clase_1"]
DEFAULT_THRESHOLD = 0.5
```

---

### 2. `predict_dto.py`

El endpoint `/predict` acepta tanto predicción inline (datos en el body) como batch
(ruta a un CSV). El campo `mode` actúa de discriminador.

```python
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class PredictInlineRequest(BaseModel):
    mode: Literal["inline"] = "inline"
    threshold: Optional[float] = None   # override del umbral (opcional)
    feat_a: float
    feat_b: float
    feat_c: float


class PredictBatchRequest(BaseModel):
    mode: Literal["batch"] = "batch"
    data_path: str                      # ruta al CSV de entrada


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id:      str
    threshold:     Optional[float]
    prediction:    int                  # 0 o 1
    confidence:    float
    label:         str
    features_used: list[str]


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id:    str
    predictions: list[dict]
    output_path: Optional[str] = None


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
```

Adapta los campos de `PredictInlineRequest` y `PredictInlineResponse` a tu modelo.

---

### 3. `train_dto.py` *(solo si el plugin implementa `train()`)*

Define los tipos que usa el endpoint `/train`. Los campos de `TrainResponse` deben
coincidir exactamente con las claves del dict que devuelve `plugin.train()`.

```python
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="Ruta al CSV de entrenamiento dentro del contenedor")


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    # Añade aquí las métricas que devuelva tu train():
    mae: float = Field(..., description="MAE en el split de test")
    n_train: int
    n_test: int
    training_time_s: float
    upload_warning: str | None = Field(
        default=None,
        description="Informativo si los artefactos se guardaron en local pero falló el upload a S3",
    )
```

> **`upload_warning`** es opcional pero recomendado: el `ArtifactStore` puede fallar
> al subir a S3 (endpoint caído, credenciales, etc.). Captúralo en `plugin.train()` con
> un `try/except` alrededor del loop de `upload_artifact()` y devuélvelo en el dict de
> retorno para que llegue al cliente sin convertir el entrenamiento en un HTTP 500.

---

### 4. `model_loader.py`

`ArtifactStore` busca los ficheros en `artifacts/<MODEL_ID>/`.
Si la variable de entorno `STORAGE_BUCKET` está definida, los descarga de S3 automáticamente.

```python
import json
import logging
import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.mi_modelo.constants import (
    DEFAULT_THRESHOLD, FEATURE_COLUMNS, MODEL_ID,
)

logger = logging.getLogger(__name__)


def load_model_bundle() -> dict:
    store = ArtifactStore(MODEL_ID)
    model    = joblib.load(store.path("model.pkl"))
    metadata = json.loads(store.path("metadata.json").read_text("utf-8"))

    return {
        "model_id":  metadata.get("model_id",        MODEL_ID),
        "model":     model,
        "features":  metadata.get("feature_columns", FEATURE_COLUMNS),
        "threshold": float(metadata.get("threshold", DEFAULT_THRESHOLD)),
    }
```

---

### 5. `preprocessing.py`

```python
import numpy as np
from app.plugins.mi_modelo.constants import FEATURE_COLUMNS


def build_feature_matrix(features: dict, feature_schema: list[str] = FEATURE_COLUMNS) -> np.ndarray:
    missing = set(feature_schema) - set(features.keys())
    if missing:
        raise ValueError(f"Faltan features: {sorted(missing)}")

    return np.array([[float(features[f]) for f in feature_schema]], dtype=float)
```

---

### 6. `postprocessing.py`

Las claves del dict que devuelves **deben coincidir exactamente** con los campos
de `PredictInlineResponse` / `PredictBatchResponse`.

```python
from app.plugins.mi_modelo.constants import CLASS_NAMES, FEATURE_COLUMNS, MODEL_ID


def build_inline_response(proba: float, threshold: float,
                           features: list[str] = FEATURE_COLUMNS,
                           model_id: str = MODEL_ID) -> dict:
    prediction = int(proba >= threshold)
    return {
        "model_id":      model_id,
        "threshold":     threshold,
        "prediction":    prediction,
        "confidence":    round(proba, 6),
        "label":         CLASS_NAMES[prediction],
        "features_used": features,
    }


def build_batch_response(probas: list[float], threshold: float,
                          model_id: str = MODEL_ID) -> dict:
    predictions = [
        {
            "row":        i,
            "prediction": int(p >= threshold),
            "confidence": round(p, 6),
            "label":      CLASS_NAMES[int(p >= threshold)],
        }
        for i, p in enumerate(probas)
    ]
    return {"model_id": model_id, "predictions": predictions, "output_path": None}
```

---

### 7. `plugin.py`

Implementa los 5 métodos abstractos de `ModelPluginPort`:
`load`, `is_loaded`, `predict_inline`, `predict_batch`, `stats`.

Si el modelo soporta reentrenamiento, sobreescribe también `train()`.
El método base en `ModelPluginPort` lanza `TrainingNotSupportedError` (HTTP 501) si no se sobreescribe.

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.plugins.mi_modelo.constants import FEATURE_COLUMNS, MODEL_ID
from app.plugins.mi_modelo.model_loader import load_model_bundle
from app.plugins.mi_modelo.postprocessing import build_batch_response, build_inline_response
from app.plugins.mi_modelo.preprocessing import build_feature_matrix

logger = logging.getLogger(__name__)


class MiModeloPlugin(ModelPluginPort):

    def __init__(self) -> None:
        self._bundle          = None
        self._predict_count   = 0
        self._last_predict_at = None

    def load(self) -> None:
        self._bundle = load_model_bundle()
        logger.info("MiModeloPlugin loaded: %s", self._bundle["model_id"])

    def is_loaded(self) -> bool:
        return self._bundle is not None

    def _require_bundle(self) -> dict:
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    def predict_inline(self, *, features: dict,
                       model_key: Optional[str] = None,
                       threshold: Optional[float] = None) -> dict:
        bundle = self._require_bundle()
        thr    = threshold if threshold is not None else bundle["threshold"]
        x      = build_feature_matrix(features, feature_schema=bundle["features"])
        proba  = float(bundle["model"].predict_proba(x)[0, 1])

        self._predict_count  += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

        return build_inline_response(proba, thr,
                                     features=bundle["features"],
                                     model_id=bundle["model_id"])

    def predict_batch(self, *, data_path: str) -> dict:
        bundle = self._require_bundle()
        df     = pd.read_csv(data_path)
        x      = df[bundle["features"]].values.astype(float)
        probas = bundle["model"].predict_proba(x)[:, 1].tolist()

        self._predict_count  += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

        return build_batch_response(probas, bundle["threshold"],
                                    model_id=bundle["model_id"])

    def stats(self) -> StatsResponse:
        bundle = self._require_bundle()
        return StatsResponse(
            model_name      = bundle["model_id"],
            model_type      = "binary_classification",
            framework       = "scikit-learn",
            artifact_path   = f"artifacts/{MODEL_ID}",
            input_schema    = {f: "float" for f in bundle["features"]},
            output_schema   = {"prediction": "int (0|1)", "confidence": "float", "label": "str"},
            predict_count   = self._predict_count,
            last_predict_at = self._last_predict_at,
        )

    # ── Entrenamiento (opcional) ──────────────────────────────────────────────

    def train(self, *, data_path: str) -> dict:
        import time
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from app.plugins.mi_modelo.model_loader import get_artifacts_dir, upload_artifact

        t0 = time.perf_counter()
        df = pd.read_csv(data_path, sep=None, engine="python")
        # … preprocesado, fit, métricas …

        artifacts_dir = get_artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifacts_dir / "model.pkl")

        upload_warning: str | None = None
        try:
            upload_artifact("model.pkl")
            upload_artifact("metadata.json")
        except Exception as exc:
            upload_warning = f"Artefactos guardados en local; fallo en S3: {exc}"

        self.load()
        return {
            "detail": "Training completed",
            "mae": round(mae, 4),
            "n_train": n_train,
            "n_test": n_test,
            "training_time_s": round(time.perf_counter() - t0, 1),
            "upload_warning": upload_warning,
        }
```

> Las claves del dict de retorno de `train()` deben coincidir exactamente con los campos
> de `TrainResponse` definidos en `train_dto.py`.

---

### 8. `app/registry.py` — registrar el modelo

Añade al final del bloque de imports y al final de la lista `REGISTRY`:

```python
# imports
from app.plugins.ml<ticket>_mi_sector_mi_modelo.plugin import MiModeloPlugin
from app.plugins.ml<ticket>_mi_sector_mi_modelo.predict_dto import (
    PredictBatchResponse  as MiModeloBatchResp,
    PredictInlineResponse as MiModeloInlineResp,
    PredictRequest        as MiModeloRequest,
    PredictResponse       as MiModeloResponse,
)
# Solo si el plugin implementa train():
from app.plugins.ml<ticket>_mi_sector_mi_modelo.train_dto import (
    TrainRequest  as MiModeloTrainReq,
    TrainResponse as MiModeloTrainResp,
)

# entrada en REGISTRY
ModelEntry(
    model_id              = "mi-modelo",
    prefix                = "/models/mi-modelo",
    version               = "1.0.0",
    plugin_class          = MiModeloPlugin,
    predict_request_type  = MiModeloRequest,
    predict_response_type = MiModeloResponse,
    batch_response_class  = MiModeloBatchResp,
    inline_response_class = MiModeloInlineResp,
    extra_predict_exceptions = (),
    # Omite las dos líneas siguientes si el plugin NO implementa train()
    train_request_type    = MiModeloTrainReq,
    train_response_type   = MiModeloTrainResp,
),
```

> `extra_predict_exceptions`: si tu plugin lanza excepciones de dominio propias
> (definidas en `app/domain/services/exceptions.py`) que deban devolver HTTP 422
> en lugar de 500, añádelas aquí como tupla.
>
> Si omites `train_request_type` / `train_response_type`, el endpoint `/train`
> sigue existiendo pero devuelve HTTP 501 y usa el `TrainRequest` genérico
> (solo campo `data_path`). Swagger no mostrará un schema de respuesta tipado.

---

### 9. `tests/conftest.py` — factories del fake

Los tests nunca cargan el modelo real. Añade dos funciones que devuelven
respuestas de ejemplo compatibles con tus DTOs:

```python
def _mi_modelo_inline(plugin, *, features, model_key, threshold):
    return {
        "model_id":      "mi-modelo",
        "threshold":     threshold,
        "prediction":    1,
        "confidence":    0.87,
        "label":         "clase_1",
        "features_used": ["feat_a", "feat_b", "feat_c"],
    }


def _mi_modelo_batch(plugin, *, data_path):
    return {
        "model_id":    "mi-modelo",
        "predictions": [{"row": 0, "prediction": 1, "confidence": 0.87, "label": "clase_1"}],
        "output_path": None,
    }


# En el dict FAKE_FACTORIES:
"mi-modelo": (_mi_modelo_inline, _mi_modelo_batch),
```

---

### 10. `tests/unit/test_mi_modelo.py` — tests del endpoint

```python
PREFIX = "/models/mi-modelo"

INLINE_PAYLOAD = {
    "mode":   "inline",
    "feat_a": 1.2,
    "feat_b": 3.4,
    "feat_c": 5.6,
}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"]  == "mi-modelo"
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == "mi-modelo"


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"]   == "mi-modelo"
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_inline_missing_field(client):
    payload = {**INLINE_PAYLOAD}
    del payload["feat_a"]
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/data.csv"})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "mi-modelo"


def test_train_returns_501(client):
    assert client.post(f"{PREFIX}/train").status_code == 501
```

---

## Artefactos

### Local (desarrollo)

```
artifacts/
└── mi-modelo/           ← mismo valor que MODEL_ID en constants.py
    ├── model.pkl
    └── metadata.json
```

`metadata.json` mínimo:

```json
{
  "model_id": "mi-modelo",
  "version": "1.0.0",
  "feature_columns": ["feat_a", "feat_b", "feat_c"],
  "threshold": 0.5
}
```

### Producción (S3)

```
s3://<STORAGE_BUCKET>/artifacts/fixed/mi-modelo/model.pkl
s3://<STORAGE_BUCKET>/artifacts/fixed/mi-modelo/metadata.json
```

El servidor los descarga automáticamente al arrancar si `STORAGE_BUCKET` está definido.
Entrega los ficheros al equipo de infra para que los suban.

---

## Qué hace el equipo de infra (no tienes que tocarlo)

Una vez mergeado tu PR, infra copia la carpeta Kustomize de otro modelo y cambia:

1. `namePrefix` → `mi-modelo-`
2. `MODEL="mi-modelo"` en el `configMapGenerator`
3. La ruta en `http-route.yaml` → `/mi-modelo`
4. El nombre del servicio → `mi-modelo-svc`
5. El tag de la imagen → la versión que incluye tu plugin

No tienes que tocar `Dockerfile`, `bitbucket-pipelines.yml` ni ningún fichero de Kustomize.

---

## Flujo de trabajo con Bitbucket

### Crear la rama

Las ramas siguen el convenio `feature/<ticket>` o `fix/<ticket>`, donde `<ticket>`
es el identificador de Jira (p.ej. `ML-26`). Puedes crearla de dos formas:

**Desde Bitbucket (recomendado al trabajar desde un ticket de Jira):**
1. Abre el ticket en Jira.
2. En el panel derecho busca **"Create branch"** → se abre Bitbucket con el nombre
   ya sugerido a partir del ticket.
3. Selecciona como base la rama `develop` (nunca `master` directamente).
4. Confirma la creación.
5. Clona o haz fetch en local: `git fetch && git checkout feature/ml-<ticket>`.

**Desde la línea de comandos:**
```bash
git checkout develop
git pull
git checkout -b feature/ml-26-mi-nuevo-modelo
```

> **Convención de nombres:**
> - Nuevas funcionalidades → `feature/ml-<ticket>-<descripcion-corta>`
> - Correcciones de bugs   → `fix/ml-<ticket>-<descripcion-corta>`
> - Usar kebab-case, sin espacios ni mayúsculas.

---

## Comprobaciones locales antes de hacer push

El pipeline ejecuta **4 pasos** en cada push a `feature/*` o `fix/*` y al abrir un PR.
Si cualquiera de ellos falla, el PR queda bloqueado. Reprodúcelos en local antes de
hacer push para detectar los errores sin esperar al pipeline.

```
Pipeline
  ├── 1. Lint style          → flake8
  ├── 2. Analyze             → SonarQube
  ├── 3. Security scan — credentials   → git-secrets-scan
  └── 4. Security scan — dependencies  → dependency-scanner
```

---

### Paso 1 — Lint style (flake8)

El pipeline ejecuta exactamente:

```bash
pip install flake8
flake8 . --extend-exclude=dist,build --show-source --statistics
```

Ejecuta el mismo comando en local. La configuración de reglas está en [.flake8](.flake8)
(línea máxima 120, E501 ignorado).

Errores más frecuentes que bloquean el pipeline:

| Código | Causa | Solución rápida |
|---|---|---|
| `F401` | Import no usado | Elimina el import |
| `E302` | Faltan 2 líneas en blanco antes de función/clase | Añade las líneas vacías |
| `W291` | Espacios al final de línea | Activa "trim trailing whitespace" en tu editor |
| `E711` | `== None` en lugar de `is None` | Usa `is None` / `is not None` |
| `W503` | Salto de línea antes de operador binario | Mueve el operador al final de la línea anterior |

---

### Paso 2 — SonarQube

El pipeline lanza un análisis estático de código y cobertura. No puedes ejecutar
SonarQube en local, pero sí puedes generar los informes que consume:

```bash
python -m pytest tests/unit/ \
    --junitxml=pytest-report.xml \
    --cov=app \
    --cov-report=xml:coverage.xml \
    -v
```

Si todos los tests pasan en verde, SonarQube no encontrará fallos de cobertura.
Los tests unitarios **no necesitan artefactos reales** — usan el `FakePlugin` del `conftest.py`.

---

### Paso 3 — Security scan: credenciales (`git-secrets-scan`)

El pipeline escanea todo el historial de commits en busca de credenciales,
tokens, contraseñas o claves de API hardcodeadas en el código.

**Regla:** nunca escribas credenciales directamente en el código ni en ficheros
que se van a commitear. Usa variables de entorno o el fichero `.env` (que ya está
en `.gitignore`).

Patrones que **bloquean** el pipeline:

```python
# MAL — el pipeline lo detecta y falla
AWS_SECRET = "AKIAIOSFODNN7EXAMPLE"
password = "mi_contraseña_secreta"
api_key = "sk-1234abcd..."

# BIEN — valor viene del entorno
import os
aws_secret = os.getenv("AWS_SECRET_ACCESS_ID")
```

Comprobación manual antes de hacer push:

```bash
# Revisa que no hay credenciales en los ficheros que vas a commitear
git diff --staged
git log --oneline -5   # revisa también commits recientes
```

---

### Paso 4 — Security scan: dependencias (`dependency-scanner`)

El pipeline analiza `requirements.txt` en busca de paquetes con vulnerabilidades
conocidas (CVEs). Si una dependencia tiene una CVE publicada, el step falla.

Comprobación local con `pip-audit` (equivalente):

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

Si aparece una CVE:
- Actualiza la versión del paquete afectado en `requirements.txt` a una versión
  parcheada (consulta el advisory del error).
- Si no existe versión parcheada, consulta con el equipo antes de hacer push.

---

### Paso 5 — Arranque en local

```bash
# Simula el comportamiento en Kubernetes (carga solo tu modelo)
MODEL=mi-modelo python main.py
```

Verifica los tres endpoints de tu modelo:

```bash
curl http://localhost:8000/models/mi-modelo/health

curl -X POST http://localhost:8000/models/mi-modelo/predict \
  -H "Content-Type: application/json" \
  -d '{"mode":"inline","feat_a":1.2,"feat_b":3.4,"feat_c":5.6}'

curl http://localhost:8000/models/mi-modelo/stats
```

---

### Resumen: checklist pre-push

```bash
# 1. Lint
flake8 . --extend-exclude=dist,build --show-source --statistics

# 2. Tests + cobertura
python -m pytest tests/unit/ --junitxml=pytest-report.xml --cov=app --cov-report=xml:coverage.xml -v

# 3. Credenciales — revisión manual
git diff --staged    # ninguna contraseña/token hardcodeado

# 4. Dependencias
pip-audit -r requirements.txt

# 5. Arranque
MODEL=mi-modelo python main.py
curl http://localhost:8000/models/mi-modelo/health
```

Si los cinco pasan sin errores, el pipeline no fallará por causas de código.

---

## Checklist completo

```
RAMA
□ 0.  Crear rama feature/ml-<ticket>-mi-modelo desde develop

NOMENCLATURA
□ 0.  Carpeta plugin:     app/plugins/ml<ticket>_<sector>_<model_name>/
      Carpeta artefactos: artifacts/ml<ticket>_<sector>_<model_name>/
      model_id (registry/URL): kebab-case sin prefijo ml (ej. "wine-sulphites")

PLUGIN
□ 1.  app/plugins/ml<ticket>_<sector>_<model_name>/__init__.py    (vacío)
□ 2.  app/plugins/ml<ticket>_<sector>_<model_name>/constants.py
□ 3.  app/plugins/ml<ticket>_<sector>_<model_name>/predict_dto.py
□ 4.  app/plugins/ml<ticket>_<sector>_<model_name>/train_dto.py   (solo si implementa train())
□ 5.  app/plugins/ml<ticket>_<sector>_<model_name>/model_loader.py
□ 6.  app/plugins/ml<ticket>_<sector>_<model_name>/preprocessing.py
□ 7.  app/plugins/ml<ticket>_<sector>_<model_name>/postprocessing.py
□ 8.  app/plugins/ml<ticket>_<sector>_<model_name>/plugin.py

REGISTRO
□ 9.  app/registry.py  → imports predict_dto + train_dto (si aplica) + ModelEntry
                          con train_request_type / train_response_type si train() existe

TESTS
□ 10. tests/conftest.py             → factories inline/batch + entrada en FAKE_FACTORIES
□ 11. tests/unit/test_mi_modelo.py  → health, stats, predict inline+batch, train (501 o 200)

ARTEFACTOS LOCAL
□ 12. artifacts/ml<ticket>_<sector>_<model_name>/model.pkl + metadata.json colocados

COMPROBACIONES LOCALES (pre-push — replica el pipeline)
□ 13. flake8 . --extend-exclude=dist,build --show-source --statistics  → 0 errores
□ 14. python -m pytest tests/unit/ --cov=app -v                        → todo verde
□ 15. git diff --staged                  → sin credenciales hardcodeadas
□ 16. pip-audit -r requirements.txt      → sin CVEs en dependencias
□ 17. MODEL=mi-modelo python main.py     → arranca sin errores
□ 18. GET /models/mi-modelo/health       → {"status":"ok","loaded":true}
□ 19. POST /models/mi-modelo/predict (inline) → 200 con predicción correcta
□ 20. POST /models/mi-modelo/train       → 200 con TrainResponse (o 501 si no implementado)

ENTREGA
□ 21. git push origin feature/ML<ticket>-<descripcion>
□ 22. Subir artefactos a S3 (o entregarlos a infra)
□ 23. Abrir PR de feature/... → develop en Bitbucket
□ 24. Pipeline pasa: lint → SonarQube → git-secrets → dependency-scan
□ 25. Revisión de código y merge
```
