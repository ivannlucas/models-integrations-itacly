---
name: plugin-integration
description: Usa este skill para integrar un modelo nuevo en app/plugins/ del repo inference-pan-model. Cubre el contrato ModelPluginPort, los ficheros obligatorios de cada plugin (incluido mlflow_utils.py, obligatorio siempre), el registro en app/registry.py, excepciones de dominio y artefactos. Requiere inbox/<model_id>/manifest.yaml ya generado (skill manifest-extraction).
---

# Integración de un plugin de modelo

## Arquitectura del repo (hexagonal)

- `app/domain/ports/model_plugin_port.py` — contrato abstracto, no se toca.
- `app/application/` — casos de uso genéricos (predict, stats, train), no se tocan por modelo.
- `app/infrastructure/` — DI, `router_factory`, artifact store, no se tocan por modelo.
- `app/plugins/<nombre>/` — todo lo específico del modelo vive aquí, autocontenido.
- `app/registry.py` — único punto de conexión entre tu plugin y el resto del sistema.

## Requisito previo

Debe existir `inbox/<model_id>/manifest.yaml` (skill `manifest-extraction`). Si no existe, para
y genera el manifest primero — no se scaffoldea directamente sobre el código crudo del equipo
de IA sin haber pasado por esa extracción.

## Contrato obligatorio: ModelPluginPort

Toda clase `XxxPlugin(ModelPluginPort)` implementa estos 5 métodos:

| Método | Firma | Notas |
|---|---|---|
| `load` | `load()` | Carga artefactos desde disco/S3 vía `ArtifactStore` |
| `is_loaded` | `is_loaded() -> bool` | Salud |
| `predict_batch` | `predict_batch(*, data_path, mlflow_run_id="")` | Inferencia sobre CSV/lote |
| `predict_inline` | `predict_inline(*, features, model_key=None, threshold=None, mlflow_run_id="")` | Inferencia de una muestra |
| `stats` | `stats(mlflow_run_id="")` | Devuelve `StatsResponse` |
| `train` | `train(*, data_path, mlflow_run_id)` | Obligatorio que exista; si no aplica, lanza `TrainingNotSupportedError` → 501 |

El parámetro `mlflow_run_id` es lo que permite servir un modelo reentrenado por un usuario
concreto en lugar del artefacto fijo — ver `mlflow_utils.py` más abajo.

## Ficheros obligatorios en app/plugins/<nombre>/

| Fichero | Obligatorio | Contenido |
|---|---|---|
| `__init__.py` | Sí | vacío |
| `plugin.py` | Sí | Clase con los 5 métodos |
| `predict_dto.py` | Sí | `PredictBatchRequest/Response`, `PredictInlineRequest/Response`, `PredictRequest = Annotated[Union[...], Field(discriminator="mode")]` |
| `constants.py` | Sí | Nombres de artefactos + `ARTIFACT_FOLDER_NAME` |
| `model_loader.py` | Sí | Usa `ArtifactStore(ARTIFACT_FOLDER_NAME)` |
| `mlflow_utils.py` | **Sí, siempre** | Ver plantilla abajo — no es opcional aunque el modelo no soporte reentrenamiento hoy |
| `preprocessing.py` | Si aplica | Transformación de entrada |
| `postprocessing.py` | Si aplica | Transformación de salida |
| `train_dto.py` | Si `train()` hace algo real | `TrainRequest/TrainResponse` |
| otros helpers | Opcional | lógica auxiliar específica del dominio |

### mlflow_utils.py — obligatorio en todos los plugins

Plantilla base (adaptar tipo de modelo, arquitectura y ficheros de artefacto al plugin concreto):

```python
"""MLflow helper for <Nombre> — download user-trained model from MLflow."""
from __future__ import annotations

import logging

from app.domain.services.mlflow_tracker import download_mlflow_artifacts
from app.plugins.<nombre>.constants import MODEL_FILENAME  # + lo que aplique (class_names, scalers...)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Download a user-trained model from MLflow.

    Returns (model, extra_artifacts, temp_dir).
    Caller MUST shutil.rmtree(temp_dir) after inference — usar try/finally.
    """
    result = download_mlflow_artifacts(run_id, artifact_path="<carpeta_artifact_en_mlflow>", prefix="mlflow_<nombre>_")
    if result is None:
        return None
    tmp, local_path = result

    # ... cargar el modelo desde local_path según su framework (torch/sklearn/etc.)

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, extra_artifacts, tmp
```

**Checklist de este fichero**:

- [ ] El caller (`plugin.py`) hace `try/finally: shutil.rmtree(tmp)` — si no, hay fuga de disco
      en cada predicción que use un modelo de usuario.
- [ ] Si el modelo tiene número de clases/salida variable según el reentrenamiento
      (clasificación), la arquitectura se reconstruye dinámicamente a partir de los metadatos
      descargados, no se asume fija.
- [ ] `predict_inline`/`predict_batch`/`stats` reciben `mlflow_run_id` y, si viene informado,
      usan `download_user_model_from_mlflow` en vez del artefacto fijo de `model_loader.py`.

## train() — patrón de reentrenamiento (fine-tuning)

Consultar `manifest.training` (skill `manifest-extraction`) antes de escribir nada aquí — define
si el modelo soporta reentrenamiento y con qué columnas/hiperparámetros.

### Si `training.supported: true`

Seguir el patrón real ya usado en `ml35_dairy_ann_cleaning_cost/plugin.py` y
`ml25_wine_sulphites/plugin.py` (ambos con `train()` funcional):

1. Cargar `data_path` (CSV) y validar que trae `training.required_columns` del manifest. Si
   faltan columnas, `raise ValueError` con el detalle explícito — no se entrena con datos
   incompletos.
2. **Clonar los pesos del modelo cargado en una instancia nueva antes de entrenar** — nunca mutar
   `self._model` in-place hasta tener el resultado final. Así una petición de `predict`
   concurrente nunca ve un modelo a medio entrenar.
3. Fine-tune con los `training.hyperparams` del manifest (mismo optimizador/loss que usó el
   equipo de IA en el entrenamiento original — nunca elegidos a criterio del agente).
4. Calcular las métricas de `training.metrics_returned` sobre los datos de entrenamiento y
   devolverlas en `TrainResponse` — deben ser las mismas métricas que reporta la memoria, para
   que sean comparables.
5. Guardar el artefacto reentrenado **localmente siempre** (vía `_store.local_dir` /
   `ArtifactStore`), y si viene `mlflow_run_id`, subirlo también a MLflow reutilizando
   `BaseMLflowTracker` (`app/domain/services/mlflow_tracker.py` — no se reimplementa):

   ```python
   if mlflow_run_id:
       tracker = BaseMLflowTracker(mlflow_run_id)
       tracker.log_params({...})       # hiperparámetros usados en este run
       tracker.log_metrics({...})      # mae, r2, n_samples, etc.
       tracker.upload_artifacts(tmp_dir, artifact_path="model")  # tmp_dir con try/finally rmtree
   ```

### train_dto.py

```python
from pydantic import BaseModel, ConfigDict, Field

class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="Path to CSV with <features> + <target_column>")
    mlflow_run_id: str = ""

class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    # + un campo por cada métrica en training.metrics_returned del manifest (mae, r2, n_samples...)
```

### Si `training.supported: false`

`train()` lanza `TrainingNotSupportedError` explícitamente (→ 501 automático vía
`router_factory`) — nunca se deja sin implementar ni se devuelve un stub silencioso. Ver
`ml2_fungal_cnn_disease_detection/plugin.py`, `ml5_meat_cow_behaviour/plugin.py` o
`ml7_cereals_grain_pest_detection/plugin.py` como referencia de este caso.

## Reference por tipo de modelo (copiar el plugin ya integrado más parecido)

| Tipo de modelo | Referencia |
|---|---|
| Imagen (CNN, PyTorch) | plugin de detección/clasificación de imagen ya integrado más cercano al sector |
| Tabular regresión (sklearn/PyTorch) | plugin tabular ya integrado más cercano |
| Tabular + optimización prescriptiva (ANN + GA) | a35, una vez integrado — usar como caso piloto |

## Registro en app/registry.py

```python
from app.plugins.<nombre>.plugin import <Nombre>Plugin
from app.plugins.<nombre>.predict_dto import PredictRequest, PredictResponse
# + train_dto si aplica

ModelEntry(
    model_id="<model_id>",
    prefix="/models/<model_id>",
    version="1.0.0",
    plugin_class=<Nombre>Plugin,
    predict_request_type=PredictRequest,
    predict_response_type=PredictResponse,
    train_request_type=TrainRequest,        # o None
    train_response_type=TrainResponse,      # o None
    extra_predict_exceptions=(...,),        # excepciones de dominio -> 422
)
```

No se toca `main.py` — `router_factory.py` genera `/health`, `/stats`, `/predict`, `/train`
automáticamente a partir de esta entrada.

## Excepciones de dominio

Si el modelo necesita códigos HTTP específicos (p. ej. una restricción de negocio violada, como
PU < 13 en a35), añadir la excepción en `app/domain/services/exceptions.py` y referenciarla en
`extra_predict_exceptions` — se traduce automáticamente a 422. Cualquier excepción no listada
cae en 500.

## Artefactos

Local: `artifacts/<ARTIFACT_FOLDER_NAME>/`. 

En producción, los artefactos se pueden encontrar en dos sitios: s3 o mlflow. En mlflow se deben almacenar los artefactos creados por el usuario (como resultado del entrenamiento de los modelos), a su vez, cuando se recibe un mlflow_run_id en el endpoint de solicitud de prediccion o estaditicas, los artefactos deben buscarse en mlflow. Por otro lado, en s3 se almacenan los artefactos estandar, aquellos que fueron entrenados por un equipo especializado de IA. Estos nunca deben de ser sobreescritos por el usuario. Cuando el usuario no envia mlflow_run_id, se deben usar estos artefactos para realizar las predicciones y devolver su estadisticas asociadas. En app\domain\services\mlflow_tracker.py se encuentra la logica de integracion para con mlflow. Cuando el usuario solicita un artefacto de s3 y lo utiliza para realizar una prediccion, posteriormente, el artefacto debe ser eliminado del contenedor para liberar espacio de almacenamiento. Si el usuario vuelve a solicitarlo, se debe descargar nuevamente.
Producción s3:
`s3://<STORAGE_BUCKET>/artifacts/fixed/<ARTIFACT_FOLDER_NAME>/` — `ArtifactStore` descarga
automáticamente si faltan o difieren y `STORAGE_BUCKET` está seteado.

## Tests

`tests/unit/test_<nombre>.py`, siguiendo el patrón de los existentes (mockeo del plugin/artefactos
vía `FakePlugin`, fixtures en `conftest.py`). Estos tests validan wiring, no correctitud — la
correctitud se valida en el skill `verification` contra el golden dataset real del manifest.

## Checklist de salida de este skill

```
[ ] app/plugins/<nombre>/ con todos los ficheros de la tabla de arriba
[ ] mlflow_utils.py presente, con try/finally rmtree documentado en plugin.py
[ ] train() implementado según manifest.training (fine-tuning real) o TrainingNotSupportedError
    explícito si training.supported=false — train_dto.py acorde en ambos casos
[ ] app/registry.py: ModelEntry añadido
[ ] Excepciones de dominio añadidas si aplica
[ ] tests/unit/test_<nombre>.py con FakePlugin
[ ] Artefactos colocados en artifacts/<ARTIFACT_FOLDER_NAME>/ local
[ ] Listo para pasar al skill "verification"
```
