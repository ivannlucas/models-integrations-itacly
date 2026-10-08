---
name: plugin-integration
description: Usa este skill para integrar un modelo nuevo en app/plugins/ del repo inference-pan-model, o para renombrar/auditar uno ya integrado. Cubre la nomenclatura obligatoria, el contrato ModelPluginPort, los ficheros obligatorios (incluido mlflow_utils.py, siempre), cómo decidir si train() es real o no aplica, el patrón de reentrenamiento vía MLflow, las trampas de fidelidad que ya han roto plugins en producción, el registro en app/registry.py y los tests. Requiere inbox/aNN/manifest.yaml ya generado (skill manifest-extraction).
---

# Integración de un plugin de modelo

> Este skill recoge lo aprendido en la auditoría de los 25 plugins (octubre 2026). Cada regla
> marcada con ⚠ procede de un bug real que llegó a producción sin que ningún test lo detectara.
> El detalle de cada caso está en `outputs/informe_cambios_modelos.md`.

## Arquitectura del repo (hexagonal)

- `app/domain/ports/model_plugin_port.py`: contrato abstracto. No se toca.
- `app/application/`: casos de uso genéricos (predict, stats, train). No se tocan por modelo.
- `app/infrastructure/`: DI, `router_factory`, `ArtifactStore`. No se tocan por modelo.
- `app/plugins/<nombre>/`: todo lo específico del modelo vive aquí, de forma autocontenida.
- `app/registry.py`: único punto de conexión entre el plugin y el resto del sistema.

## Requisito previo

Debe existir `inbox/aNN/manifest.yaml` (skill `manifest-extraction`). Si no existe, para y
genera el manifest primero. Nunca se scaffoldea directamente sobre el código del equipo de IA
sin esa extracción: ml21 y ml23 se integraron saltándose este paso y ambos llegaron a
producción con `/predict` roto.

## Nomenclatura obligatoria

| Elemento | Formato | Ejemplo |
|---|---|---|
| Carpeta del plugin | `mlNN_<sector>_<desc>` | `ml46_dairy_fouling_clog_detection` |
| `MODEL_ID` (y `model_id` / `prefix` en registry) | La misma cadena con guiones | `ml46-dairy-fouling-clog-detection` |
| `ARTIFACT_FOLDER_NAME` | Igual que la carpeta | `ml46_dairy_fouling_clog_detection` |
| Clase del plugin | CamelCase de la carpeta + `Plugin` | `Ml46DairyFoulingClogDetectionPlugin` |
| Tests | `tests/unit/test_<carpeta>.py` | `tests/unit/test_ml46_dairy_fouling_clog_detection.py` |

- `NN` es el número del caso de uso, sin ceros a la izquierda (`ml7`, no `ml07`). Va siempre
  con prefijo `ml`: nunca `modelo`, `m` ni `a`.
- Sector en inglés: `cereals`, `dairy`, `meat`, `wine`. Para modelos lácteos nuevos usa `dairy`
  (ml10, ml34, ml35, ml46, ml47); ml4 y ml23 usan `lactic` por historia y se dejan así.
- `<desc>` en inglés, en `snake_case`, y describe la tarea (no el algoritmo, salvo que sea lo
  que distingue al modelo).
- `MODEL_ID`, `ARTIFACT_FOLDER_NAME` y `VERSION` van al principio de `constants.py`, nunca
  definidos dentro de `plugin.py`.
- `inbox/aNN/` y `outputs/aNN/` conservan el código `aNN` del caso de uso (`a07`, `a33`): no se
  renombran.

### Renombrar un plugin ya integrado

Cambiar el nombre **es un cambio de contrato**: cambia la ruta de la API
(`/models/<model_id>/...`) y la ruta de los artefactos en S3. Procedimiento:

1. `git mv` de la carpeta del plugin y de sus tests, para conservar el historial. Los ficheros
   no rastreados de dentro viajan con la carpeta.
2. Sustituir los identificadores con un script de reglas ordenadas, de la más específica a la
   más general. Primero lanzarlo en seco para ver el recuento por fichero, y revisar
   `app/`, `tests/` (incluido `conftest.py`: alias, factorías y `TEST_REGISTRY`), los imports
   de otros tests (p. ej. `test_infrastructure.py`), `requirements.txt` y el manifest.
   Cuidado con las subcadenas: `wine_sulphite` está dentro de `ml25_wine_sulphites`.
3. Renombrar la carpeta local `artifacts/<viejo>/` → `artifacts/<nuevo>/` (no se commitea).
4. **Copiar en S3** `artifacts/fixed/<viejo>/` → `artifacts/fixed/<nuevo>/` **antes de
   desplegar**. Es una acción operativa del responsable; el agente no escribe en S3. Si no se
   hace, el modelo deja de cargar en producción.
5. Avisar de que el `model_id` cambia para el front, el servicio de explicabilidad y el ODD
   (skills `front-integration`, `explainability-integration` y `odd-integration`).
6. Actualizar `model_key` y `model_slug` (el slug es el id sin `mlNN-`) en
   `outputs/aNN/datos_*.json`, regenerar las fichas con `docs-generation` y añadir una nota
   `[RENOMBRADO <fecha>]` al principio del manifest y del `verification_report.md`. El texto
   histórico de debajo no se reescribe.
7. Verificar: `flake8`, `from app.registry import REGISTRY`, la suite completa y un arranque
   real (`MODEL=<nuevo_id>`) con `/health` → `loaded: true`.

## Contrato obligatorio: ModelPluginPort

Toda clase `XxxPlugin(ModelPluginPort)` implementa estos 6 métodos:

| Método | Firma | Notas |
|---|---|---|
| `load` | `load()` | Carga los artefactos vía `ArtifactStore` |
| `is_loaded` | `is_loaded() -> bool` | Salud |
| `predict_batch` | `predict_batch(*, data_path, mlflow_run_id="")` | Inferencia sobre CSV/ZIP/directorio |
| `predict_inline` | `predict_inline(*, features, model_key=None, threshold=None, mlflow_run_id="")` | Inferencia de una muestra |
| `stats` | `stats(mlflow_run_id="")` | Devuelve `StatsResponse` |
| `train` | `train(*, data_path, mlflow_run_id)` | Siempre existe. Si no aplica, lanza `TrainingNotSupportedError` (→ 501) |

⚠ **`predict_inline`/`predict_batch` devuelven siempre su DTO Pydantic tipado**
(`PredictInlineResponse(...)` / `PredictBatchResponse(...)`), nunca un `dict`. El caso de uso
genérico hace `type(result).model_validate(...)`, y un `dict` no tiene ese método: el
resultado es un 500 en *todas* las llamadas reales (pasó en ml21).

## Ficheros en app/plugins/<nombre>/

| Fichero | Obligatorio | Contenido |
|---|---|---|
| `__init__.py` | Sí | Vacío |
| `constants.py` | Sí | `MODEL_ID`, `ARTIFACT_FOLDER_NAME`, `VERSION` en cabecera, nombres de artefactos e hiperparámetros de entrenamiento (`TRAIN_*`) |
| `plugin.py` | Sí | Clase con los 6 métodos |
| `predict_dto.py` | Sí | `PredictBatchRequest/Response`, `PredictInlineRequest/Response`, `PredictRequest = Annotated[Union[...], Field(discriminator="mode")]` |
| `model_loader.py` | Sí | `ArtifactStore(ARTIFACT_FOLDER_NAME)` con carga perezosa (ver Artefactos) |
| `mlflow_utils.py` | **Sí, siempre** | Descarga y subida reales (plantilla abajo). En modelos sin entrenamiento, un stub documentado |
| `train_dto.py` | Si `train()` es real | `TrainRequest` (con `mlflow_run_id` **obligatorio**) y `TrainResponse` |
| `training.py` | Si el entrenamiento es largo | Lógica de entrenamiento portada del código original (ml5, ml23) |
| `preprocessing.py` / `postprocessing.py` | Si aplica | Transformación de entrada y salida |

## ¿El modelo es entrenable? Decídelo con el código, no con el texto

⚠ En cinco modelos (ml2, ml4, ml7, ml17, ml23) el manifest o el plugin decían "no se entregó
código de entrenamiento", y sí se había entregado. El reentrenamiento estuvo deshabilitado (501)
sin motivo. Antes de fijar `training.supported`:

1. Busca en `inbox/aNN/codigo/` scripts y módulos de entrenamiento: `find -iname "*train*"`,
   `scripts/train.py`, `src/training/`, `src/main.py train`, flags como
   `training_enabled: true` en `config.yaml` y los hiperparámetros en la configuración.
2. Si existen, comprueba **qué entrenan**. Que haya código de entrenamiento no basta: en ml28 y
   ml33 el código entrena un modelo que el motor desplegado no usa (una comparativa
   experimental o un *benchmark* NEAT). Lo que cuenta es si entrena **el motor que sirve el
   plugin**.
3. `training.supported: false` solo es válido si el motor desplegado no tiene parámetros
   aprendidos: optimizadores exactos (ml31 LP, ml33 MILP) o motores de reglas (ml28). Documenta
   el motivo con evidencia de código o de memoria. Ver `outputs/informe_modelos_sin_reentrenamiento.md`.

## train(): patrón de reentrenamiento

### Reglas de contrato (todas obligatorias)

- ⚠ **`mlflow_run_id` es obligatorio y no vacío**: `TrainRequest.mlflow_run_id: MlflowRunId`
  (tipo común de `app.application.dto.train_dto`, `min_length=1` tras quitar espacios), sin
  valor por defecto, y `train(self, *, data_path, mlflow_run_id)` sin `= ""`. `/train` sin
  `mlflow_run_id`, o con `""`, devuelve 422. Con `str` a secas, `""` pasaba la validación: se
  entrenaba y el modelo se descartaba. Lo exigen los 23 plugins entrenables.
- ⚠ **El modelo reentrenado se guarda SOLO en MLflow** (directorio temporal →
  `upload_artifacts` → `rmtree`). Nunca se escribe en `artifacts/<ARTIFACT_FOLDER_NAME>/` ni en
  S3, y el ajuste fino parte siempre del modelo base, nunca de un reentrenamiento anterior. En
  ml25, un `train()` antiguo generó el artefacto que acabó sirviéndose como "base", con un
  protocolo distinto del del equipo de IA; ml41 sobrescribía el checkpoint base local y
  ml15/16/3/40/9 dejaban copias `user_*` en local.
- ⚠ **Si no se puede guardar en MLflow, `/train` falla**: el `except` de la subida hace
  `raise ModelPersistenceError(...) from exc` (→ 502). Nunca se responde 200 con un aviso: como
  el modelo solo vive en MLflow, el resultado del entrenamiento se habría perdido sin que la
  plataforma lo supiera.
- **No mutar `self._model`**: se entrena sobre una instancia nueva (`copy.deepcopy`, un
  `YOLO(path)` fresco o `load_*()`). Una petición de predict concurrente nunca debe ver un
  modelo a medio entrenar.
- Validar `training.required_columns`. Si falta algo, `raise ValueError(...)` con detalle
  (→ 400).

### Fidelidad al procedimiento original

- Porta **el mismo** optimizador, *loss*, *scheduler*, *sampler*, semillas e hiperparámetros
  que el código del equipo de IA (`constants.TRAIN_*`, citando el fichero de origen). Nunca
  elegidos a criterio del agente: ml8 tenía 6 divergencias (backbone V2 en vez de V1, sin
  balanceo de clases, sin `WeightedRandomSampler`, sin acumulación de gradiente, sin semilla…)
  y modelo10 tenía hiperparámetros inventados.
- Si el original hace una **búsqueda de arquitectura o una comparativa de modelos** (NAS
  neuroevolutiva en ml30, comparativa Naive/XGBoost/LSTM/GRU en ml23), `/train` **no la
  repite**. Haz *refit* solo de la arquitectura ya seleccionada y desplegada, y documéntalo en
  el docstring y en el manifest.
- **Verifica que el refit reproduce las métricas reportadas** con una ejecución real y barata.
  ml17 reproduce la Tabla 7 de la memoria a 4 decimales y ml23 reproduce las métricas del
  artefacto servido a 4 decimales en 48 s. Si el entrenamiento real dura horas, verifica cada
  pieza por separado (scheduler, split, *transforms*) con *smoke tests* de segundos, y reduce
  `epochs` solo con *monkeypatch* en el script de prueba, nunca tocando `constants.py`.
- Si encuentras bugs en el **código de entrenamiento entregado** (ml5: *scheduler* por época,
  fuga de datos por split por imagen, `track_id` no único, `ColorJitter` en validación),
  corrígelos en el plugin **y** en `inbox/aNN/codigo/`, y documéntalos en `known_issues`.

### Respuesta

`TrainResponse` lleva `detail`, las métricas de `training.metrics_returned` (las mismas que
reporta la memoria, para poder compararlas), los conteos (`n_train`/`n_test`),
`training_time_s` y `upload_warning: str | None`. `upload_warning` se mantiene solo por
compatibilidad con la plataforma (`train-task-manager.ts` lo lee) y siempre vale `None`: un
fallo de subida es un `ModelPersistenceError`, no un aviso. Si el plugin no tiene DTO propio,
usa el genérico `app.application.dto.train_dto.TrainResponse`.

```python
class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="Ruta (local o s3://) a <contrato real del dataset>")
    mlflow_run_id: MlflowRunId = Field(..., description="Run de MLflow donde se persiste el modelo "
                                       "reentrenado. Obligatorio: el artefacto base nunca se sobrescribe.")
```

### Si `training.supported: false`

`train()` lanza `TrainingNotSupportedError` con un mensaje que explica el motivo (→ 501). En
`registry.py` **no** se conectan `train_request_type`/`train_response_type`. Referencias:
`ml28_meat_neuroevolutionary_raw_materials_prediction`, `ml31_cereals_residue_optimizer`,
`ml33_cereals_reuse_strategy_optimizer`.

## mlflow_utils.py y uso de mlflow_run_id en predict

Plantilla real, con la misma API que usan todos los plugins
(`app/domain/services/mlflow_tracker.py`):

```python
import logging, os, shutil, tempfile
from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.<nombre>.constants import ARTIFACT_FOLDER_NAME, MODEL_FILENAME

logger = logging.getLogger(__name__)


@require_user_model   # None o error de artefacto al cargar, con run_id → UserModelUnavailableError (→ 422)
def download_user_model_from_mlflow(run_id: str):
    """Return (model, ..., temp_dir). Caller MUST shutil.rmtree(temp_dir) in finally."""
    tmp = tempfile.mkdtemp(prefix="mlflow_<nombre>_")
    try:
        local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
        if not local_path or not os.path.exists(os.path.join(local_path, MODEL_FILENAME)):
            logger.warning("MLflow run_id=%s sin artefacto completo en 'model'", run_id)
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        model = ...  # reconstruir desde los metadatos descargados (nº de clases, feature_cols, scaler…)
        return model, tmp
    except BaseException:   # p. ej. falta un fichero secundario: no dejar el temporal huérfano
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def upload_artifacts_to_mlflow(artifact_dir: str, mlflow_run_id: str, metrics: dict | None = None):
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)
    if metrics:
        tracker.log_metrics(metrics)
        tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})
    tracker.upload_artifacts(artifact_dir, artifact_path="model")
```

⚠ **`predict_inline`, `predict_batch` y `stats` deben usar `mlflow_run_id`**, no descartarlo
con `_ = mlflow_run_id`. En ml2, ml4, ml7, ml17 y ml23 se aceptaba y se ignoraba: el usuario
creía estar usando su modelo y recibía el base. ⚠ Por lo mismo, **si el modelo del run no se
puede cargar, la petición falla** con `UserModelUnavailableError` (→ 422) gracias a
`@require_user_model`: nunca se cae en silencio al modelo base. La plataforma guarda el run en
el modelo del usuario antes de entrenar, así que un entrenamiento fallido deja un run vacío, y
predecir con él debe dar error, no las predicciones del base. El decorador solo convierte en 422
los errores de **artefacto** (`ARTIFACT_LOAD_ERRORS` en `mlflow_tracker.py`: `OSError`,
`ValueError`, `KeyError`, `EOFError`, `UnpicklingError`, `RuntimeError` de `load_state_dict`),
que se han comprobado contra joblib, pickle, torch, numpy, json, Keras y YOLO con ficheros
ausentes, truncados, corruptos o de otra arquitectura. Un `AttributeError`, `TypeError` o
`NameError` es un bug del loader y sale como **500**, para que se reintente y salte la alerta.
Si tu loader necesita otra excepción de artefacto, añádela a esa tupla con su prueba, no captures
`Exception`. Los no entrenables (ml28, ml31,
ml33) sí ignoran el run: el base es su único modelo posible. Patrón:

```python
def _resolve_for_predict(self, mlflow_run_id):
    if not mlflow_run_id:
        return self._model, None
    return download_user_model_from_mlflow(mlflow_run_id)   # (model, temp_dir) o excepción

model, tmp = self._resolve_for_predict(mlflow_run_id)
try:
    ...                           # usa `model`, NUNCA asigna a self._model
finally:
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
```

Si el modelo reentrenado puede tener otro número de clases u otras columnas, la arquitectura se
reconstruye con los metadatos que se descargan, no con las constantes del modelo base. En
`stats()`, con `mlflow_run_id` informado, añade `tracker.get_params()` y `get_metrics()` del run.

## Trampas de fidelidad que ya han roto plugins

| ⚠ | Regla | Qué pasó |
|---|---|---|
| Decodificación de clases | **Nunca** confíes en los nombres de clase embebidos en el checkpoint (`model.names`, `behavior_to_idx`…) sin contrastarlos con las etiquetas reales del dataset. Fija el orden real de entrenamiento en una constante (`TRAINING_*_TO_*`) documentada con la evidencia | ml7 etiquetaba mal la especie en el 61 % de los casos (39 % → 94,4 % tras el fix) y ml5 en 10 de 12 clases (accuracy 0,44 → 0,91). Las métricas de detección y los tests no lo veían |
| Preprocesado | Replica **exactamente** el de inferencia del original: `imgsz`, resize/crop, normalización, orden de canales | ml7 no pasaba `imgsz=512` (usaba 640). A modelo10 le faltaba `CenterCrop` |
| Features derivadas | Nunca rellenes con 0 las features que el original calcula o busca. Si el contrato de entrada es reducido (provincia + fecha), busca la fila real en el dataset de referencia | ml21 rellenaba con 0 ~90 features: predicciones corruptas |
| Columnas derivadas en batch | Replica las derivaciones del predictor original (p. ej. `current_price = target_precio_medio`) | ml23 devolvía 0 predicciones con HTTP 200 |
| Autotest de CUDA | `torch.nn.Conv2d(1,1,1).cuda()(torch.zeros(1,1,4,4).cuda())`: **el módulo también va a CUDA**. Sin `.cuda()` en el módulo, la prueba falla siempre y se cae a CPU | 7 plugins corrían en CPU con GPU disponible |
| Device explícito | Pasa `device=` explícito a YOLO y a cualquier `.predict()` de librería | El detector de modelo10 daba un 500 real |
| Artefacto base | Comprueba con md5 que `artifacts/<ARTIFACT_FOLDER_NAME>/` es **bit a bit** el entregado por el equipo de IA | ml25 servía un modelo generado por un `train()` antiguo |
| `predict_inline` de series temporales | Si para construir la secuencia se repite (`tile`) una sola fila, documéntalo como aproximación en `known_issues` | ml23, ml16, ml40, ml46 |

## Registro en app/registry.py

```python
from app.plugins.<nombre>.plugin import <Nombre>Plugin
from app.plugins.<nombre>.predict_dto import PredictRequest as <Alias>_Request, PredictResponse as <Alias>_Response
from app.plugins.<nombre>.train_dto import TrainRequest as <Alias>_TrainReq, TrainResponse as <Alias>_TrainResp

ModelEntry(
    model_id="<model-id>",
    prefix="/models/<model-id>",
    version="1.0.0",                      # coherente con constants.VERSION
    plugin_class=<Nombre>Plugin,
    predict_request_type=<Alias>_Request,
    predict_response_type=<Alias>_Response,
    train_request_type=<Alias>_TrainReq,     # ambos o ninguno
    train_response_type=<Alias>_TrainResp,
    extra_predict_exceptions=(...,),         # excepciones de dominio → 422
)
```

⚠ `train_request_type` y `train_response_type` van **los dos o ninguno**. Si falta el de
respuesta, `/train` cae al DTO genérico y descarta en silencio las métricas reales (pasó en
ml21, m47 y ml5). No se toca `main.py`: `router_factory.py` genera `/health`, `/stats`,
`/predict` y `/train` a partir de esta entrada.

⚠ Tras cada merge que toque `registry.py`, ejecuta
`python -c "from app.registry import REGISTRY"`. Los tests unitarios **no importan**
`app/registry.py`. El merge de ml15 dejó un `),` y un `]` sobrantes, y `main` no arrancaba
aunque la suite pasaba en verde.

## Códigos HTTP (router_factory)

| Endpoint | Situación | Código |
|---|---|---|
| todos | Body inválido según el DTO (p. ej. falta `mlflow_run_id` en `/train`) | 422 |
| `/predict` | Excepción listada en `extra_predict_exceptions` | 422 |
| `/predict`, `/stats` | `UserModelUnavailableError` (el run pedido no tiene un modelo cargable: sin artefacto, o error de `ARTIFACT_LOAD_ERRORS` al cargarlo) | 422 |
| `/predict` | Cualquier otra excepción | 500 |
| `/train` | `TrainingNotSupportedError` | 501 |
| `/train` | `ValueError` o `FileNotFoundError` (datos inválidos o ausentes) | 400 |
| `/train` | `ModelPersistenceError` (no se pudo guardar el modelo en MLflow) | 502 |
| `/train` | Cualquier otra excepción | 500 |
| `/health` | Modelo sin cargar | 503 |

Las excepciones de dominio nuevas (una restricción de negocio, un historial insuficiente) van
en `app/domain/services/exceptions.py` y se listan en `extra_predict_exceptions`. Un plugin sin
artefactos disponibles debe arrancar con `loaded: false` y un error explícito, nunca romper.
ml41 es el ejemplo correcto.

## Artefactos

| Origen | Ubicación | Quién escribe | Cuándo se usa |
|---|---|---|---|
| Artefacto base (equipo de IA) | `s3://<STORAGE_BUCKET>/artifacts/fixed/<ARTIFACT_FOLDER_NAME>/`, con copia local en `artifacts/<ARTIFACT_FOLDER_NAME>/` | Solo el equipo de IA o el responsable. **Nunca** `train()` ni el agente | Peticiones sin `mlflow_run_id` |
| Modelo del usuario | Run de MLflow, `artifact_path="model"` (u otro documentado) | `train()` | Peticiones con ese `mlflow_run_id` |

- En `model_loader.py` usa `ArtifactStore(ARTIFACT_FOLDER_NAME).path(fichero)` **por
  fichero**. Es perezoso: solo descarga de S3 si el fichero no está en local y `STORAGE_BUCKET`
  está definido. ⚠ No uses `download_all_if_needed()`: falla sin `STORAGE_BUCKET` aunque los
  artefactos ya estén en local (pasó en ml23 y en otros 5 plugins).
- Los modelos descargados de MLflow viven en un `tempfile.mkdtemp()` que se borra en `finally`
  tras cada petición.
- `artifacts/` está en `.gitignore`. Los artefactos nunca se commitean.

## Tests

`tests/unit/test_<carpeta>.py` más su entrada en `tests/conftest.py`: alias de DTOs, factorías
en `FAKE_FACTORIES` y, si es entrenable, en `TRAIN_FACTORIES` y en el `ModelEntry` de
`TEST_REGISTRY` con los dos tipos de train. Mínimo: health, stats, predict inline/batch,
`test_train_returns_200_with_metrics` y `test_train_without_mlflow_run_id_returns_422` (o
`test_train_returns_501` si no es entrenable).

⚠ **Estos tests usan `FakePlugin`, que devuelve el DTO correcto sin ejecutar el código del
plugin.** Validan el *wiring*, no el plugin. Ningún bug de la tabla de trampas los hizo fallar.
Por eso, antes de dar el plugin por integrado:

- Arranca el servidor real (`MODEL=<model-id> python main.py`) y llama a `/health`, `/predict`
  (inline y batch) y `/train` sin `mlflow_run_id` (→ 422) con datos reales del entregable.
  Para el proceso al terminar, incluidos los hijos `multiprocessing.spawn` del `--reload`.
- `BaseMLflowTracker` **no tiene timeout de conexión**. En los tests que ejecuten el `train()`
  real, haz `patch(...BaseMLflowTracker)`, o la suite se bloquea minutos contra un MLflow
  inalcanzable. Para probar `train()` de verdad fuera del clúster, sustitúyelo por un stub.
- Hay dos tests de regresión que se aplican **solos** a todo plugin nuevo, sin registrarlo en
  ninguna lista (`tests/unit/test_user_model_download_hygiene.py`):
  - ningún `predict*`/`stats` asigna atributos de modelo en `self`, ni tampoco los métodos de
    la misma clase a los que llama con `self.X(...)` (se siguen de forma transitiva). **No ve**
    funciones de módulo que reciban el plugin como argumento, escrituras con `setattr`/`__dict__`
    ni métodos heredados de otro fichero. La regla de fondo es otra: los helpers que resuelven
    el modelo del usuario lo **devuelven**, nunca lo guardan;
  - todo `download_*_from_mlflow` borra su temporal cuando el run no trae un modelo cargable
    y responde con `UserModelUnavailableError`.

  Si un plugin nuevo los rompe, no los relajes: corrige el plugin. El aislamiento entre
  peticiones concurrentes se prueba además con `test_user_model_isolation.py`; añade ahí el
  plugin si es entrenable.
- Los tests que entrenan de verdad y comparan dos ejecuciones deben fijar el dispositivo a CPU
  (`monkeypatch` de la función de dispositivo): en GPU, cuDNN no es determinista y el test
  pasaría o fallaría según la máquina.

La correctitud numérica se valida después, en el skill `verification`, contra el golden dataset.

## Checklist de salida de este skill

```
[ ] Nomenclatura: carpeta mlNN_<sector>_<desc>, MODEL_ID con guiones,
    ARTIFACT_FOLDER_NAME = carpeta, constantes en la cabecera de constants.py
[ ] Todos los ficheros obligatorios, incluido mlflow_utils.py con descarga/subida reales
[ ] predict_inline/predict_batch devuelven el DTO tipado (nunca dict)
[ ] mlflow_run_id usado de verdad en predict/stats, sin mutar self._model, con rmtree en finally
[ ] training.supported decidido con el código real (qué entrena y si es el motor desplegado)
[ ] Si es entrenable: mlflow_run_id obligatorio, guardado solo en MLflow, procedimiento
    original portado fielmente y métricas reportadas reproducidas con una ejecución real
[ ] Si no lo es: TrainingNotSupportedError explicado y sin train types en registry
[ ] Trampas de fidelidad revisadas: orden de clases frente a etiquetas reales, preprocesado,
    features derivadas, CUDA, device, md5 del artefacto base
[ ] registry.py: ModelEntry con train types (ambos o ninguno); `from app.registry import REGISTRY` OK
[ ] conftest.py (FAKE_FACTORIES, TRAIN_FACTORIES, TEST_REGISTRY) y tests/unit/test_<carpeta>.py
[ ] Suite completa en verde y smoke test contra el servidor real con datos reales
[ ] Artefactos en artifacts/<ARTIFACT_FOLDER_NAME>/ (local) y en S3 con el mismo nombre
[ ] Listo para pasar al skill "verification"
```
