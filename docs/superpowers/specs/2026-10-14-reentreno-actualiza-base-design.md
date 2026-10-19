# Reentreno que actualiza el modelo base (ml13, 14, 15, 16, 17, 18, 19, 21, 23, 24)

**Fecha:** 14 de octubre de 2026 · **Rama:** `feature/retrain-updates-base-model` (desde `main` 1c2ebe6)
**Estado:** implementado (commit e34bb9d). Revisado el 15 de octubre de 2026: decisiones de operación
y baja en otros servicios en el §9. Ampliado a ml19 y ml24 el 19 de octubre de 2026 (§10)

---

## 1. Objetivo

Los modelos de series temporales 13, 14, 15, 16, 17, 18, 21 y 23 se consumen por API desde otra
plataforma. Allí se desarrollan los scrapers, que llaman a nuestro orquestador para predecir o
reentrenar según la frecuencia de cada modelo (semanal en 13, 16 y 17; mensual en general; anual
para el reentreno completo de 15, 18 y 21). Se quiere que **cada `/train` correcto de estos modelos
deje el modelo reentrenado como modelo base**, el que se usa sin `mlflow_run_id`, para que nadie
tenga que guardar un `run_id` para usarlo.

Ese calendario sale de la degradación esperada de cada modelo. Se conoce la frecuencia con la que
se actualizan sus fuentes de datos y su ventana de predicción, así que se puede estimar cada cuánto
dejan de ser representativas sus predicciones y fijar ahí el reentreno. `/train` solo lo lanza un
entorno controlado: un script recopila la última versión de los datasets en esa secuencia fija y
llama a `/train` con ellos. Los usuarios de la plataforma no reentrenan estos modelos: solo
predicen, siempre con la versión base vigente. Ver §9.

Toma de `docs/propuesta_actualizacion_continua_series_temporales.md` (§6.3 opción B y §7.1) el
principio *"la base nunca se sobrescribe en el sitio: se publica una versión nueva y se mueve un
puntero"*, versionado y reversible.

### Decisiones tomadas

| # | Decisión | Elegido |
|---|---|---|
| D1 | Cuándo se promueve | **Siempre, automáticamente**: todo `/train` correcto pasa a ser la base. El rollback es la red de seguridad |
| D2 | Dónde vive la versión vigente | **Puntero en S3** (`current.json` + `versions/<vN>/`), no MLflow Model Registry |
| D3 | Réplicas | **Un solo pod**: recarga en proceso tras promover, sin sondeo periódico del puntero |
| D4 | `mlflow_run_id` en `/train` | **Opcional** en estos modelos. Si viene, el bundle también se sube al run (trazabilidad) |
| D5 | Datos de contexto | ml21 publica el CSV de entrenamiento como panel; ml15 acepta CSV de referencia opcionales; ml14 lleva sus scalers en el bundle |

### Fuera de alcance

- Scrapers, orquestador, monitorización del error y champion/challenger (resto de la propuesta).
- Explicabilidad (SHAP) y detección de drift (ODD): dejan de aplicar a estos modelos y se dan
  de baja en esos servicios (§9).
- Recarga entre réplicas, listado de versiones (`GET /versions`) y limpieza de versiones antiguas.
- Integración con el entrenamiento asíncrono (`feature/async-train`, no está en `main`). Ver §8.
- Distinguir reentreno "mínimo" y "completo": cada modelo tiene un único `train()`, el portado del
  equipo de IA. En ml23 el completo implicaría repetir la selección de arquitectura, que `train()`
  no hace.
- El resto de modelos: su `/train` no cambia (el modelo nuevo solo vive en su run de MLflow).

---

## 2. Disposición de los artefactos

### 2.1 En S3

```
s3://<STORAGE_BUCKET>/artifacts/fixed/<ARTIFACT_FOLDER_NAME>/
    <ficheros actuales>            ← "v0": la base entregada, nunca se mueve ni se borra
    current.json                   ← {"version": "v3"}; si no existe, la base es v0
    versions/
        v1/  <bundle completo> + version.json
        v2/  <bundle completo> + version.json
        v3/  ...
```

- Cada `versions/<vN>/` es **autocontenida**: tiene todos los ficheros que el loader necesita
  (los heredados de la versión padre + los del bundle reentrenado). Así un rollback o un arranque
  no tiene que componer versiones.
- `version.json`:

  ```json
  {
    "version": "v3",
    "parent": "v2",
    "created_at": "2026-10-14T10:00:00+00:00",
    "mlflow_run_id": "abc123" | null,
    "data_path": "s3://xai/data/processed/.../train.csv",
    "metrics": { ... métricas de test que devolvió train() ... }
  }
  ```

  `v0` no tiene `version.json`: se describe como `{"version": "v0", "parent": null}`.
- Nada se borra nunca. El número de una versión nueva es **el máximo existente en `versions/` + 1**,
  de modo que una carpeta que quedó a medias por un fallo nunca se reutiliza.

### 2.2 En local

- Con `STORAGE_BUCKET`: caché en `artifacts/<carpeta>/versions/<vN>/` para vN ≥ 1 y en
  `artifacts/<carpeta>/` para v0 (como hoy). El puntero que manda es el de S3.
- Sin `STORAGE_BUCKET` (desarrollo y tests): el mismo esquema, con `current.json` y `versions/`
  dentro de `artifacts/<carpeta>/`.
- Como cada versión tiene su carpeta, el criterio actual de "descargar solo si falta o cambia de
  tamaño" nunca confunde una versión con otra.

---

## 3. Componentes

### 3.1 `ArtifactStore` con versiones (`app/infrastructure/artifact_store.py`)

Cambios, manteniendo la API que usan hoy los `model_loader.py` (`path()`, `local_dir`,
`download_all_if_needed()`):

- **Versión activa por carpeta, compartida por proceso.** Un registro a nivel de módulo
  `{carpeta: versión}` que comparten todas las instancias de `ArtifactStore` de esa carpeta (ml15,
  por ejemplo, tiene más de un módulo que lee de su store). `path()` y `local_dir` resuelven
  contra la carpeta de la versión activa.
- `refresh()`: lee `current.json` (S3 o local) y fija la versión activa; `v0` si no existe. Se
  llama antes de cada `load()` del plugin (ver §3.3), así que el arranque usa la versión vigente.
- `publish_version(bundle_dir, info) -> VersionInfo`:
  1. asegura en local la versión activa completa;
  2. compone en una carpeta temporal: ficheros de la versión activa (sin `current.json`,
     `versions/` ni `version.json`) y, encima, los del bundle;
  3. sube todo a `versions/<vN+1>/` y luego `version.json`;
  4. **al final** escribe `current.json` y fija la versión activa.

  Si falla cualquier paso anterior al 4, la base no cambia.
- `set_active_version(version)`: reescribe `current.json` y la versión activa. Lo usan el rollback y
  la restauración tras una recarga fallida. Valida que la versión exista (o sea `v0`).
- `version_info(version=None) -> VersionInfo`: lee el `version.json` (de la activa por defecto).

Las versiones se tratan solo con `current.json`. Para los demás modelos, que nunca publican, no
existe y todo sigue exactamente igual que hoy.

### 3.2 Helper común de persistencia (`app/infrastructure/base_model_publisher.py`)

Un único helper que usan los 8 `train()`. Vive en infraestructura porque usa `ArtifactStore`:

```python
def persist_retrained_bundle(
    bundle_dir: str | Path, *, artifact_folder: str, data_path: str, metrics: dict,
    mlflow_run_id: str = "", upload_to_mlflow: Callable[[], None] | None = None,
) -> PersistResult:   # base_version: dict (VersionInfo), upload_warning: str | None
```

1. Si hay `mlflow_run_id`, llama a `upload_to_mlflow`: la subida propia de cada plugin
   (`log_params`, `log_metrics` y `upload_artifacts(bundle_dir, "model")`). Si falla, **no
   aborta**: `upload_warning` con el error. Ahora la base es la persistencia que importa; el run es
   trazabilidad.
2. `ArtifactStore(artifact_folder).publish_version(bundle_dir, info)`. Si falla →
   `ModelPersistenceError` (502), la base no cambia.

Cada `train()` sigue construyendo su bundle con los nombres de fichero canónicos de su base (ya lo
hacen para el run de MLflow) y llama a este helper en lugar de su bloque de subida actual. La
respuesta de `train()` añade `base_version`.

### 3.3 Recarga y lock en el contenedor (`app/infrastructure/http/dependencies/container.py`)

- Los 8 plugins declaran el atributo de clase `BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME`. El
  contenedor promueve solo si el plugin lo tiene; el resto de modelos no cambia.
- **Lock por modelo** (`threading.Lock`) alrededor de `train` + recarga y de `rollback` + recarga.
  Un segundo `/train` simultáneo espera al primero y hereda de la versión que este publicó.
- `reload()`: `store.refresh()`, crea `type(plugin)()`, llama a `load()` y, si va bien, sustituye
  la referencia del plugin en el contenedor y en sus casos de uso y servicio. Las peticiones en curso
  terminan con la instancia vieja, que no se toca.
- **Si la recarga falla**: `set_active_version(<versión anterior>)`, se sigue sirviendo la
  instancia vieja y se devuelve 500 con el error de carga. La versión nueva queda en S3 sin
  apuntar.
- Los contadores de `runtime_stats` se reinician con la instancia nueva (aceptado).

Esto no contradice el test de `main` "ningún `train()` recarga el modelo servido": la recarga la
hace el contenedor, fuera de `train()`, y con una instancia nueva.

### 3.4 Endpoints (`app/infrastructure/http/router_factory.py`)

- `POST /models/<id>/train`: mismo contrato. Respuesta + `base_version` (`VersionInfo`).
- `POST /models/<id>/rollback` (el router lo monta en todos los modelos; los que no tienen
  `BASE_ARTIFACT_FOLDER` responden 501). Cuerpo opcional
  `{"version": "v2"}`:
  - sin cuerpo: vuelve al `parent` de la versión vigente;
  - 409 si la vigente es `v0` (no hay padre);
  - 404 si la versión pedida no existe;
  - recarga como tras `/train` (mismo lock y misma restauración si falla);
  - respuesta: `{"base_version": VersionInfo, "previous_base_version": VersionInfo}`.
- `GET /models/<id>/stats`: `StatsResponse` gana un campo opcional `base_version`, que rellena el
  caso de uso para los plugins con `BASE_ARTIFACT_FOLDER` (`None` en el resto).

### 3.5 DTOs

- `TrainRequest` de los 8: `mlflow_run_id: str = ""` (sin `MlflowRunId`/`min_length=1`).
- `TrainResponse` de los 8: `base_version: VersionInfo`. `upload_warning` ya existe en todos.
- `VersionInfo` (común, `app/application/dto/`): `version`, `parent`, `created_at`,
  `mlflow_run_id`, `data_path`, `metrics`.
- ml15 `TrainRequest`: `reference_prices_path: str = ""` y `reference_financial_path: str = ""`.

---

## 4. Cambios por modelo

| Modelo | Bundle reentrenado | Cambio |
|---|---|---|
| ml13 | `ml_model.pkl`, `scaler.pkl`, `feature_schema.json`, `model_config.json` | Usar el helper. Quitar `MlflowRunId` |
| ml14 | `gru_model.pt` + `scalers.json` | Helper. **Loader de la base**: si la versión trae `scalers.json`, usarlo (como `load_user_bundle`); si no, recalcular desde `final_dataset_for_modeling.csv` como hoy (v0) |
| ml15 | `horizon_6_ridge.pkl` | Helper. Quitar el `ModelPersistenceError("Sin run de MLflow…")`. Si vienen `reference_prices_path`/`reference_financial_path`, validarlos con `history.load_monthly_csv` y las columnas de los actuales **antes de entrenar** (400 si no cuadran) e incluirlos en el bundle como `global_phytosanitary_prices_b2020.csv` / `financial_proxies_monthly.csv`. Si no vienen se heredan y `upload_warning` avisa de que los históricos exógenos no se han renovado |
| ml16 | 2 modelos, 2 scalers, 2 bagging, `train_config.json` | Helper. Quitar el error "Sin run de MLflow" |
| ml17 | `ridge_official_v1_4.pkl` | Helper |
| ml18 | `model.joblib`, `x_scaler.json`, `y_scaler.json` (+ `best_config.json`, `metrics.json`) | Helper |
| ml21 | 6 `datagia_best_h*_{reg,clf}.joblib` + `model_metadata.json` | Helper. Validar **antes de entrenar** que el CSV contiene todas las columnas del panel vigente (400 si no) y añadirlo al bundle como `dataset_entrenamiento_final.csv`, para que `predict_inline` encuentre los meses nuevos |
| ml23 | `gru_model.pt`, `rnn_scaler.npz`, `manifest.json` | Helper. Quitar el error "Sin run de MLflow" |

Predecir con un `mlflow_run_id` antiguo sigue funcionando igual en los 8 (`mlflow_utils.py` no cambia).

---

## 5. Errores

| Fallo | HTTP | La base |
|---|---|---|
| CSV inválido / columnas que faltan (incluidas las validaciones de ml15 y ml21) | 400 | no cambia |
| Error al entrenar | 500 | no cambia |
| Falla la subida al run de MLflow (con run) | 200 + `upload_warning` | **cambia** |
| Falla la subida a `versions/<vN>/` o la escritura de `current.json` | 502 `ModelPersistenceError` | no cambia |
| La versión nueva no carga | 500 | vuelve a la anterior; la instancia vieja sigue sirviendo |
| Rollback sin padre / a versión inexistente | 409 / 404 | no cambia |

Al arrancar, si S3 falla al leer `current.json` o descargar, el modelo no carga, igual que hoy
(`load_errors`, 503 en sus endpoints).

---

## 6. Reglas del repo

- `CLAUDE.md` y `.claude/skills/plugin-integration/SKILL.md`: la regla pasa a ser *"la base nunca se
  sobrescribe en el sitio. En los modelos con `BASE_ARTIFACT_FOLDER` (13, 14, 15, 16, 17, 18, 19, 21, 23 y 24)
  cada `/train` publica una versión nueva en `versions/<vN>/` y mueve `current.json`. En el resto,
  el modelo reentrenado vive solo en su run de MLflow"*.
- README: endpoints `/rollback` y el campo `base_version`.

---

## 7. Tests (unitarios, sin S3 ni MLflow reales)

**`ArtifactStore`**, con un cliente S3 falso en memoria y en modo local (`tmp_path`):
- sin `current.json` → v0 y rutas de hoy;
- `publish_version` → v1 con los ficheros heredados + los del bundle, y luego v2 con `parent=v1`;
- un fallo en mitad de la subida no mueve `current.json`, y la siguiente publicación salta el número;
- `set_active_version` y lectura de `version.json`;
- dos stores de la misma carpeta ven la misma versión activa;
- caché local por versión.

**Contenedor y router:**
- `/train` correcto → `base_version` nueva, el plugin del contenedor es otra instancia y `predict`
  sin run usa el modelo nuevo;
- la recarga falla → `current.json` restaurado, instancia vieja sirviendo, 500;
- `/rollback` sin cuerpo, con versión, 409 en v0 y 404 en versión inexistente;
- `/stats` devuelve `base_version`;
- un plugin sin `BASE_ARTIFACT_FOLDER` no promueve y no tiene `/rollback`.

**Por modelo (los 8):** `/train` con datos pequeños sin `mlflow_run_id` deja la base en v1; con
run, también sube al run; un fallo de MLflow da `upload_warning` y promueve igualmente. ml21
actualiza el panel y rechaza un CSV sin las columnas del panel; ml15 incluye los CSV de referencia
cuando vienen; ml14 carga una versión con `scalers.json`.

Se adaptan los tests actuales de estos 8 modelos que comprueban "run obligatorio" o "la base no se
toca". Los de los demás modelos no cambian. Se mantiene en verde el test de `main` "ningún `train()`
recarga el modelo servido".

**Recarga real por modelo** (`tests/unit/test_base_reload_per_model.py`, ml14 en
`test_ml14_training.py`): para ml13, 15, 16, 17, 18, 21 y 23, un `ModelContainer` real arranca en
v0 (copia de `artifacts/<carpeta>/` en un directorio temporal; ml13, que solo está en S3, con una v0
sintética en el mismo formato), `/train` pasa por el contenedor y la instancia nueva debe leer
todos sus ficheros de `versions/v1/` (los heredados de v0 + el bundle) y predecir con ellos. ml15
comprueba además que sirve los CSV de referencia heredados, y ml21 que su panel incluye el mes
nuevo. Los modelos cuyos artefactos no están en local se saltan.

**Sin probar:** S3 y MLflow reales y el servicio arrancado en INT (ver §9.3).

---

## 8. Riesgos y notas

- **Promoción sin condición (D1).** Un `/train` con datos malos pero válidos deja una base peor. Se
  mitiga con el rollback y con las métricas en `version.json` y `/stats`. Si se quiere una condición
  más adelante, encaja antes del paso 2 del helper (§3.2).
- **`/train` síncrono.** La rejilla de ml21 y las GRU de 14, 18 y 23 tardan minutos. El límite sigue
  siendo el timeout de 22 min del orquestador.
- **Entrenamiento asíncrono.** Si se fusiona `feature/async-train`, `train()` corre en un proceso
  hijo: el helper publicaría la versión desde allí y el proceso principal tendría que hacer
  `reload()` (o `refresh()`) al ver el job terminado. El lock tendría que cubrir también ese caso.
- **Más de una réplica.** Fuera de alcance (D3). Si cambia, añadir la comprobación periódica de
  `current.json`.
- **ODD y SHAP** se calcularían sobre la v0 (el servicio de explicabilidad descarga su propia
  copia de la raíz `artifacts/fixed/<carpeta>/`, sin leer `current.json`). No se corrige: dejan
  de aplicar a estos modelos y se dan de baja (§9).
- **ml15 lee sus CSV de referencia en cada predict** desde la versión activa del proceso, no
  desde la instancia. Mientras dura una recarga, o si la recarga falla y se restaura el puntero,
  la instancia vieja puede leer los históricos de la versión nueva. Es una ventana corta y el
  resto de modelos fija todo al cargar.
- **Espacio en S3.** Cada versión es una copia completa. Con la frecuencia de reentreno de estos
  modelos son unas decenas de versiones al año por modelo; la limpieza queda para más adelante.

---

## 9. Operación y baja en otros servicios (revisión del 15/10/2026)

### 9.1 Decisiones de operación

| Tema | Decisión |
|---|---|
| Quién reentrena | Solo un entorno controlado, fuera de la plataforma. Un script recopila la última versión de los datasets y llama a `POST /models/<id>/train` con ellos |
| Cuándo | En una secuencia fija por modelo, que sale de su degradación esperada: la frecuencia de actualización de sus fuentes y su ventana de predicción (§1) |
| Usuarios de la plataforma | Solo predicen, siempre con la versión base vigente. No pueden reentrenar estos modelos |
| Explicabilidad (SHAP) | Deja de aplicar a estos modelos |
| Detección de drift (ODD) | Deja de aplicar a estos modelos |

Por eso `/train` y `/rollback` no llevan autenticación propia en este repo: el control es el
entorno desde el que se llaman. Queda abierto añadir un token como defensa adicional si el API
sigue siendo accesible desde la misma red que la plataforma.

### 9.2 Baja en otros repos

Inventario hecho sobre `develop` de cada repo el 15/10/2026. Los nombres con los que cada
servicio conoce a los modelos no coinciden con el `MODEL_ID` de este repo: se listan tal cual.

| Modelo (`MODEL_ID`) | Front (predicción) | Explicabilidad (`TABULAR_MODELS`) | ODD (`models_config.json`) |
|---|---|---|---|
| ml13-wine-price-fluctuation-prediction | Se mantiene | `wine_price_fluctuation` | `modelo-13` |
| ml14-wine-phyto-price-forecast | Se mantiene | `ml14_wine_phyto_price_forecast` | `modelo-14` |
| ml15-wine-ipi-price-forecast | Se mantiene | `ml15_wine_ipi_price_forecast` | `modelo-15` |
| ml16-meat-raw-material-price-alert | Se mantiene | `ml16_meat_price_alert` | `modelo-16` |
| ml17-meat-market-price-analysis | Se mantiene | `meat_price_forecast` | — (no registrado) |
| ml18-meat-spatial-price-forecast | Se mantiene | `ml18_meat_spatial_price_forecast` | `modelo-18` |
| ml21-cereals-price-spatial | Se mantiene | `modelo21_cereales` | `modelo-21` |
| ml23-lactic-market-price-forecast | Se mantiene | `dairy_price_forecast` | — (no registrado) |
| ml19-cereals-cost-forecast | Sin dar de alta: al integrarlo, solo predicción | — (no registrado) | — (no registrado) |
| ml24-dairy-spatial-price-forecast | Sin dar de alta: al integrarlo, solo predicción | — (no registrado) | — (no registrado) |

ml19 y ml24 se añadieron el 19/10/2026 (§10). En las copias locales de `develop` de los tres repos
(13 y 15 de octubre) no están dados de alta en ningún servicio: no hay nada que dar de baja, y en
explicabilidad y ODD no deben darse de alta.

**Front: `retech-lote2-xai-plataforma`**

1. Quitar la acción de entrenamiento de estos modelos en la interfaz: la opción "Training" de
   `static/js/vue/costs-price-predict.js` (ya existe `isTrainActionDisabled()`) y la de cualquier
   otra pantalla que los muestre. Regenerar los bundles (`bundle.js`, `bundle-shared.js`).
2. Rechazar en el servidor los `model_key` de estos modelos en `createTrainTask`
   (`src/controllers/intelligence-controller.ts`). Ocultar el botón no basta: la API de la
   plataforma seguiría aceptando la tarea.
3. Saltar explicabilidad y drift tras predecir con estos modelos en
   `src/service/predict-task-manager.ts`, que hoy llama a `XaiService` y `DriftService` después de
   cada predicción. La predicción se mantiene tal cual.
4. Quitar la opción de explicación o drift de la interfaz para estos modelos, si alguna pantalla la
   ofrece.
5. Actualizar `test/modelo-13-21-wiring.js` y `test/modelo-36-integration.js` (ml23).

**Explicabilidad: `retech-lote2-xai-explicabilidad`**

1. Quitar las 8 entradas de `TABULAR_MODELS` en `app/infrastructure/http/dependencies/container.py`
   (claves de la tabla).
2. Quitar sus ramas en `app/domain/services/plugins/tabular/plugin.py` (`_load()`, `_ml14_*`,
   `_ml15_explain`, `_ml16_explain`, `_ml17_predict_fn`, `_ml18_*`…) y los submódulos que solo usan
   ellos: `tabular/ml14_wine_phyto_price_forecast/`, `tabular/ml16_meat_price_alert/` y
   `tabular/gru_model.py` (ml23), tras comprobar que ningún otro modelo los importa.
3. Quitar las referencias en `app/infrastructure/storage/artifact_store.py` (ml13), los datos de
   fondo `data/background/` de estos modelos y sus tests (`test_explain.py`,
   `test_new_models_xai.py`, `test_ml16_meat_price_alert.py`, `test_renamed_artifact_folders.py`).

Motivo, además de la decisión: este servicio descarga su propia copia de
`artifacts/fixed/<carpeta>/` (la raíz, es decir, la v0) sin leer `current.json`. Tras la primera
promoción explicaría un modelo distinto del que se sirve.

**ODD: `retech-lote2-xai-odd-detection`**

1. Quitar de `models_config.json` las entradas `modelo-13`, `modelo-14`, `modelo-15`, `modelo-16`,
   `modelo-18` y `modelo-21`.
2. Borrar sus baselines ya generados en S3 (`baselines/modelo-NN/baseline.json`) y sus datos
   `datasets/modelo-NN/train.csv`, para que el servicio no los use ni los regenere.

**Skills de este repo**

`front-integration`, `explainability-integration` y `odd-integration` llevan una nota que excluye
estos modelos, para que no se vuelvan a dar de alta.

### 9.3 Pendiente de verificar en INT

- **Código de error del proxy S3 sin `current.json`.** `refresh()` solo trata como "no existe"
  `NoSuchKey`, `404` y `NotFound`. Si el proxy de la plataforma responde otra cosa (por ejemplo,
  403), estos modelos no cargan en el primer despliegue (503).
- **Permiso de escritura** de las credenciales del pod en `artifacts/fixed/` (`put_object` y
  `upload_file`). Hasta ahora ese prefijo solo se leía.
- **Una sola réplica** por pod de estos modelos, sin autoescalado (decisión D3). Con dos réplicas,
  un `/train` fallido en una restaura el puntero a la versión que sirve ella y deshace la promoción
  de la otra.
- **Primer `/train` de cada modelo** contra S3 y MLflow reales: `versions/v1/` completa,
  `current.json` movido y `/stats` con `base_version` v1.
- **ml19 y ml24: artefactos en S3.** A 19/10/2026 `artifacts/fixed/ml19_cereals_cost_forecast/` y
  `artifacts/fixed/ml24_dairy_spatial_price_forecast/` están vacíos: hasta subirlos, los dos dan 503
  al arrancar (acción del responsable, ver los `verification_report.md`).

---

## 10. Ampliación a ml19 y ml24 (19/10/2026)

Los dos son modelos de series temporales que entraron en `main` después de este diseño (PR #20 y
#22) con el patrón anterior. Siguen ya las mismas reglas que el resto (D1-D5, §3-§9).

### 10.1 ml19 (coste de cereales, DATAGIA-19)

- **`/train` pasa de 501 a real.** El manifest lo descartaba porque el entrenamiento solo opera
  sobre el `dataset_v7_fe.csv` que genera el ETL completo (MAPA, ESYRCE, Earth Engine, mercados).
  Con el flujo del §9 ese ETL lo ejecuta el script del entorno controlado y `/train` recibe el
  dataset resultante.
- **Port fiel** de `src/training/train.py::run_multihorizon_training` en `training.py`, con las
  funciones del pipeline oficial copiadas literalmente en `_vendor/training_core.py` (un test
  compara su AST con el entregable). Verificado con el dataset entregado: features, modelos
  elegidos, métricas walk-forward y predicciones de los `.pkl` idénticos bit a bit al original en
  el mismo entorno. ~70 s.
- **Dataset de inferencia.** El CSV recibido se publica como `dataset_v7_fe.csv` de la versión
  nueva, igual que el panel de ml21: resuelve el known issue "dataset congelado en 2026-01".
- `TRAIN_END = 2019-12-31` se mantiene como en el original: solo acota la selección de features y
  el inicio del walk-forward; los modelos finales se reentrenan sobre todos los datos.
- **Fichero sobrante.** Si la selección cambia de modelo (p. ej. h2 clasificación XGBoost →
  LogReg), la versión nueva hereda también el `.pkl` anterior. No se usa (el loader resuelve por
  `training_meta.json`) pero queda en la carpeta.
- **ETL semimanual.** Las descargas de Precios Pagados del MAPA y del cuadro provincial de ESYRCE
  (anual) y la autenticación de Earth Engine son pasos manuales del `GUION_ACTUALIZACION_DATOS.md`
  del entregable: el script de ml19 no puede ser completamente automático.

### 10.2 ml24 (precio lácteo espacial, ESP-LACTEO)

- **Años de validación y test.** Con los del entregable (2019/2020) los paneles se cortan en 2020 y
  un reentreno programado republicaría el modelo de 2020 sin avisar (comprobado: mismas 4063
  filas que el artefacto entregado). Si `/train` no los recibe, se derivan de la serie: test = último
  año con los 12 meses del precio ancla y validación = el anterior. Siguen pudiendo indicarse.
- **Serie de inferencia.** La serie recibida se publica en la versión nueva restringida a los 5
  productos que usan los paneles (26 MB frente a 338 MB; predicciones idénticas, comprobado).
- **Exógenas.** `exog_path` opcional con el formato de `input_costs_monthly.csv`, validado contra las
  columnas vigentes antes de entrenar (400 si faltan). Si viene, se entrena con él y se publica; si
  no, se usan y heredan las vigentes y `upload_warning` lo indica, como los CSV de referencia de ml15.
- **`ml_alpha = 0` con datos recientes (decisión: se activa y se documenta).** Reentrenando con
  validación 2024 y test 2025, la calibración deja `ml_alpha = 0` en h1 y h3 (y `w1 ≈ 0`): la
  predicción queda en el baseline estacional `y_persist_12` y el CatBoost no aporta. Exactitud de
  test 2025: 95-96 % (KPI ≥ 80 %). Es el procedimiento del equipo de IA aplicado a datos nuevos, no
  un fallo del port; queda trasladado al equipo en el manifest.
- **GPU.** CatBoost entrena en GPU si la ve (como el original) y en CPU si no. Los resultados no
  son reproducibles bit a bit entre hardware (ya documentado en el manifest).

### 10.3 Versión de xgboost

El entregable de ml19 usa `xgboost==3.2.0`; el repo pide `xgboost>=2.0.0` (el entorno tenía 3.4.1),
y reentrenar ml19 con 3.4.1 elige LogReg en lugar de XGBoost en h2 de clasificación (AUC de XGBoost
0,7782 → 0,7672). Antes de fijar 3.2.0 se comprobó el impacto en el resto de modelos (3.4.1 frente a
3.2.0):

| Comprobación | Resultado |
|---|---|
| Inferencia: los 20 estimadores xgboost de ml16, ml19, ml21 y ml25 | Idéntica bit a bit |
| Reentreno ml13 (datos entregados) | Idéntico |
| Reentreno ml16 (`dataset_clasificacion_base.csv`) | Umbral `target_animales` 0,44 → **0,48** = el del artefacto entregado; resto igual |
| Reentreno ml21, regresor H3 (único XGBoost) | MAE test 0,1014 → 0,0986 (memoria: 0,1012) |
| Tests de ml13, ml16, ml19, ml21, ml25 y de recarga con 3.2.0 | En verde |

ml25 entrena RandomForest y ml23/ml30 solo mencionan xgboost en metadatos. **Decisión (19/10/2026):
se fija `xgboost==3.2.0` en `requirements.txt`**; el cambio en el H3 de ml21 se acepta.

**Corrección tras fijar la versión.** Con 3.2.0 el reentreno de ml19 sigue eligiendo LogReg en h2 de
clasificación (AUC walk-forward del XGBoost: 0,7538 con 3.2.0, 0,7672 con 3.4.1, 0,7782 entregado). Los
`.pkl` entregados se entrenaron con xgboost 3.2.0 y la misma configuración (`hist`, cpu, semilla 42)
pero en Windows (`run_manifests`: Python 3.11.9 MSC, Windows 10); reajustar en Linux con la misma
versión, datos, scaler (idéntico) y parámetros da árboles distintos (hasta 0,095 en probabilidad).
La diferencia es de plataforma, no de versión: ningún pin la elimina. Los modelos XGBoost de los
reentrenos no son reproducibles bit a bit frente al entregable; la inferencia de los artefactos sí lo
es. El pin se mantiene por el resto de resultados de la tabla.

### 10.4 Tests

`test_ml19_base_promotion.py`, `test_ml24_base_promotion.py` y los casos de ml19 y ml24 en
`test_base_reload_per_model.py` (recarga real desde `versions/v1/` con los artefactos entregados).
La lista de modelos que promueven su base del test de contrato MLflow se lee ahora del registro
(`BASE_ARTIFACT_FOLDER`), con un test que la compara con la documentada.
