# Reentreno que actualiza el modelo base (ml13, 14, 15, 16, 17, 18, 21, 23)

**Fecha:** 14 de octubre de 2026 · **Rama:** `feature/retrain-updates-base-model` (desde `main` 1c2ebe6)
**Estado:** diseño aprobado en conversación, pendiente de revisión de esta spec

---

## 1. Objetivo

Los modelos de series temporales 13, 14, 15, 16, 17, 18, 21 y 23 se consumen por API desde otra
plataforma. Allí se desarrollan los scrapers, que llaman a nuestro orquestador para predecir o
reentrenar según la frecuencia de cada modelo (semanal en 13, 16 y 17; mensual en general; anual
para el reentreno completo de 15, 18 y 21). Se quiere que **cada `/train` correcto de estos modelos
deje el modelo reentrenado como modelo base**, el que se usa sin `mlflow_run_id`, para que nadie
tenga que guardar un `run_id` para usarlo.

Toma de `docs/propuesta_actualizacion_continua_series_temporales.md` (§6.3 opción B y §7.1) el
principio *"la base nunca se sobrescribe en el sitio: se publica una versión nueva y se mueve un
puntero"*, versionado y reversible.

### Decisiones tomadas

| # | Decisión | Elegido |
|---|---|---|
| D1 | Cuándo se promueve | **Siempre, automáticamente**: todo `/train` correcto pasa a ser la base. El rollback es la red de seguridad |
| D2 | Dónde vive la versión vigente | **Puntero en S3** (`current.json` + `versions/<vN>/`), no MLflow Model Registry |
| D3 | Réplicas | **Un solo pod**: recarga en proceso tras promover, sin sondeo periódico del puntero |
| D4 | `mlflow_run_id` en `/train` | **Opcional** en estos 8 modelos. Si viene, el bundle también se sube al run (trazabilidad) |
| D5 | Datos de contexto | ml21 publica el CSV de entrenamiento como panel; ml15 acepta CSV de referencia opcionales; ml14 lleva sus scalers en el bundle |

### Fuera de alcance

- Scrapers, orquestador, monitorización del error y champion/challenger (resto de la propuesta).
- Regenerar el baseline de ODD y los datos de fondo de SHAP tras promover.
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

### 3.2 Helper común de persistencia (`app/domain/services/base_model_publisher.py`)

Un único helper que usan los 8 `train()`:

```python
def persist_retrained_bundle(
    bundle_dir: str, *, artifact_folder: str, mlflow_run_id: str = "",
    data_path: str, metrics: dict, params: dict | None = None,
) -> PersistResult:   # base_version: VersionInfo, upload_warning: str | None
```

1. Si hay `mlflow_run_id`: `log_params`, `log_metrics` y `upload_artifacts(bundle_dir, "model")`.
   Si falla, **no aborta**: `upload_warning` con el error. Ahora la base es la persistencia que
   importa; el run es trazabilidad.
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
  sobrescribe en el sitio. En los modelos con `BASE_ARTIFACT_FOLDER` (13, 14, 15, 16, 17, 18, 21 y 23)
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

**Sin probar:** S3 y MLflow reales y el servicio arrancado en INT.

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
- **ODD y SHAP** siguen calculados sobre los datos de la v0 hasta que se regeneren a mano.
- **Espacio en S3.** Cada versión es una copia completa. Con la frecuencia de reentreno de estos
  modelos son unas decenas de versiones al año por modelo; la limpieza queda para más adelante.
