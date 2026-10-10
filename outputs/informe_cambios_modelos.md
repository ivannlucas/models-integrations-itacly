# Informe de cambios aplicados a los plugins de modelos

**Repositorio:** `inference-pan-model` (DatagIA) · **Rama:** `fix/model-21-integration`
**Fecha de cierre:** 6 de octubre de 2026
**Alcance:** los 25 plugins registrados en `app/registry.py`

> Nada de lo descrito aquí está commiteado ni tiene PR abierto. Todo queda en el árbol de
> trabajo, pendiente de revisión humana, como exige el `CLAUDE.md` del repo.

---

## 1. Resumen

La auditoría empezó con un único modelo (ml21) y se extendió a todo el repositorio. Su
objetivo era localizar bugs de integración, afirmaciones obsoletas en la documentación y
trabajo que había quedado a medias, para dejar los 25 plugins funcionales.

| Indicador | Valor |
|---|---|
| Plugins registrados | 25 |
| Plugins con código modificado | 20 (más el renombrado `m21_` → `ml21_`) |
| Plugins con reentrenamiento real (`/train`) operativo | **22** (6 de ellos estaban deshabilitados sin motivo; ver patrón C) |
| Plugins donde el reentrenamiento no aplica (justificado) | 3 (ver informe aparte) |
| Bugs críticos que afectaban a predicciones servidas en producción | 6 |
| Ficheros de `app/` y `tests/` modificados | 76 (+2.945 / −2.088 líneas) |
| Ficheros nuevos en `app/plugins/` | 15 |
| Tests (`pytest tests/unit/`) | **505/505** en verde |
| `flake8` sobre los plugins tocados | 0 errores |

### Bugs críticos que afectaban a lo que se sirve hoy

Son los más importantes porque el modelo devolvía resultados incorrectos (o ninguno) a través
de su API real, sin que ningún test lo detectara:

| Modelo | Qué fallaba | Impacto medido |
|---|---|---|
| **ml5** (comportamiento bovino) | El índice de clase se decodificaba con un diccionario distinto del usado al entrenar | 10 de 12 clases mal etiquetadas. Accuracy 0,44 → **0,91** tras el fix |
| **ml7** (plagas en grano) | El nombre de especie se leía de `model.names`, que el pipeline original guarda en orden alfabético mientras las etiquetas usan el orden de `config.yaml` | Especie correcta en el **39,0%** de los casos → **94,4%** tras el fix (500 imágenes reales) |
| **ml21** (precio cereal espacial) | `/predict` devolvía un `dict` en lugar del DTO tipado, y las ~90 features se rellenaban con ceros | `/predict` devolvía **HTTP 500 en todas las llamadas** y, una vez arreglado, las predicciones salían corruptas |
| **ml23** (precio lácteo, GRU) | `predict_batch` no derivaba `current_price` | Devolvía **0 predicciones** (HTTP 200 con lista vacía) |
| **ml25** (sulfitos en vino) | El artefacto servido no era el entregado por el equipo de IA, sino uno generado por un `train()` antiguo con un protocolo inventado | Se servía un modelo distinto del auditado en la memoria |
| **ml2, ml4, ml5, ml7, ml8, ml41, modelo10** | El autotest de CUDA (`_safe_device()`) estaba mal escrito y fallaba siempre | La inferencia corría en **CPU aunque hubiera GPU funcional** |

---

## 2. Patrones de error transversales

La mayoría de los fallos se repitieron en varios plugins. Agruparlos permite entender por qué
existían y cómo evitar que vuelvan.

### 2.1 Por qué los tests no los detectaban

`tests/conftest.py` usa un `FakePlugin` cuyas factorías devuelven directamente el DTO
correcto. Los tests unitarios validan el *routing*, pero **nunca ejecutan el código real del
plugin contra artefactos reales**. Todos los bugs de esta tabla pasaban la suite en verde. Por
eso cada corrección se verificó levantando el servidor real (`MODEL=<id> python main.py`) y
llamando a la API por HTTP.

### 2.2 Tabla de patrones

| # | Patrón | Plugins afectados | Corrección aplicada |
|---|---|---|---|
| A | `/predict` devuelve `dict` en vez del DTO Pydantic; el caso de uso hace `type(result).model_validate(...)` → 500 | ml21 | Envolver la salida en `PredictInlineResponse` / `PredictBatchResponse`. Revisados los 25 plugins por AST: ningún otro caso |
| B | Preprocesado que diverge silenciosamente del código original | ml21 (features a 0), ml7 (`imgsz` 640 en vez de 512), modelo10 (faltaba `CenterCrop`), ml8 (backbone y balanceo distintos) | Replicar el preprocesado exacto del código entregado y verificarlo con datos reales |
| C | `train()` marcado como "no soportado" (HTTP 501) cuando el código entregado sí traía entrenamiento real | ml2, ml4, ml7, ml17, **ml23**, ml5 (clasificador) | Portar fielmente el procedimiento original (optimizador, *loss*, *scheduler*, hiperparámetros) al `train()` del plugin |
| D | `mlflow_run_id` aceptado en predict/stats, pero descartado (`_ = mlflow_run_id`) | ml2, ml4, ml7, ml17, ml23 | Descargar el modelo del usuario desde MLflow y usarlo solo durante esa petición, sin mutar `self.*` y limpiando el directorio temporal en `finally` |
| E | `train_request_type` / `train_response_type` sin conectar en `registry.py` (cae al DTO genérico: 422 o métricas descartadas) | ml21, m47, modelo10, ml5 | Conectar los DTOs reales en `app/registry.py` y en `tests/conftest.py` |
| F | Autotest de CUDA roto: `Conv2d(1,1,1)(tensor.cuda())` deja el módulo en CPU → siempre lanza excepción → siempre CPU | ml2, ml4, ml5, ml7, ml8, ml41, modelo10 | `Conv2d(1,1,1).cuda()(tensor.cuda())`. Verificado: ya no queda ninguna instancia rota en el repo |
| G | Decodificación de clase con un orden distinto al de entrenamiento | ml5, ml7 | Constante fija con el orden real de entrenamiento (`TRAINING_BEHAVIOR_TO_IDX`, `TRAINING_CLASS_ID_TO_CODE`), documentada con la evidencia |
| H | `mlflow_run_id` opcional en `train()` (permitía reentrenar sin dejar rastro en MLflow) | 14 plugins (ver §3.3) | Campo obligatorio, sin valor por defecto. El modelo reentrenado se guarda **solo** en MLflow; el artefacto base de S3 nunca se sobrescribe |
| I | Falta `mlflow_utils.py` (regla obligatoria del repo) | ml2, ml4, ml5, ml7, ml17 | Añadido con lógica real de descarga/subida (no un *stub*) |

---

## 3. Cambios por modelo

Los modelos están agrupados por la profundidad de la intervención. Cada uno tiene su detalle
completo en `outputs/aNN/verification_report.md` y en `inbox/aNN/manifest.yaml`.

### 3.1 Ciclo completo de auditoría (manifest + plugin + verificación + fichas)

#### ml21 · `ml21-cereals-price-spatial` (a21)
- **Renombrado** de `m21_cereal_price_spatial` a `ml21_cereals_price_spatial` (carpeta, clase,
  `model_id`, tests, manifest), siguiendo el estándar `mlNN_<sector>_<desc>`. Comprobado que no
  queda ninguna referencia al nombre antiguo.
- **Crítico:** `/predict` devolvía `dict` → HTTP 500 en todas las llamadas (patrón A).
- **Crítico:** `predict_inline` rellenaba con 0 las ~90 features de ingeniería. Ahora
  `preprocessing.lookup_panel_row()` busca la fila real en `dataset_entrenamiento_final.csv`,
  que pasa a ser artefacto obligatorio.
- **Alto:** `/train` sin conectar en `registry.py` (patrón E).
- **Medio:** `causal_drivers` era siempre un texto fijo. El nuevo `explain.py` replica la
  importancia de features y el ranking por z-score del original.
- Fichas institucionales regeneradas con la nomenclatura nueva.

#### ml5 · `ml5-meat-cow-behaviour` (a05) — 3 ciclos
- **Ciclo 2 — crítico:** el código de entrenamiento entregado tiene dos diccionarios de clases
  incompatibles (`configs/` y `utils/data_utils.py`), que solo coinciden en 2 de los 12 índices.
  El plugin decodificaba con el incorrecto. Se añadió `TRAINING_BEHAVIOR_TO_IDX` en
  `constants.py`. Con 1.500 clips reales, la accuracy pasa de 0,44 a 0,91 y el F1 macro de 0,17
  a 0,87.
- Golden cases reconstruidos desde el dataset real: **24/25** (el único fallo es un error real
  del modelo, investigado y documentado).
- **Ciclo 3 — 4 bugs del código de entrenamiento**, corregidos tanto en el plugin
  (`training.py`, nuevo) como en el código original de `inbox/a05/codigo/`. Se verificaron sin
  reentrenar:

  | Bug | Verificación |
  |---|---|
  | El *scheduler* avanzaba una vez por época, pero su fórmula cuenta pasos por *batch* | Smoke test de 19 s: el LR progresa exactamente como se espera |
  | El split se hacía por imagen, no por clip (fuga de datos entre train/val/test) | 0 clips solapados entre los 3 splits (351/75/76) |
  | `track_id` no era único entre vídeos | 3.693 IDs únicos (antes, 23) |
  | `ColorJitter` también se aplicaba en validación | Inspección del pipeline de *transforms* |

- `/train` habilitado para el clasificador SlowFast (`train_dto.py`, `mlflow_utils.py`,
  conectado en `registry.py`). El detector Faster R-CNN queda fuera del alcance: usa
  Detectron2 con un COCO único fusionado y un split pre-registrado, que nadie ha portado.

#### ml7 · `ml7-cereals-grain-pest-detection` (a07) — 2 ciclos
- **Ciclo 1:**
  - `imgsz` no se pasaba a `model.predict()`, así que se usaba 640 en vez del 512 de
    entrenamiento (patrón B).
  - El *device* estaba fijado a CPU con un comentario incorrecto. Ahora se resuelve con
    `safe_device()`.
  - `train()` estaba deshabilitado sin motivo; se portó con los 300 *epochs* e hiperparámetros
    reales.
  - `mlflow_utils.py` añadido.
- **Ciclo 2 — crítico (con el dataset real de 18.000 imágenes):** el nombre de especie se
  decodificaba mal en 3 de las 5 clases (patrón G). Detalle en el §1. La conclusión del ciclo 1
  ("no afecta a la inferencia") era incorrecta y está corregida en el manifest.
  - Corregido en el plugin (`TRAINING_CLASS_ID_TO_CODE`) y en el código original
    (`create_dataset_yaml()` deja de usar `sorted()`), para que un reentrenamiento futuro no
    reproduzca el fallo.
  - `model.val()` sobre 2.700 imágenes: mAP50 = 0,871, coherente con la memoria (el detector
    siempre estuvo bien).
  - Golden cases reconstruidos: **23/23** contra el servidor real.

#### ml23 · `ml23-lactic-market-price-forecast` (a23) — 2 ciclos
- **Ciclo 1:**
  - `predict_batch` no derivaba `current_price` y devolvía 0 predicciones.
  - `load()` fallaba sin `STORAGE_BUCKET`; ahora carga los artefactos de forma perezosa.
  - `mlflow_utils.py` añadido.
- **Ciclo 2 (detectado al preparar este informe):**
  - `train()` estaba marcado como no soportado, pero `inbox/a23/codigo/` trae entrenamiento
    real (`scripts/train.py` → `runner.py` → `compare_models.py`, con
    `training_enabled: true` en `config.yaml`) (patrón C).
  - Se portó el *refit* de la arquitectura GRU ya seleccionada (`training.py`, nuevo). No se
    repite la búsqueda completa Naive/Drift/XGBoost/LSTM/GRU.
  - Ejecutado sobre el dataset real completo: MAE 0,0135, RMSE 0,017, R² 0,8994. **Coincide a
    4+ decimales** con las métricas del artefacto servido. Duración: 48 s.
  - `mlflow_run_id` ahora se usa de verdad en predict y stats (patrón D).
  - Fichas regeneradas.

#### ml25 · `wine-sulphite` (a25)
- **Crítico:** el artefacto base no era el del equipo de IA. Lo había generado una versión
  antigua de `train()` con `n_estimators=200`, un split 80/20 cronológico y sin el filtro de
  limpieza química. Se sustituyó por el entregado y `train()` replica ahora el protocolo
  oficial (CV 5-fold + ajuste final sobre el 100% de los datos). Reproduce MAE 0,427 / 14,511.
- `build_simulation_grid()` no acotaba la dosis simulada por el percentil 99 de SO₂ libre del
  entrenamiento, así que podía extrapolar. Corregido.
- Golden cases: 10/14 en calidad y 14/14 en SO₂ ligado. Los 4 fallos son vinos de calidad
  extrema; se confirmó que el `.pkl` original (sin pasar por el plugin) falla igual, así que es
  una limitación del modelo.
- Manifest, informe y fichas generados (la memoria estaba pendiente de entrega).

### 3.2 Plugins de visión y regresión: verificación con datos reales y GPU

#### ml2 · `ml2-fungal-cnn-disease-detection` (a02)
- Autotest CUDA (patrón F), `train()` deshabilitado (patrón C), `mlflow_run_id` descartado
  (patrón D) y `mlflow_utils.py` añadido.
- Golden cases reconstruidos con el split de test real: **18/18**. Inferencia y `train()`
  confirmados en CUDA.

#### ml4 · `ml4-lactic-cnn-thermal-early-disease-detection` (a04)
- Autotest CUDA, `train()` deshabilitado (*fine-tuning* real con Focal Loss) y `mlflow_run_id`
  descartado.
- Split oficial de 42 imágenes del dataset TIDS: 61,9% de acierto, frente al 59,52% de la
  memoria. El rendimiento moderado es del modelo, no un fallo de integración.

#### ml8 · `ml8-cereals-img-anomaly-detector` (a08)
- `train()` no replicaba el procedimiento real. Se corrigieron 6 divergencias: pesos del
  *backbone* V1 (no V2), `balance_hongos`, pesos por clase, `WeightedRandomSampler`,
  acumulación de gradiente ×2 y semilla.
- Autotest CUDA corregido.
- Golden cases reales: **19/19** frente al checkpoint servido (16/19 frente al *ground truth*).

#### modelo10 · `modelo10-lacteo` (a10) — 3 ciclos
- **Ciclo 1:**
  - Faltaba `CenterCrop` en el preprocesado.
  - El detector YOLO no recibía `device=` (provocaba un 500 real).
  - Los hiperparámetros de `train()` eran inventados.
  - `/train` devolvía 422 por falta de wiring.
- **Ciclo 2:** autotest CUDA corregido.
- **Ciclo 3, con los 3 datasets reales:** **25/25** golden cases.
  - Hallazgo no bloqueante: el pipeline completo mide 94,94% de acierto (86,73% en mosquitos),
    frente al 98,1% que la memoria reporta solo para el clasificador.
  - **Aceptado como válido por el responsable del proyecto.**

#### ml17 · `ml17-meat-market-price-analysis` (a17)
- `train()` deshabilitado sin motivo. El nuevo *refit* Ridge reproduce **exactamente** la
  Tabla 7 de la memoria (MAE 5,2979, RMSE 7,1946).
- `mlflow_run_id` descartado en predict; `mlflow_utils.py` y `train_dto.py` añadidos.
- Golden cases: 16/18. Los 2 fallos son *shocks* de mercado reales: el mayor es una caída del
  −17,7% intermensual.

#### ml30 · `ml30-meat-traceability-detection` (a30)
- Auditado con el mismo rigor que ml21: **no se encontró ningún bug**.
- 18/18 golden cases, con una diferencia máxima de 2e-8.
- Único cambio: `mlflow_run_id` obligatorio (patrón H).

#### ml41 · `ml41-meat-curing-machinery-acoustic-anomaly` (a41)
- Autotest CUDA corregido y `mlflow_run_id` obligatorio.
- Verificación completada. **El modelo no funciona en producción** porque los 48 *checkpoints*
  nunca se subieron a S3. El plugin lo detecta y lo reporta correctamente (`ready=False`). Es
  una acción operativa, no un bug de código.

### 3.3 Cambio de contrato: `mlflow_run_id` obligatorio en `/train`

Afecta a todo plugin con reentrenamiento real. Se aplicó el patrón H a la firma de `train()` y
al `TrainRequest` de:

`ml3`, `ml9`, `ml16`, `ml30`, `ml34`, `ml35`, `ml40`, `ml41`, `ml45`, `ml46`

…y se implantó así desde el inicio en los plugins cuyo `/train` se habilitó en esta auditoría
(`ml2`, `ml4`, `ml5`, `ml7`, `ml17`, `ml21`, `ml23`, `modelo10`). **Los 22 plugins entrenables
lo exigen hoy**. `/train` sin `mlflow_run_id` devuelve 422, comprobado contra el servidor real
en cada modelo auditado.

Se actualizaron los tests afectados (`test_ml16`, `test_ml34`, `test_ml35`, `test_ml3`,
`test_ml40`, `test_ml45`, `test_wine_sulphite_*`, `test_use_case_system_forwarding`). En los de
wine-sulphite se mockea `BaseMLflowTracker`, que no tiene *timeout* y bloqueaba la suite.

### 3.4 Wiring y limpieza documental (sin cambios de lógica)

| Modelo | Cambio |
|---|---|
| m47 | `TrainResponse` real conectado en `registry.py`: antes descartaba `exact_match`, `accuracy`, `f1_macro`… |
| a47, a31, a40, a46, a43 | Afirmaciones obsoletas en manifests e informes marcadas `[RESUELTO]`. Se conserva el texto original por trazabilidad |
| a31 | Fichas regeneradas: describían el modelo *surrogate* ANN+GA v1.x, no el optimizador LP v2.0 que se sirve |
| a23 | La ficha técnica cita ahora las métricas del artefacto servido (`GRU_artifact_metrics`), no la media de 3 semillas |

Los plugins ml28, ml31, ml33, modelo43, ml3, ml9, ml16, ml34, ml35, ml40, ml45, ml46 y m47 ya
tenían verificación previa en verde. En esta auditoría solo se les aplicaron los cambios de
§3.3 y §3.4, y se confirmó que siguen pasando.

---

## 4. Documentación institucional

Para cada modelo auditado se generaron o regeneraron `manifest.yaml`, `verification_report.md`
y las fichas técnica y funcional. Las fichas usan siempre la plantilla paramétrica fija de
`docs-generation`, nunca se editan a mano.

| Modelo | Manifest | Informe de verificación | Fichas |
|---|---|---|---|
| a02, a04, a05, a08, a10, a17, a30 | Nuevo | Nuevo | Generadas |
| a07 | Nuevo | Nuevo (2 ciclos) | Regeneradas (estaban desfasadas: decían "solo CPU" y "train 501") |
| a21 | Actualizado | Nuevo | Regeneradas con la nomenclatura `ml21` |
| a23 | Actualizado | Ampliado (ciclo 2) | Regeneradas |
| a25 | Nuevo | Nuevo | Generadas |
| a31 | Actualizado | — | Regeneradas (LP v2.0) |
| a41 | Existente | Nuevo | — |

---

## 5. Verificación técnica final

- `pytest tests/unit/ -q` → **505/505 passed**.
- `flake8` → 0 errores en todos los plugins tocados.
- `pylint` → entre 8,5 y 9,7/10 por plugin. Los avisos restantes son los mismos que ya acepta
  el repo: `invalid-name` en variables `X`/`X_train`, imports perezosos de torch/ultralytics y
  falsos positivos `E1101` de cv2/PIL.
- `pip-audit` → solo `setuptools==80.9.0` (PYSEC-2026-3447). Es preexistente y afecta a todo
  el repo, no a ningún plugin concreto.
- Ningún proceso de servidor quedó vivo tras las pruebas en vivo.

---

## 6. Pendientes fuera del código

Ninguno se resuelve desde el repositorio:

| # | Modelo | Acción | Prioridad |
|---|---|---|---|
| 1 | ml41 | Subir los 48 *checkpoints* a `s3://…/artifacts/fixed/ml41_…/`. Sin ellos el modelo no sirve ninguna predicción | **Bloqueante para producción** |
| 2 | ml25 | Sustituir en S3 el artefacto antiguo por el corregido (solo está corregido en local) | **Alta** |
| 3 | ml21 | Confirmar que `dataset_entrenamiento_final.csv` está subido a S3 junto a los `.joblib`. Sin él, `load()` falla | Alta |
| 4 | m47 | Recalibrar con datos reales de planta. Se entrenó con un banco de laboratorio de aceite, no de leche | Bloqueante para producción, no para el PR |
| 5 | ml5 | Ejecutar el entrenamiento completo (~50 h de GPU) cuando haya presupuesto. Revisar la diferencia de AP del detector (0,705 real frente a 0,882 en la memoria) | Media |
| 6 | Todos los entrenables | Probar `/train` contra un MLflow real accesible (desde el *sandbox* no hay red hacia el clúster). Por decisión del responsable, queda documentado tal cual | Baja |
| 7 | a35 | El manifest no tiene bloque `training:`, porque es de un formato anterior a esa convención. El entrenamiento funciona; es solo un hueco documental | Baja |

---

## 7. Anexo (6 de octubre de 2026): nomenclatura, memoria de ml33 y documentación del repo

Cambios posteriores a la redacción de este informe. En las secciones anteriores, los nombres
antiguos se mantienen como referencia histórica.

### 7.1 Nomenclatura unificada (`mlNN_<sector>_<desc>`)

| Antes | Ahora (carpeta = `ARTIFACT_FOLDER_NAME`) | `MODEL_ID` / ruta de API |
|---|---|---|
| `modelo10_lacteo` | `ml10_dairy_disease_vector_detection` | `ml10-dairy-disease-vector-detection` |
| `modelo43_cereales` (artefactos `modelo_43_cereales`) | `ml43_cereals_dnsl_anomaly_fault_detection` | `ml43-cereals-dnsl-anomaly-fault-detection` |
| `m47_dnsl_fallas_maquinaria_pasteurizado` (artefactos `a47_…`) | `ml47_dairy_dnsl_pasteurization_fault_detection` | `ml47-dairy-dnsl-pasteurization-fault-detection` |
| ml8, artefactos `modelo8_cereales` | `ml8_cereals_img_anomaly_detector` | sin cambios |
| ml25, id `wine-sulphite`, artefactos `wine_sulphite` | `ml25_wine_sulphites` (`MODEL_ID` y `VERSION` movidos a `constants.py`) | `ml25-wine-sulphites` |

Carpetas y tests movidos con `git mv` (se conserva el historial). Se actualizaron clases,
alias de `registry.py` y `conftest.py`, manifests, `datos_*.json` (`model_key`/`model_slug`) y
las fichas regeneradas (a08, a10, a25, a47). Los manifests y los informes de verificación
llevan una nota `[RENOMBRADO 2026-10-06]`. Verificación: flake8 limpio, 536/536 tests y los 5
modelos arrancan contra el servidor real con `/health → loaded: true`.

**Acciones operativas necesarias antes de desplegar:**
1. **S3:** copiar cada `artifacts/fixed/<viejo>/` a `artifacts/fixed/<nuevo>/`. Sin esto, los 5
   modelos dejan de cargar en producción.
2. **Consumidores de la API:** el front, el servicio de explicabilidad y el ODD deben usar los
   nuevos `model_id` (cambian las rutas `/models/<id>/...` de ml10, ml43, ml47 y ml25).

### 7.2 ml33: memoria entregada

Se añadió la memoria v2.0 (21/07/2026) en `inbox/a33/entregable/`. Contrastada con el manifest,
el informe y el plugin: **todas las cifras coinciden** y no cambia ningún resultado. Su §10.3
confirma de forma explícita que el modelo desplegado no se reentrena. Cambios:
- Manifest con título oficial, acrónimo (MILPCO2), versión 2.0, métrica diagnóstica (27,39 %)
  y un known_issue sobre restos de redacción v1.x en la memoria.
- Fichas regeneradas con la robustez Monte Carlo y tres limitaciones nuevas.
- `a33_metadatos.docx` con organismo, lote, versión y fecha.
- El borrado de los 98 ficheros de `inbox/a33/` es correcto: el código se movió a
  `inbox/a33/codigo/`, que no se commitea.

### 7.3 Documentación del repo

- `CLAUDE.md`: `inbox/aNN/` (manifest y memoria) y `outputs/` **sí se commitean**; solo
  `inbox/*/codigo/` queda excluido. Se añade la regla de nomenclatura de plugins.
- `.claude/skills/plugin-integration/SKILL.md` reescrita con lo aprendido en la auditoría.
  Corrige tres instrucciones que inducían a error: `mlflow_run_id` opcional en `train()`,
  guardar el reentrenamiento "localmente siempre" y ml2/ml5/ml7 como ejemplos de modelo no
  entrenable. También corrige una plantilla de `mlflow_utils.py` que importaba una función
  inexistente.

---

## 8. Anexo (7 de octubre de 2026): persistencia solo en MLflow y aislamiento de modelos de usuario

Cambios de la rama `fix/retrain-mlflow-only-persistence` sobre esta auditoría. Algunos corrigen
errores de la propia auditoría y se indican como tales.

> **Corrección de cifras.** Con ml15 (integrado en `main`) el registro tiene **26 plugins, 23
> entrenables**, no 25 y 22 como dicen las secciones anteriores.

### 8.1 Corrige errores de esta auditoría

| Problema | Alcance | Corrección |
|---|---|---|
| **Fuga del modelo de usuario entre peticiones.** El patrón D de la §2.2 afirmaba "sin mutar `self.*`", pero ml2, ml4 y ml17 (y antes ml8, ml9, ml21, ml25, ml30, ml34, ml35 y ml46) sustituían el modelo en `self` durante la petición. Hay una sola instancia del plugin, así que una petición sin `mlflow_run_id` podía recibir el modelo de otro usuario | 11 plugins. Reproducido: 46/46 casos fallan en la rama anterior | El modelo se resuelve en variables locales (`test_user_model_isolation.py`) |
| **ml31** seguía sustituyendo sus datos de referencia en `self` y llamando a MLflow, aunque no es entrenable | 1 plugin | Ignora el run sin tocar MLflow, igual que ml28 y ml33 |
| **ml41** guardaba el checkpoint ajustado **encima del checkpoint base local**, y ml15, ml16, ml3, ml40 y ml9 dejaban copias `user_*` en la carpeta de artefactos | 6 plugins | El reentrenamiento solo se guarda en MLflow |
| **ml47**: su `trainer.py` asignaba a todos los ciclos la etiqueta del primero, mezclaba ciclos en las medias móviles y calculaba `f1_macro = accuracy` | 1 plugin | Port fiel de los dos procedimientos entregados (`fine_tune` y `full`), verificado con datos reales |

### 8.2 Contrato nuevo de MLflow

- `mlflow_run_id` en `/train` es obligatorio **y no vacío** (`MlflowRunId`). Antes `""` pasaba la
  validación, se entrenaba y el modelo se descartaba.
- Si el modelo reentrenado no se puede guardar en MLflow, `/train` responde **502**
  (`ModelPersistenceError`) en lugar de 200 con un aviso.
- Si se pide un run cuyo modelo no se puede cargar, `/predict` y `/stats` responden **422**
  (`UserModelUnavailableError`) en lugar de usar el modelo base sin avisar.

### 8.3 Correcciones añadidas en la revisión de esa rama

| Problema | Corrección |
|---|---|
| Un run con artefactos **incompletos** (p. ej. un entrenamiento que falló después de que la plataforma creara el run) hacía que 12 plugins respondieran **500**, en lugar del 422 previsto | `require_user_model` convierte en `UserModelUnavailableError` (422) los errores de artefacto al cargar un run pedido (`ARTIFACT_LOAD_ERRORS`: `OSError`, `ValueError`, `KeyError`, `EOFError`, `UnpicklingError`, `RuntimeError`, comprobados con ficheros ausentes, truncados, corruptos y de otra arquitectura). Cualquier otra excepción (`AttributeError`, `TypeError`…) es un bug del loader y sigue saliendo como 500 |
| Los 23 `download_*_from_mlflow` dejaban su directorio temporal en disco cuando el run no tenía modelo o fallaba la carga. Con el nuevo 422, cada reintento de un usuario sumaba un directorio huérfano | Limpieza en todas las salidas de error (`try/except` + `rmtree`) |
| `test_full_training_is_reproducible` (ml47) fallaba en máquinas con GPU: cuDNN no es determinista | El test fija la CPU |
| ml47 elegía la GPU con `torch.cuda.is_available()` sin autotest, en inferencia y en entrenamiento | `safe_device()`, igual que ml2, ml4 y ml8 |
| Nada impedía que un plugin nuevo repitiera la fuga entre peticiones o la de temporales | `tests/unit/test_user_model_download_hygiene.py`: test estructural sobre todos los plugins, que sigue también los métodos de la clase llamados desde `predict*`/`stats`, más un test de limpieza para cada helper de descarga. No cubre funciones de módulo, `setattr` ni métodos heredados de otro fichero |

### 8.4 ml47: reanálisis completo contra su código original

Con el código entregado completo en `inbox/a47/codigo/` se ha comparado el plugin, fichero a fichero,
con `preprocess.py`, `trainer.py`, `fine_tuner.py`, `model.py` y `predictor.py`, y se ha comprobado
con los datos reales del entregable.

**Qué coincide con el original:**

- Artefactos (md5 idéntico).
- Split 70/15/15 (idéntico a `cycle_splits.json`, incluido el orden).
- `ts1_mean_train`.
- Features de test (diferencia máxima 9,6e-5, por redondeo del CSV en valores de ~2.400).
- Métricas del modelo base en test (0,9879 / 0,9970 / 0,9970).
- Inferencia: los 50 primeros ciclos de `hydraulic_raw.csv` dan las mismas clases que `prediction_output.csv`, con una diferencia de confianza de 1,2e-4 como máximo.

**Corregido:**

| Problema | Corrección |
|---|---|
| `/predict` por lotes fallaba con `KeyError: 'Time_Segundos'` si el CSV venía en el formato bruto del banco (columna `Time` a 100 Hz, el que lee `main.py predict`) | Se usa `Time` como hace `predictor.apply_digital_twin_inference`. Si falta la columna de tiempo, el error lo indica explícitamente |
| El modelo se construía con `dropout_prob=0.5` en lugar del 0,2022 de `config.yaml`. En inferencia no influye, pero `fine_tune` entrenaba `dropout_final` con 0,5 | `load_artifacts_from_dir` usa `TRAIN_HYPERPARAMS["dropout_rate"]`. El helper de MLflow reutiliza ese mismo cargador (y con él `weights_only=True` para el `state_dict` del usuario) |
| Los tensores de entrenamiento seguían el orden barajado del split. El original los ordena por `Cycle_ID` | Orden ascendente por `Cycle_ID` en `build_tensors` y en los ids de las copias aumentadas. Con datos reales los tensores coinciden con los del pipeline original elemento a elemento, en el mismo orden |

**Resultado:** el entrenamiento `full` desde cero (GPU, `hydraulic_10hz_raw.csv`, gemelo térmico
activo) reproduce exactamente las métricas del modelo entregado: 0,9879 / 0,9970 / 0,9969 / 0,9972 /
0,9970, en 89 épocas y 122 s. Antes de la corrección daba 0,9940. `fine_tune` sobre `val_split.csv`
da un exact match de 0,96 y deja el modelo base intacto.

**Verificado y sin cambios:**

- `fine_tuner.py` también llama a `model.train()` sobre el modelo completo, así que las BatchNorm del backbone congelado actualizan sus estadísticas igual que en el plugin.
- La densidad y el Cp del fluido (`fluid_density_kg_l`, `fluid_cp_kj_kgK`) solo se registran en el informe de calibración, no intervienen en el cálculo.
- El fine-tuning original no aplica el gemelo térmico.

## 9. Anexo (8 de octubre de 2026): revisión de la PR de auditoría

### 9.1 ZIPs sin límite de descompresión

`zf.extractall` se usaba en **12 sitios de 7 plugins** (ml2, ml4, ml5, ml7, ml8, ml10, ml41), en
`/train` y en `/predict` por lotes. Python ya neutraliza `..` y las rutas absolutas, así que no
había *zip-slip*, pero no había ningún tope: un ZIP de pocos KB podía expandirse a cientos de GB.

Ahora los 12 pasan por `safe_extract_zip` (`app/infrastructure/archive.py`):

- Descomprime en *streaming* y cuenta los bytes realmente escritos. No se fía de los tamaños de
  la cabecera del ZIP.
- Corta al superar el tamaño total (`ARCHIVE_MAX_TOTAL_BYTES`, 10 GiB), el número de ficheros
  (`ARCHIVE_MAX_FILES`, 200.000) o el ratio de compresión por fichero (`ARCHIVE_MAX_RATIO`, 200).
  Los tres se configuran por entorno.
- Comprueba antes el espacio libre en disco.
- Rechaza las rutas que salen del destino.

El ratio es el que detecta un zip bomb: imágenes, vídeo y audio comprimen cerca de 1x. Los
límites de tamaño y de número de ficheros quedan holgados respecto a los datasets entregados.

Respuestas: **413** (`ArchiveLimitExceededError`) en `/train` y `/predict`; un ZIP corrupto da 400
en `/train`. Comprobado contra el servidor real de ml2: un ZIP de imágenes reales da 200 y un zip
bomb de 64 KB → 64 MB da 413.

### 9.2 ml10 `/train`

| Problema | Corrección |
|---|---|
| Rama `if not mlflow_run_id` muerta (`mlflow_run_id` es obligatorio en `TrainRequest`) | Eliminada, junto con las comprobaciones redundantes |
| Si `torch.save` fallaba, el temporal de MLflow no se borraba | `try/finally` |
| `random.shuffle` sin semilla en el split automático | Ficheros ordenados y `random.Random(42)` local, con la semilla de `config.yaml`. Además, `torch.manual_seed(42)` |
| El 15 % de test se separaba y nunca se evaluaba | Evaluado con las fórmulas de `src/predict/predictor.py` del original: `test_accuracy` y `test_per_class` (P/R/F1). Se registra en MLflow |
| No detectado en la revisión: el split automático no aplicaba el tope `splits.max_ticks: 1400` del original | Aplicado |

Los hiperparámetros fijos no son un defecto: son los de `config.yaml` del equipo de IA.

### 9.3 ml5 `training.py`

- `lr_lambda` devuelve un `float` con `math.cos(math.pi * progress)`. El original usaba un tensor
  y `3.14159`; la diferencia en el LR es ≤ 1e-6 relativo.
- Con menos de 4 clips, el split 70/15/15 por clip lanza un error claro. Antes salía el
  `ValueError` de sklearn. El mínimo real es 4, comprobado.
- `train_classifier` rechaza la validación vacía. `plugin.train()` ya lo comprobaba antes de
  llamarla; ahora también lo hace la función.

### 9.4 ml23 `training.py`

- Corregida la anotación de `prepare_rnn_split_with_test`: devuelve 8 elementos.
- `prepare_horizon_dataset` da un error claro si ninguna serie supera el horizonte. Antes fallaba
  con `No objects to concatenate`, o más adelante con un error de forma.
- Se mantienen las 3 semillas del original (48 s medidos).

### 9.5 Entrenamiento síncrono

No se añade un timeout dentro del plugin: un hilo de PyTorch no se puede cancelar. La petición
respondería 504 mientras el entrenamiento sigue en la GPU y guarda en MLflow un run que la
plataforma da por fallido.

Queda documentado en `plugin-integration` (§ Duración) y en el campo
`training.duracion_estimada` de los manifests de a05, a10, a23 y a47:

- ml5: más de 19 min sin terminar.
- ml47 `full`: 122 s.
- ml23: 48 s.

La solución de fondo es un `/train` asíncrono (202 + estado), que requiere cambios en la
plataforma.

### 9.6 Cobertura de tests

`test_ml10_…_unit.py` y `test_ml25_…_unit.py` sustituyen el modelo por `MagicMock`: cubren las
funciones puras, no la correctitud de la API. El SKILL lo indica de forma explícita.

Los tests nuevos de esta ronda llaman al código real:

- `test_training_robustness.py`: split de ml10, métricas de test, errores de ml5 y ml23.
- `test_archive_limits.py`.

La correctitud de cada modelo sigue respaldada por su informe de verificación contra el servidor
real.

## 10. Anexo (8-10 de octubre de 2026): ml13, ml14, ml18, ml26 y ml41

### 10.1 ml13 (`ml13-wine-price-fluctuation-prediction`)

La verificación original era sólida: inferencia idéntica al código del equipo de IA (diferencia
máxima 1e-16) y reentrenamiento equivalente. Lo que faltaba era el contrato de reentrenamiento de
esta rama.

| Antes | Ahora |
|---|---|
| `mlflow_run_id` opcional | Obligatorio (422 si falta o va vacío) |
| Modelo reentrenado guardado en `artifacts/` (`user_*`) y, opcionalmente, en MLflow | Solo en MLflow |
| Fallo de subida → 200 con `upload_warning` | 502 (`ModelPersistenceError`) |
| `/predict` con un run sin modelo → 200 con el modelo **base** | 422 (`UserModelUnavailableError`) |
| Temporal de descarga sin limpiar si fallaba la carga | `@require_user_model` y limpieza en todas las salidas |

Re-verificación con datos reales:

- `/train` sobre `mapa_wine_prices_raw.csv`: AUC 0.8438, F1 0.5926, P 0.4211, R 1.0 y Acc 0.5417,
  igual que antes.
- Predicción base sin cambios (0.4621).

Hay tests nuevos sobre el plugin real: solo se sube a MLflow, un fallo de subida da 502 y un run
vacío da 422.

### 10.2 ml18 (`ml18-meat-spatial-price-forecast`): verificado y `/train` habilitado

La primera copia del código subida era la versión anterior (6 features, MAPE 12,15 %). Con la
versión v4.2 (commit 5ceeeb7, 7 features), todo se ha comprobado con datos reales.

**Inferencia.**
- El plugin coincide con `predict --forecast` del original en las 400 combinaciones CCAA-Producto
  (diferencia máxima 2e-6).
- Los 4 golden cases dan 4/4.
- El original reproduce las métricas de la memoria (test MAPE 9,932 %, R² 0,8114).

**Reentrenamiento.** Antes no estaba disponible (501) porque el script lee una ruta fija. Con el
criterio de `plugin-integration` eso no lo impide: el motor servido es una GRU con pesos
aprendidos. Ahora `/train` funciona:

- `training.py` es un port literal de `src/main.py::train()`.
- Ejecutados con el mismo TensorFlow, el original y el plugin dan **las mismas métricas a 4
  decimales**: test MAPE 9,9552 %, R² 0,8110, 17 épocas.
- Tarda unos 2,5 min en CPU.
- El modelo se guarda solo en MLflow y se usa con `mlflow_run_id`.
- Los errores siguen el contrato común (422/502).

**Otros cambios.**
- Resolución de la columna de precio con cabeceras con variantes de codificación, como el original.
- `/stats` incluye los datos del run.
- Tests nuevos: `test_ml18_training.py`, con código real y TensorFlow.
- ml13, ml18 y ml26 añadidos a `test_user_model_isolation.py`.
- Fichas técnica y funcional regeneradas.


### 10.3 ml14 (`ml14-wine-phyto-price-forecast`): verificado y `/train` habilitado

El código subido es la entrega aprobada: los hashes de `gru_model.pt` y del dataset coinciden con
los del manifest.

**Inferencia.** El plugin coincide con el predictor original en 40 ventanas repartidas por todo el
histórico (diferencia máxima 1e-6).

**Reentrenamiento.** Antes no estaba disponible (501) con el mismo argumento que ml18, la ruta fija
del dataset. `training.py` porta la parte GRU de `compare_models.py`:

- El `/train` del plugin y el original re-ejecutado reproducen `model_comparison.json` a 4
  decimales (RMSE 2,2873 ± 0,0268).
- El modelo de la semilla 42 es idéntico al del original re-ejecutado y difiere del entregado en
  2e-7.
- Tarda unos 15 s.
- Acepta el dataset de modelado o las series en bruto de `/predict`.

**Hallazgos nuevos.**

- El early stopping del original se decide sobre el test, así que las métricas son optimistas. Se
  reproduce tal cual y queda documentado.
- La evaluación mensual del propio equipo confirma que la GRU mejora al Drift (RMSE 2,55 frente a
  2,72).

**Otros cambios.**

- `/stats` incluye los datos del run.
- Tests nuevos: `test_ml14_training.py`.
- ml14 añadido a `test_user_model_isolation.py`.
- Fichas regeneradas.

### 10.4 ml26 (`ml26-wine-sulfite-gru-pso-forecast`): `/train` corregido, verificado y documentado

Con el código original ya en `inbox/a26/codigo/` (commit 62862cc, memoria v2.5):

**`/train` no era el procedimiento del equipo de IA.**

- **Antes:** hacía un ajuste fino de los pesos servidos y conservaba la normalización servida. El
  equipo de IA nunca definió ese procedimiento, y KI-04 lo describía de otra forma.
- **Ahora:** `training.py` porta literalmente `train_sequence_with_config`. Crea un modelo nuevo con
  la configuración final de PSO, fija la semilla 7 justo antes de entrenar, como el original, y
  ajusta el z-score sobre el train del usuario. La búsqueda PSO no se repite.
- **Resultado:** con los datos entregados, el plugin y el original ejecutado aislado dan pesos
  idénticos (diferencia 0). El reentrenamiento tarda unos 4 minutos en CPU.
- **Hallazgo del port:** las ventanas deben ser C-contiguas. Si no, la media float32 del z-score
  cambia en el último bit y el modelo diverge tras 25 épocas.

**Inferencia verificada frente al original.**

- La preparación de datos coincide con los `.npy` de los splits (diferencia 0).
- Las 3.096 ventanas de test coinciden con `gru_pso_test_predictions.csv` (diferencia máxima 4e-6).
- 30 lotes en modo operativo coinciden con `raw_inputs.py` (diferencia máxima 1,7e-5).
- Los 18 golden cases pasan en los dos modos contra el servidor real. Sin `stage_progress` la
  petición se rechaza con 422.

**Documentación.** Primer `outputs/a26/verification_report.md` del repo; el manifest lo citaba, pero
nunca se commiteó. También se han generado `datos_a26.json`, `datos_a26_funcional.json`, las fichas
técnica y funcional (plantilla paramétrica) y `a26_metadatos.docx` (plantilla corporativa). En el
manifest se corrige KI-04 y se añade KI-13 (artefactos ausentes en S3).

### 10.5 ml41 (`ml41-meat-curing-machinery-acoustic-anomaly`): `/train` corregido, verificado y documentado

Con el código original en `inbox/a41/codigo/` (los 48 checkpoints Audio-MAE y los 48 del baseline,
sin audio) y la memoria v1.4:

**Inferencia y umbrales verificados frente al original.**

- Los 48 umbrales del plugin coinciden con `reports/table_auc_por_caso.csv`.
- 144 casos (48 combinaciones × 3 WAV sintéticos), en CPU: espectrograma idéntico, `maha_score` con
  diferencia relativa ≤ 4,8e-7 y etiqueta idéntica.
- Matiz al manifest: `maha_score` no es determinista al bit, porque el encoder baraja parches incluso
  con `mask_ratio=0`. La variación es ≤ 6e-7 en CPU y llega a 1,7e-4 entre CPU y GPU.

**`/train` no era el procedimiento del equipo de IA.**

- **Antes:** hacía un ajuste fino del checkpoint base con su normalización, un split
  `default_rng`, la semilla fijada después de crear el modelo y `drop_last` condicional.
- **El original:** `run_training` entrena siempre un modelo nuevo, o se salta el entrenamiento si ya
  existe uno.
- **Ahora:** es un port literal con el mismo split (`RandomState(42)`, 80/20), normalización sobre
  el train, semilla antes de crear el modelo y `drop_last=True`.
- **Resultado:** con los mismos datos, pesos idénticos al original (diferencia 0).
- **Coste:** en CPU, 1-1,5 h por combinación (el original tardaba 155 s en GPU).

**Cierre con el dataset MIMII completo (mismo día).**

- **Golden cases: 22/22** contra el servidor real. El audio de `sample_data/` resultó ser el de
  `-6_dB_fan/id_00`, no el de 0 dB; un barrido de las 48 carpetas lo localizó. Corregido en el
  manifest.
- **Métricas de las 48 combinaciones** recalculadas con el plugin sobre el test real: AUC medio
  idéntico (0,7720), diferencia máxima por combinación 2,9e-4 y 48/48 con FNR ≤ 10 %.
- **Entrenamiento en GPU.** El original usa autocast bf16 y GradScaler, y el plugin entrenaba en fp32;
  portado. Con eso, el reentrenamiento real de fan/id_00/0_dB en GPU da lo mismo que el original a 4
  decimales (val_loss 0,7400, AUC 0,7896, umbral 7,2069), en unos 10 minutos.
- **Pendiente:** solo la acción operativa de subir los 48 directorios `vit_tiny_*` a S3.

**Documentación.** Fichas técnica y funcional (plantilla paramétrica), `a41_metadatos.docx`,
`datos_a41*.json`, manifest (training, determinismo de `maha_score`, memoria v1.4, S3) y sección
nueva en el informe de verificación.

### 10.6 S3

Ni ml13, ni ml14, ni ml18, ni ml26, ni ml41 tienen artefactos en `artifacts/fixed/<ARTIFACT_FOLDER_NAME>/`, así que en
despliegue arrancarían con `loaded=false`. Los artefactos válidos son:

- ml13: `inbox/a13/codigo/models/prod/`.
- ml14: `inbox/a14/codigo/models/artifacts/gru_model.pt` y `data/processed/final_dataset_for_modeling.csv`.
- ml18: `inbox/a18/codigo/models/artifacts/best_gru_df_2008_con_renta/`.
- ml26: `inbox/a26/codigo/models/artifacts/gru_pso.pkl` y `best_model.json`.
- ml41: los 48 directorios `vit_tiny_*` de `inbox/a41/codigo/models/artifacts/`.
