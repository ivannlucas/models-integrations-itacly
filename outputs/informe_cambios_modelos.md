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
