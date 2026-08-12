# Verificación — ml9-cereals-infestation-sequence-classifier (a09)

- **Modelo**: Desarrollo de un modelo de Ensemble learning para la clasificación de objetos anómalos
  o infestaciones (INF-CER, lote 2, memoria v3.0 de 13/07/2026)
- **Plugin**: `app/plugins/ml9_cereals_infestation_sequence_classifier/`
- **Rama**: `feature/model-09-integration` (sin commit, sin PR — pendiente de revisión humana)
- **Manifest**: `inbox/a09/manifest.yaml` (20 golden cases, 10 known_issues)
- **Fecha**: 2026-08-04
- **Entorno de verificación**: WSL2, Python 3.10, torch 2.1.2, scikit-learn 1.7.0, pandas 2.2.0,
  fastapi 0.115.0 / starlette 0.38.6 (el repo pinea `fastapi==0.136.1`; ver la nota de entorno)

---

## Checklist técnico (Parte A)

- [x] **flake8** — 0 errores en el plugin, su test y los ficheros tocados
      (`app/registry.py`, `tests/conftest.py`, `app/domain/services/exceptions.py`).
      En el repo quedan 8 avisos **preexistentes** ajenos a a09 (`mlflow_tracker.py` F841,
      `m47/mlflow_utils.py` F841, `ml31/plugin.py` F401 ×2, `exceptions.py` E303/E302/W391,
      `tests/conftest.py` E128) — todos presentes ya en `HEAD`.
- [x] **pytest** — `306 passed, 12 failed`, cobertura global 22 %. Los 10 tests nuevos de a09:
      **9 passed, 1 failed** por el problema de entorno descrito abajo (no por el plugin).
- [x] **pylint** — `10.00/10` en el código propio del plugin (`--ignore=_vendor`, sin
      `line-too-long`, que el repo permite hasta 120 col vía `.flake8`).
      Incluyendo `_vendor/` (copia casi literal del código entregado, que no se refactoriza para
      preservar la reproducibilidad numérica): `8.49/10`, por encima del baseline del repo
      (ml46, mismo patrón de vendorizado: `8.05/10`).
- [x] **pip-audit** — 4 CVEs, **todas preexistentes y ajenas a a09**: `torch 2.11.0`
      (PYSEC-2025-194 → 2.13.0), `setuptools 81.0.0` (PYSEC-2026-3447 → 83.0.0, ×2),
      `cryptography 49.0.0` (CVE-2026-69247 → 50.0.0). a09 **no añade ninguna dependencia nueva**:
      `torch`, `pandas`, `numpy` y `scikit-learn` ya estaban en `requirements.txt`, que no se ha
      modificado.
- [x] **Arranque local + health + stats + predict + train** — OK (ver detalle abajo).
- [x] **/train** — 200 con las 6 métricas de `manifest.training.metrics_returned`
      (`manifest.training.supported: true`, fine-tuning real implementado).
- [x] **Registro en `app/registry.py`** — verificado (AST): imports, `ModelEntry`, tipos de DTO,
      `extra_predict_exceptions=(InsufficientSequenceHistoryError,)` y las 16 entradas previas
      intactas (16 + ml9 = 17).
- [x] **Contrato `ModelPluginPort`** — los 6 métodos implementados
      (`load`, `is_loaded`, `predict_batch`, `predict_inline`, `stats`, `train`).
      No se ha tocado el puerto, ni `app/application/`, ni `app/infrastructure/`, ni `main.py`.
- [x] **`mlflow_utils.py`** presente y funcional, con `try/finally: shutil.rmtree(tmp)` en las tres
      rutas que lo usan (`predict_batch`, `predict_inline` y el `tempfile` de subida en `train`).

### Detalle del arranque local

`python main.py` **no puede ejecutarse en este entorno**: `app/registry.py` importa todos los
plugins del repo y faltan dependencias pesadas de otros modelos (`timm` y, tras él, `cv2`,
`detectron2`, `pytorchvideo`…). Es una limitación conocida del entorno local, no de a09.

Sustituto empleado, con las piezas **reales** del repo (`make_model_router` + `ModelContainer` +
la clase del plugin + los DTO de `predict_dto`/`train_dto`), montando solo la ruta de a09 —
equivalente a `MODEL=ml9-cereals-infestation-sequence-classifier python main.py`:

| Llamada | Resultado |
|---|---|
| `GET /health` | `{"status":"ok","loaded":true,"version":"1.0.0"}` |
| `GET /stats` | 200 — 8 inputs, 8 outputs, `f1_macro=0.9435867`, bloque `ventana` (48/12/last/65 features), `synthetic_data_warning` |
| `POST /predict` (batch, 480 filas) | 200 — 37 ventanas, 1 serie, `class_distribution` |
| `POST /predict` (batch, split de test completo, 28.800 filas) | 200 — 2220 ventanas, 60 series, `evaluated_metrics` |
| `POST /predict` (inline, 48 filas) | 200 — clase + probabilidades de la ventana más reciente |
| `POST /predict` (inline, 20 filas) | 422 de Pydantic (`min_length=48`) |
| `POST /predict` (inline, 50 filas en 2 series de 25) | 422 `InsufficientSequenceHistoryError` |
| `POST /predict` (`mlflow_run_id` no descargable) | 200 — degrada al artefacto fijo con aviso en log |
| `POST /train` (30 series etiquetadas) | 200 — ver tabla de `/train` |
| `POST /train` (CSV sin `target`) | 400 `El CSV de entrenamiento no trae las columnas requeridas: ['target']` |

### Nota de entorno: los 12 tests `*_maps_to_422`

`app/infrastructure/http/router_factory.py` usa `status.HTTP_422_UNPROCESSABLE_CONTENT`, constante
que existe en el `fastapi==0.136.1` que pinea `requirements.txt` pero **no** en el starlette 0.38.6
instalado en este entorno. Consecuencia: **los 12 tests `*_maps_to_422` del repo fallan en local**
(wine-sulphite, ml2, ml4, ml5, ml7, ml8, ml34, ml40 ×2, ml46, ml31, y el de a09), y la misma llamada
por HTTP devuelve 500 en vez de 422.

Es preexistente y ajeno a a09 — no se ha tocado `router_factory.py`. Para demostrar que el mapeo de
la excepción de dominio de a09 **sí** es correcto con la versión pineada, se reejecutó el router real
añadiendo la constante ausente:

```
[1] historial insuficiente -> HTTP 422
    detail: El histórico recibido no genera ninguna ventana: se necesitan al menos 48
            observaciones horarias consecutivas del mismo sample_id (pad_short_sequences=false).
```

---

## Correctitud (golden dataset) — Parte B

**Tolerancia usada**: `float_rtol = 1.0e-4` sobre cada probabilidad (con piso absoluto 1e-6 para
probabilidades ~0) y **coincidencia exacta de clase**, tal como declara
`manifest.metrics_reported.tolerancia_derivada_para_verificacion`. La tolerancia no mide el acierto
del modelo (eso ya está en `metrics_reported`, medido sobre 2220 ventanas): mide la **reproducción
numérica del checkpoint** a través del plugin. Es una tolerancia de aritmética float32, justificada
porque la inferencia es determinista en CPU (`model.eval()`, sin dropout activo, sin muestreo).

`expected` = salida del pipeline **entregado** ejecutado localmente sobre el split de test real
(60 `sample_id` reservados en `model_bundle_metadata.json`). Esa ejecución de referencia reproduce
exactamente las 6 métricas de `metrics_summary.json` y el desglose por clase de la memoria §7.2,
así que es una base fiable.

| Caso | Modo | Ventana | Esperado | Obtenido | Máx. dif. probabilidad | ¿OK? |
|---|---|---|---|---|---|---|
| caso_001 | serie completa | S_0_0061 w4 | sano | sano | 2,48e-08 | sí |
| caso_002 | serie completa | S_0_0024 w17 | sano | sano | 1,50e-08 | sí |
| caso_003 | serie completa | S_0_0052 w27 | sano | sano | 1,43e-08 | sí |
| caso_004 | serie completa | S_1_0007 w1 | sano | sano | 2,54e-08 | sí |
| caso_005 | serie completa | S_1_0046 w9 | insectos | insectos | 4,50e-09 | sí |
| caso_006 | serie completa | S_1_0046 w22 | insectos | insectos | 7,00e-09 | sí |
| caso_007 | serie completa | S_1_0038 w32 | insectos | insectos | 1,50e-09 | sí |
| caso_008 | serie completa | S_2_0019 w0 | sano | sano | 3,50e-09 | sí |
| caso_009 | serie completa | S_2_0026 w3 | insectos | insectos | 2,28e-08 | sí |
| caso_010 | serie completa | S_2_0091 w11 | moho_critico | moho_critico | 1,61e-08 | sí |
| caso_011 | serie completa | S_2_0088 w12 | moho_critico | moho_critico | 4,00e-09 | sí |
| caso_012 | serie completa | S_2_0098 w34 | moho_critico | moho_critico | 4,20e-09 | sí |
| caso_013 † | serie completa | S_1_0045 w2 | insectos | insectos | 1,30e-09 | sí |
| caso_014 † | serie completa | S_1_0050 w14 | sano | sano | 1,87e-08 | sí |
| caso_015 † | serie completa | S_2_0046 w7 | moho_critico | moho_critico | 1,60e-08 | sí |
| caso_016 † | serie completa | S_2_0074 w6 | insectos | insectos | 1,26e-08 | sí |
| caso_iso_001 ‡ | ventana aislada (48 filas) | S_1_0046 | sano | sano | 6,00e-10 | sí |
| caso_iso_002 | ventana aislada (48 filas) | S_0_0061 | sano | sano | 5,50e-09 | sí |
| caso_iso_003 | ventana aislada (48 filas) | S_0_0024 | sano | sano | 2,60e-09 | sí |
| caso_iso_004 | ventana aislada (48 filas) | S_0_0052 | sano | sano | 4,20e-09 | sí |

† Ventanas que el modelo **clasifica mal** respecto a la etiqueta real (`y_true`): caso_013
(y_true=sano → insectos), caso_014 (insectos → sano), caso_015 (insectos → moho_critico), caso_016
(moho_critico → insectos). Están en el golden set **a propósito**: un set solo de aciertos no
detectaría un portado que "arreglara" predicciones. El plugin reproduce también estos fallos, que es
lo que se exige.

‡ `caso_iso_001` documenta la sensibilidad al historial aportado: la misma ventana da *insectos* con
la serie completa (caso_006) y *sano* con solo sus 48 filas. El plugin reproduce **ambos**
comportamientos exactamente.

**Resultado: 20/20 casos dentro de tolerancia.** Máxima desviación observada en todo el set:
**2,54e-08** (4 órdenes de magnitud por debajo del 1e-4 permitido).

### Verificación agregada adicional: split de test completo

Además de los 20 casos, se pasó por el endpoint del plugin el **split de test entero** (60 series,
28.800 filas, 2220 ventanas) con la columna `target`:

| Métrica | Plugin | Declarada (`metrics_summary.json` / memoria §7.2) | \|dif\| |
|---|---|---|---|
| accuracy | 0,941441 | 0,941441 | 0,00e+00 |
| balanced_accuracy | 0,945202 | 0,945202 | 0,00e+00 |
| f1_macro | 0,943587 | 0,943587 | 0,00e+00 |
| precision_macro | 0,942235 | 0,942235 | 0,00e+00 |
| recall_macro | 0,945202 | 0,945202 | 0,00e+00 |
| log_loss | 0,183020 | 0,183020 | 9,90e-09 |

Comparación ventana a ventana contra la ejecución del código entregado:

- ventanas emparejadas: **2220 / 2220**
- discrepancias de clase predicha: **0**
- discrepancias de `y_true`: **0**
- máxima diferencia de probabilidad: **1,19e-07**

Distribución de clases predichas sobre el hold-out: `sano` 802, `insectos` 787, `moho_critico` 631
(la memoria §10 recomienda vigilar desviaciones sostenidas >20 % en esta distribución como señal de
deriva; el plugin la devuelve en cada `predict_batch`).

### /train — resultado real

CSV de prueba: 30 series (10 por clase global) tomadas del split de **train** del entregable,
14.400 filas con `target` y `target_global`.

| Campo | Valor |
|---|---|
| series train / validation / test | 18 / 6 / 6 |
| ventanas train / validation / test | 666 / 222 / 222 |
| época del mejor f1_macro de validación | 3 |
| accuracy / balanced_accuracy | 0,99550 / 0,99608 |
| f1_macro / precision_macro / recall_macro | 0,99518 / 0,99435 / 0,99608 |
| log_loss | 0,014343 |
| f1_macro de validación (early stopping) | 0,98758 |
| `baseline_f1_macro` (modelo servido sobre el mismo hold-out) | 0,99038 |
| `artifact_path` | `artifacts/…/user_final_winner.pt` |

Comprobaciones asociadas:

- Los artefactos fijos **no se sobrescriben**: `final_winner.pt`, `sequence_scaler.pkl` y
  `model_bundle_metadata.json` conservan el md5 del entregable tras ejecutar `/train`. El
  reentrenamiento escribe `user_final_winner.pt`, `user_model_bundle_metadata.json` y
  `user_sequence_scaler.pkl`.
- `self._model` no se muta: el fine-tuning clona los pesos en una instancia nueva, así que un
  `/predict` concurrente nunca ve un modelo a medio entrenar.
- El escalador **se reutiliza, no se reajusta** — el contrato de 65 features del bundle debe seguir
  cumpliéndose y un CSV pequeño de usuario no es base sólida para rederivar el escalado.
- Las métricas de esta prueba son optimistas por construcción (las series salen del split de train
  original, donde el modelo ya estaba ajustado); sirven para validar el endpoint, no para publicar
  rendimiento.

---

## Diferencias deliberadas respecto al código entregado

Todas están documentadas en el docstring del fichero correspondiente y en el manifest:

1. **`_vendor/preprocess.py`** — se eliminan `_ensure_dirs()` y `prepare_processed_datasets()`
   (crean directorios y escriben CSV desde `cfg["paths"]`; la inferencia del plugin es en memoria);
   se corrige el mojibake `"no vÃ¡lidos"`; se normalizan espacios en blanco para flake8 (cosmético,
   reverificado contra los golden cases después del cambio).
2. **`_vendor/sequential.py`** — `train_sequence_model()` acepta `init_state_dict` opcional para
   permitir fine-tuning (con `None` reproduce el comportamiento original desde cero);
   `load_checkpoint()` pasa `weights_only=True` a `torch.load` (el valor por defecto desde torch 2.6,
   que es lo que pinea el repo; comprobado contra el `final_winner.pt` entregado).
3. **`train()` es fine-tuning, no la búsqueda 8+8 desde cero** del entregable, siguiendo el patrón
   del skill `plugin-integration` y el precedente de ml46. La arquitectura ganadora **no** se
   reselecciona: se sigue sirviendo GRU.
4. **`threshold`** de `predict_inline` no altera la predicción: solo marca `low_confidence`. El
   modelo entregado no define ningún umbral de decisión.

---

## Estado final

**LISTO PARA REVISIÓN HUMANA** — el checklist técnico está en verde salvo el fallo de entorno
compartido por todo el repo (starlette/fastapi), y la correctitud es exacta: 20/20 golden cases y
2220/2220 ventanas del hold-out sin una sola discrepancia de clase.

Puntos que requieren decisión o visto bueno humano antes del PR:

1. **Datos 100 % sintéticos** (`known_issues.validacion_solo_sobre_datos_sinteticos`, severidad
   alta). Ninguna cifra de este informe acredita rendimiento sobre cereal real. La ficha funcional
   debe llevar la advertencia de validación pendiente.
2. **Título "Ensemble learning" vs. implementación LSTM/GRU con selección de ganador**
   (`known_issues.discrepancia_titulo_vs_implementacion`). Se ha integrado lo que hace el código y
   se ha documentado; queda confirmar que las fichas mantienen el título administrativo describiendo
   la arquitectura real.
3. **Sensibilidad al historial aportado**: 48 filas es el mínimo operativo, no el óptimo. Conviene
   decidir si el consumidor del API (plataforma/orquestador) va a alimentar el histórico completo de
   cada serie o solo la ventana mínima.
4. **Desfase de entorno starlette/fastapi**: si el pipeline de Bitbucket usa el `requirements.txt`
   pineado, los 12 tests `*_maps_to_422` pasarán allí. Si no, es un arreglo de entorno pendiente
   ajeno a a09.
5. **Artefactos de métricas ausentes en el entregable** (class reports, search summaries, figuras,
   `data/predictions/*`). El desglose por clase se ha reproducido localmente y coincide con la
   memoria; los resúmenes por trial no son verificables y se aceptan como declarados.
6. **`sequence_scaler.pkl` serializado con scikit-learn 1.8.0** y cargado aquí con 1.7.0
   (`InconsistentVersionWarning`). Las predicciones reproducen exactas, pero conviene fijar el rango
   de scikit-learn del despliegue.
