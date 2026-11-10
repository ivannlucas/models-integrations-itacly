# Informe pylint del repositorio — 2026-11-05

Revisión completa con pylint de `app/`, `main.py` y `tests/`. No se ha modificado código.

## Cómo se ha ejecutado

```bash
.venv/bin/python -m pylint app main.py tests --recursive=y --jobs=0 --output-format=json
```

| Dato | Valor |
|---|---|
| Commit revisado | `398bbab` (`main`, merge del PR #31) |
| pylint / astroid / Python | 4.0.8 / 4.0.4 / 3.12.14 |
| Configuración | La de pylint por defecto: el repo no tiene `.pylintrc` ni sección `[tool.pylint]` |
| Pipeline | `bitbucket-pipelines.yml` no ejecuta lint (paso `echo` de marcador); flake8 se pasa a mano |
| flake8 (`.flake8`, E501 ignorado) | 0 avisos |

Al no haber configuración, pylint aplica sus valores por defecto: líneas de 100 caracteres (flake8 usa 120), docstrings obligatorios, máximo 5 argumentos, etc. Buena parte del volumen viene de ahí, no de defectos.

## Resumen

| Tipo | `app/` + `main.py` | `tests/` | Total |
|---|---:|---:|---:|
| Error (E) | 56 | 58 | 114 |
| Warning (W) | 88 | 693 | 781 |
| Refactor (R) | 406 | 436 | 842 |
| Convention (C) | 1803 | 1839 | 3642 |
| **Total** | **2353** | **3026** | **5379** |

Tras revisar los 114 errores y los avisos con riesgo funcional:

- **1 fallo real en producción: ml35, modo `optimize`**, con dos causas independientes (sección 1).
- **El resto de errores son falsos positivos** de pylint con librerías nativas o dinámicas (OpenCV, TensorFlow, PyTorch, Pydantic, pandas). Se ha comprobado cada familia (sección 2).
- Unos pocos avisos señalan código mejorable sin efecto funcional hoy (sección 3).
- El resto es estilo, complejidad o duplicación (sección 4).

---

## 1. Fallo real: ml35 `optimize` no funciona

`app/plugins/ml35_dairy_ann_cleaning_cost/plugin.py`, método `_run_optimize` (líneas 197-235). Pylint lo señala por `import-error` en `pygad` (línea 199). Al revisarlo aparece un segundo fallo que pylint no detecta.

**1a. `pygad` no está instalado.** No figura en `requirements.txt`, ni en este repo ni en el de Bitbucket (`itacyl-pan-modelsinference-svc`). El manifest `inbox/a35/manifest.yaml` sí lo declara (`pygad==3.5.0`). El import es local al método, así que el plugin arranca y el modo inline funciona: solo falla al pedir `optimize`.

**1b. `ctx` se sobrescribe antes de usarse.** El parámetro `ctx` (tupla `model, scaler_X, scaler_y`) se reasigna a un dict de 4 features en la línea 201. Después, `model, scaler_X, scaler_y = ctx` desempaqueta ese dict. Aunque se instale `pygad`, falla con `ValueError: too many values to unpack (expected 3)`.

Reproducido llamando a `_run_optimize` con las features del test:

```
1) ModuleNotFoundError No module named 'pygad'
2) con pygad: ValueError too many values to unpack (expected 3)
```

**Origen y alcance:**

- El fallo 1b entró en `aebc62d` (2026-10-07, "persist retraining only in MLflow, isolate user models…"). Está igual en `origin/develop` de Bitbucket.
- `outputs/a35/verification_report.md` dice "modo optimize: OK (GA ejecuta y devuelve setpoints + ahorro)". Es anterior a ese commit y está **desactualizado**.
- `tests/unit/test_ml35_dairy_ann_cleaning_cost.py::test_predict_optimize` pasa porque usa el plugin falso de `tests/conftest.py`, no el real. Ningún test ejecuta el GA de ml35.

**Arreglo propuesto (pendiente):**

- Usar otro nombre para el dict de features (p. ej. `feats`) y desempaquetar `ctx` al principio del método.
- Añadir `pygad==3.5.0` a `requirements.txt`, en los dos repos.
- Añadir un test que ejecute `_run_optimize` real con pocas generaciones.
- Volver a verificar ml35 y actualizar su informe.

---

## 2. Errores revisados que son falsos positivos

| Mensaje | Nº | Dónde | Por qué es falso positivo |
|---|---:|---|---|
| `no-member` `cv2.*` (`imread`, `resize`, `cvtColor`, `VideoCapture`…) | 30 | ml1, ml2, ml4, ml5 | OpenCV es una extensión C y pylint no ve sus miembros. Se resuelve con `extension-pkg-allow-list=cv2`. |
| `import-error` / `no-name-in-module` / `no-member` `tensorflow.keras` | 15 | ml3, ml18 | TF 2.21 carga `tf.keras` de forma perezosa. Se ha comprobado que `from tensorflow.keras.{models,layers,optimizers,callbacks} import …` funciona en el venv. |
| `no-member` `PIL.Image.LANCZOS` | 5 | ml2, ml4, ml7, ml8, ml10 | Existe en Pillow 12.3 (comprobado con `hasattr`). |
| `no-member` `numpy.random.RandomState` | 1 | ml16 `training.py:202` | Existe en numpy 2.5. |
| `not-callable` `F.softplus` / `F.one_hot` | 2 | ml46 `_vendor/model_arch.py:129`, test | Funciones de PyTorch definidas en C. |
| `unsubscriptable-object` `best` | 1 | ml46 `preprocessing.py:79` | `best` es `None` o una tupla, y se comprueba `is None` antes de indexar. |
| `no-member` `FieldInfo` (`version`, `mlflow_run_id`…) | 21 | tests ml17-19, 21, 23, 26 | Pylint confunde los campos de modelos Pydantic con su `Field(...)`. |
| `unsubscriptable-object`, `not-an-iterable`, `unsupported-membership-test`, `unsupported-assignment-operation` | 24 | tests ml10, 15, 17-19, 21, 23, 26, 47, 48 | Mismo origen: campos Pydantic opcionales (`None` por defecto) o valores devueltos por funciones con varias ramas de retorno. |
| `no-member` `DatetimeIndex.month` | 5 | tests `base_reload_per_model`, `ml17_base_promotion` | El atributo existe; pylint no infiere pandas bien. |
| `raising-bad-type` | 2 | `tests/conftest.py:486, 500` | `raise exc` solo se ejecuta tras comprobar `is not None`. |

---

## 3. Avisos con algo de sustancia (sin fallo funcional hoy)

**Código de la aplicación:**

| Aviso | Dónde | Comentario |
|---|---|---|
| `consider-using-with` | ml43 `plugin.py:503` | `pickle.load(open(p, "rb"))` no cierra el fichero: deja un descriptor abierto hasta el GC. Afecta solo al entrenamiento. |
| `consider-using-with` | ml5 `plugin.py:90` | `NamedTemporaryFile(delete=False)` se cierra justo después. Es correcto, solo estilo. |
| `expression-not-assigned` | ml47 `plugin.py:134` | `self._require_loaded() if mlflow_ctx is None else None` funciona, pero es un `if` escrito como expresión. |
| `unused-variable` | ml45 `plugin.py:207` (`entity_ids`), ml41 `audio_mae.py:73`, ml43 / ml45 `_vendor` (`B`), ml46 `_vendor/evaluation.py:281` | Valores desempaquetados que no se usan. En `_vendor` es código original del equipo de IA. |
| `reimported` / `redefined-outer-name` | ml21 `plugin.py:516` (`datetime`, `timezone`), ml25 `plugin.py:166, 341` (`os`), `app/registry.py:90` (`TrainResponse`), `main.py:32` (`app`) | Imports repetidos dentro de funciones. Inofensivo. |
| `unspecified-encoding` | ml10 (`mlflow_utils.py:40`, `model_loader.py:76`, `plugin.py:634`), ml25 (`mlflow_utils.py:45`, `plugin.py:523`) | `open()` sin `encoding`. En el contenedor es UTF-8; conviene ponerlo explícito. |
| `raise-missing-from` | ml43 / ml45 `_vendor/xai/temporal_explainer.py` | Re-lanzar `ImportError` sin `from exc` pierde la traza original. |
| `comparison-with-itself` | ml43 `plugin.py:517, 551`, ml45 `plugin.py:572` | `test_auc == test_auc` es el truco para detectar NaN. Funciona; sería más claro `math.isnan`. |
| `logging-fstring-interpolation` | ml19 `_vendor/training_core.py` (10), ml45 `_vendor/model.py` (3) | f-strings en `logger.*`. Código vendorizado. |
| `broad-exception-caught` | 38 en `app/` (más en ml8: 5, `app/domain`: 4, ml4: 4, ml10: 3) | `except Exception`. En su mayoría son fallbacks deliberados (XAI, subida a MLflow); no se ha visto ninguno que oculte un error de predicción. |
| `useless-return` | `mlflow_utils.py` de ml28, ml31 y ml33 | `return None` final en las funciones de los modelos sin reentrenamiento. |
| `no-else-raise`, `chained-comparison` | ml3, ml7, ml40, ml41 | Estilo. |

**Tests:**

| Aviso | Dónde | Comentario |
|---|---|---|
| `unknown-option-value` | `test_artifact_store_versions.py:155` | Un comentario en la misma línea que `# pylint: disable=` se lee como nombres de regla (`simulates`, `restart`). El `disable` no tiene efecto. |
| `dangerous-default-value` | `test_ml47_training.py:21`, `test_ml48_training.py:25`, `test_ml21_base_promotion.py:52` | Listas globales como argumento por defecto. Solo se leen, así que no hay efecto real. |
| `arguments-differ`, `abstract-method` | `test_infrastructure.py:36-58`, `test_ml6_…py:191` | Plugins y módulos falsos con firma reducida. Aceptable en tests. |
| `redefined-outer-name` (212), `protected-access` (80), `reimported` (43) | Toda la carpeta | Patrón normal de fixtures de pytest y de tests que inspeccionan internos. |

---

## 4. Estilo, complejidad y duplicación (volumen)

| Mensaje | Nº | Comentario |
|---|---:|---|
| `line-too-long` | 1976 | Límite de pylint 100 frente a 120 en `.flake8`. Solo 282 pasan de 120 (máx. 253). |
| `missing-function-docstring` / `-class-` / `-module-` | 884 / 79 / 36 | Sobre todo funciones de tests. |
| `import-outside-toplevel` | 431 | Mayoritariamente intencionado: imports pesados (torch, TF, shap) cargados solo al usarse. |
| `duplicate-code` | 389 | Bloques repetidos entre plugins hermanos (`mlflow_utils.py`, `model_loader.py`, ml43/45/47/48), inherente a "1 plugin autocontenido por modelo". |
| `too-many-locals` / `-arguments` / `-positional-arguments` / `-statements` / `-branches` / `-instance-attributes` | 157 / 63 / 54 / 25 / 24 / 33 | Funciones de entrenamiento y predicción largas. |
| `invalid-name` | 168 | `X`, `X_train`, `B`… nomenclatura habitual en ML. |
| `unused-argument` | 331 | Firmas impuestas por `ModelPluginPort` y por callbacks (p. ej. `ga_instance`, `solution_idx` de pygad). |
| `too-few-public-methods`, `consider-using-from-import`, `wrong-import-order`, otros | ~120 | Estilo. |

---

## 5. Recomendaciones

1. **Corregir ml35 `optimize`** (sección 1) y re-verificarlo. Es lo único con impacto en producción.
2. **Añadir un `.pylintrc`** alineado con el repo, para que el informe muestre solo lo útil:
   - `max-line-length=120` (como flake8);
   - `extension-pkg-allow-list=cv2`;
   - `ignored-modules=tensorflow.keras`;
   - `load-plugins=pylint_pydantic` (o `generated-members` para los campos Pydantic);
   - desactivar `missing-docstring`, `import-outside-toplevel`, `duplicate-code` e `invalid-name` en `tests/` y `_vendor/`.
3. **Meter lint en el pipeline**: hoy el paso de Bitbucket es un `echo`. Basta con que flake8 y pylint con `--disable=all --enable=E,W` y la configuración anterior bloqueen los errores.
4. Limpiezas menores cuando se toquen esos plugins:
   - el `open()` sin cerrar de ml43;
   - el `if` como expresión de ml47;
   - el `disable` roto de `test_artifact_store_versions.py`;
   - los `encoding` de ml10 y ml25.

El JSON completo de pylint (5379 mensajes) se ha guardado fuera del repo; se regenera con el comando del principio (~2 min).

---

## Estado tras la corrección — 2026-11-05

- ml35 `optimize` corregido (sección 1): `ctx` ya no se sobrescribe, `pygad==3.5.0` en
  `requirements.txt` y test con el plugin real contra la salida del GA del equipo de IA.
- Resto de mensajes corregidos en todo el código salvo `inbox/` y `outputs/` (incluido
  `_vendor/`), con un `.pylintrc` mínimo: excluye esas carpetas, desactiva `duplicate-code`
  (plugins independientes por diseño) y permite la extensión C `cv2`. No relaja ninguna otra regla.
- Resultado: **pylint 0 mensajes (10.00/10)** en `app/`, `main.py`, `tests/`, `deployment/`,
  `setup.py`, `data/` y `.claude/`; flake8 0; **1349 tests pasan** (los mismos que antes).
- ~4300 mensajes corregidos en código (líneas, docstrings, imports, nombres locales, fixtures…) y
  ~860 desactivados en línea con motivo: sobre todo métricas de tamaño (`too-many-*`) en lógica de
  modelos que no se refactoriza por fidelidad, imports pesados que se mantienen perezosos, y falsos
  positivos (cv2, `tf.keras`, campos Pydantic, pandas).
