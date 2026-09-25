# Verificación — ml14-wine-phyto-price-forecast (a14)

Fecha: 2026-09-25
Plugin: `app/plugins/ml14_wine_phyto_price_forecast/`
Manifest: `inbox/a14/manifest.yaml`

## Estado de aceptación del modelo — leer antes de lo demás

El LSTM 2.0.0 servido por este plugin es el artefacto real y auditado por el equipo de IA
(`docs/MODEL_CARD.md`), seleccionado por RMSE de validación entre LSTM/GRU/XGBoost. En el test
final, evaluado una única vez con el protocolo ya corregido (sin fuga de datos entre
arquitectura/horizonte y test), **no supera al baseline de Drift**: RMSE LSTM `3.0168` frente a
RMSE Drift `2.5406` (mejora relativa `-18.75%`). Este dato no se oculta ni se ajusta la
tolerancia para disimularlo — ver `inbox/a14/manifest.yaml::model_status` para el detalle
completo, la trazabilidad a las fuentes y la decisión de integrar igualmente el artefacto real
(no el GRU/protocolo retractado de la memoria/README) a la espera de revisión humana antes de
abrir la PR.

Durante esta verificación se encontraron y documentaron, además, **5 inconsistencias
reproducibles** en el código entregado (hashes de artefactos que no coinciden con lo declarado,
un bug de auto-selección de modelo, y un bug de reajuste de scalers en caliente que hace que el
propio CLI original no sea reproducible frente a sus propios artefactos congelados) — todas
detalladas en `inbox/a14/manifest.yaml::known_issues`, con el razonamiento de qué decisión tomó
este plugin en cada caso y por qué.

## Checklist técnico

- [x] **flake8** (`app/plugins/ml14_wine_phyto_price_forecast/`, `tests/unit/test_ml14_*.py`,
      `app/registry.py`, `tests/conftest.py`): **0 errores** (config del proyecto: `.flake8`,
      `max-line-length=120`, `E501` ignorado explícitamente).
- [x] **pytest** (`tests/unit/`): **7/7 passed** (tests propios de ml14: health, stats, predict
      inline, rechazo por `min_length` a nivel de DTO, `InsufficientDataError` → 422, predict
      batch, `train` → 501). Suite completa del repo: **351/364 passed**, sin regresiones
      atribuibles a este cambio — los 13 fallos son preexistentes por
      `ModuleNotFoundError: torch` (`torch` no está instalado en este entorno sandbox; afecta a
      `test_ml46_dairy_fouling_clog_detection.py`, `test_modelo10_lacteo_unit.py` y parte de
      `test_infrastructure.py`, ninguno relacionado con ml14). Confirmado con una segunda
      instalación aislada de `torch==2.10.0+cpu` (ver más abajo) que la suite propia de ml14 no
      depende de este problema de entorno.
- [x] **pylint** `app/plugins/ml14_wine_phyto_price_forecast/ --disable=import-error`:
      **9.38/10**. Los avisos son todos `C0301` (líneas >100 car. — criterio no real del
      proyecto: `.flake8` fija `max-line-length=120` con `E501` ignorado), `R0402`/`R0903` en
      `rnn_models.py` (mismo patrón que `ml23_lactic_market_price_forecast/rnn_models.py`, ya en
      `main`, sin corregir allí tampoco) y un `R0914` (too-many-locals) en
      `preprocessing.py::run_inference` — comparado explícitamente contra
      `ml23_lactic_market_price_forecast` (9.66/10, mismo tipo de modelo: forecast RNN PyTorch),
      sin categorías de aviso nuevas.
- [x] **pip-audit**: ml14 no añade ninguna dependencia nueva a `requirements.txt` (usa
      `torch`/`pandas`/`numpy`, ya declaradas para otros plugins; el escalado se hace a mano con
      los `mean`/`scale` ya serializados en `preprocessing.json`, sin depender de
      `scikit-learn` en tiempo de inferencia). No hay superficie nueva que auditar.
- [x] **Arranque local + health/predict/stats/train — OK**, contra un servidor FastAPI real (no
      mocks): se montó una app standalone (`make_model_router` + `ModelContainer`, el mismo
      wiring que usa `main.py`) solo para `ml14-wine-phyto-price-forecast`, evitando el *import*
      eager de `app.registry` (que carga los ~26 plugins del repo, varios con dependencias
      pesadas -- detectron2/opencv/etc. -- no instalables en este entorno sandbox). Se instaló
      `torch==2.10.0+cpu` en un venv aislado para esta prueba; el artefacto real
      `artifacts/ml14_wine_phyto_price_forecast/lstm_model.pt` (+ `preprocessing.json` +
      `lstm_features.json`) del equipo de IA se cargó sin mocks.
  - `GET /health` → `200 {"status":"ok","model":"ml14-wine-phyto-price-forecast","loaded":true}`
  - `POST /predict` (inline, los 5 `golden_cases`) → ver tabla de correctitud abajo.
  - `GET /stats` → `200`, incluye `metrics_reported` real del manifest (RMSE test final LSTM
    vs. Drift, `accepted_for_production: false`).
  - `POST /train` → **`501`** con el detalle de por qué (procedimiento de entrenamiento
    entregado sin contrato de datos de cliente — ver manifest).
- [x] **`/train` según `manifest.training`**: `supported: false` en el manifest → el endpoint
      devuelve `501` (`TrainingNotSupportedError`). Confirmado, comportamiento esperado.
- [x] Adicionalmente, se verificó por separado (sin pasar por HTTP) que
      `PredictInlineRequest(mode="inline", rows=[...]).model_dump(exclude={"mode","model_key",
      "threshold","mlflow_run_id","data_path"})` produce un diccionario **plano** con `rows`
      como clave de primer nivel — el mismo que `PredictModelUseCase.execute()` pasa a
      `plugin.predict_inline(features=...)`. Esto descarta explícitamente la clase de bug
      encontrada durante la verificación de ml15 (un campo `features: dict` anidado que
      `model_dump()` duplicaba) — aquí el campo se llama `rows`, no `features`, por diseño.

## Correctitud (golden dataset)

`inbox/a14/manifest.yaml::golden_cases` contiene 5 casos: 4 predicciones sobre ventanas reales
de `data/processed/final_dataset_for_modeling.csv` (mismo dataset con el que se entrenaron los
scalers/el modelo) y 1 rechazo controlado por historial insuficiente. Los 5 se ejecutaron contra
el endpoint HTTP real (`POST /predict`, modo inline) — no contra una llamada directa a Python —
para verificar el wiring completo (DTO → use case → plugin → feature_engineering → scaler →
LSTM → reconstrucción del precio).

| Caso | Esperado (predicted_price) | Obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|
| caso_001_ultima_ventana_minima | 120.03087 | 120.03087 | 0.0 | ✅ |
| caso_002_ventana_extendida_mismo_resultado | 120.03087 | 120.03087 | 0.0 | ✅ |
| caso_003_ventana_historica_2012 | 96.995947 | 96.995947 | 0.0 | ✅ |
| caso_004_frontera_inicio_test | 104.652653 | 104.652653 | 0.0 | ✅ |
| caso_005_rechazo_historial_insuficiente | HTTP 422 | HTTP 422 (`min_length` Pydantic) | — | ✅ |

Tolerancia usada: `tolerance_pct: 1.0` declarada en el manifest (margen de reproducibilidad
numérica entre entornos/versiones de PyTorch); en la práctica los 4 casos de predicción
reprodujeron el valor exacto (diferencia 0.0) — el wiring del plugin reconstruye exactamente el
mismo tensor `[1, 12, 39]` escalado que espera el artefacto, con los scalers congelados de
`preprocessing.json`.

**Resultado: 5/5 casos dentro de tolerancia (ninguno silenciado ni ajustado).**

### Nota importante sobre los valores esperados

Los valores de `predicted_price` en el manifest **no coinciden** con lo que da hoy en vivo
`python -m src.main predict --input ... --model LSTM` sobre el código tal como se entregó
(p. ej. para `caso_001`, el CLI da `120.227294`, no `120.03087`). Esto no es un error de este
plugin: es la **quinta inconsistencia** documentada en `inbox/a14/manifest.yaml::known_issues` —
`run_prediction()` del CLI original siempre reajusta (`fit`) los scalers desde
`data/processed/final_dataset_for_modeling.csv` en caliente, en vez de cargar los scalers ya
congelados de `preprocessing.json`, y ese CSV en disco no coincide con el dataset sobre el que
`preprocessing.json` declara haberse ajustado (tercera inconsistencia). Este plugin usa siempre
los scalers **congelados y versionados** de `preprocessing.json` — la fuente de verdad
autocontenida y reproducible, y la que la propia auditoría del equipo de IA declaró como
equivalente al entrenamiento original. El hallazgo se comprobó de forma reproducible
reconstruyendo ambos caminos manualmente antes de fijar el valor esperado en el manifest.

## Hallazgos durante esta verificación

Ningún hallazgo nuevo respecto al wiring del plugin en sí — el trabajo de investigación pesado
(las 5 inconsistencias del código entregado) se hizo e incorporó al manifest **durante**
`manifest-extraction`/`plugin-integration`, no se descubrió tarde aquí. Esta verificación
confirma, contra un servidor HTTP real y el artefacto real, que las decisiones tomadas
(scalers congelados, LSTM fijo sin replicar `--model auto`, `training.supported=false`) están
correctamente implementadas y son reproducibles.

## Estado final

**LISTO PARA PR** en cuanto a wiring técnico y correctitud numérica frente al golden dataset.
Pendiente exclusivamente de la revisión humana ya acordada sobre `model_status` (el LSTM no
supera a Drift en test final) antes de decidir si se abre la PR — ver
`inbox/a14/manifest.yaml::model_status.decision_de_integracion`.
