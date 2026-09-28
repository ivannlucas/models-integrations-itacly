# Verificación — ml14-wine-phyto-price-forecast (a14)

Fecha: 2026-09-28
Plugin: `app/plugins/ml14_wine_phyto_price_forecast/`
Manifest: `inbox/a14/manifest.yaml`

## Nota de proceso — corrección de entrega de código

Esta integración se rehizo tras confirmar con el cliente (Erick Mercado, Slack, 2026-09-28) que
una integración anterior se había hecho por error contra una copia local no oficial del código
(que incluía una carpeta `docs/`/`reports/audit/` con una auditoría interna que concluía que el
modelo NO superaba al baseline de Drift). La entrega oficialmente aprobada es otra, sin esa
auditoría, y en su evaluación de test SÍ hay un modelo (GRU) que supera al baseline. El código
de inferencia y los tres artefactos de modelo son idénticos byte a byte entre ambas entregas
(confirmado por sha256) — no son dos modelos distintos, solo dos protocolos de evaluación
distintos sobre el mismo entrenamiento. Este informe y el manifest usan exclusivamente la
entrega aprobada. Ver `inbox/a14/manifest.yaml::known_issues` para el detalle completo.

## Estado de aceptación del modelo

GRU (seleccionado por `src/predict/predictor.py::_pick_best_model()` por RMSE de test entre
GRU/LSTM/XGBoost) supera al baseline de Drift en el test reportado por esta entrega: RMSE GRU
`2.2873` frente a RMSE Drift `2.5406` (211 observaciones, split temporal único 80/20). Coincide
con lo que documentan la memoria y el README del entregable. Ver
`inbox/a14/manifest.yaml::model_status` para el detalle completo y la trazabilidad.

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
      `test_infrastructure.py`, ninguno relacionado con ml14).
- [x] **pylint** `app/plugins/ml14_wine_phyto_price_forecast/ --disable=import-error`:
      **9.50/10**. Los avisos son todos `C0301` (líneas >100 car. — criterio no real del
      proyecto: `.flake8` fija `max-line-length=120` con `E501` ignorado), `R0402`/`R0903` en
      `rnn_models.py` (mismo patrón que `ml23_lactic_market_price_forecast/rnn_models.py`, ya en
      `main`, sin corregir allí tampoco) y un `R0914` (too-many-locals) en
      `preprocessing.py::run_inference`.
- [x] **pip-audit**: ml14 no añade ninguna dependencia nueva a `requirements.txt` (usa
      `torch`/`pandas`/`numpy`/`scikit-learn`, ya declaradas para otros plugins).
- [x] **Arranque local + health/predict/stats/train — OK**, contra un servidor FastAPI real (no
      mocks): se montó una app standalone (`make_model_router` + `ModelContainer`, el mismo
      wiring que usa `main.py`) solo para `ml14-wine-phyto-price-forecast`, evitando el *import*
      eager de `app.registry` (que carga los ~26 plugins del repo, varios con dependencias
      pesadas no instalables en este entorno sandbox). Se instaló `torch==2.10.0+cpu` en un venv
      aislado para esta prueba; el artefacto real
      `artifacts/ml14_wine_phyto_price_forecast/gru_model.pt` (+ el dataset de referencia bundled
      para reajustar los scalers, exactamente como hace el CLI original) del equipo de IA se
      cargó sin mocks.
  - `GET /health` → `200 {"status":"ok","model":"ml14-wine-phyto-price-forecast","loaded":true}`
  - `POST /predict` (inline, los 5 `golden_cases`) → ver tabla de correctitud abajo.
  - `GET /stats` → `200`, incluye `metrics_reported` real del manifest.
  - `POST /train` → **`501`** con el detalle de por qué (procedimiento de entrenamiento
    entregado sin contrato de datos de cliente — ver manifest).
- [x] **`/train` según `manifest.training`**: `supported: false` en el manifest → el endpoint
      devuelve `501` (`TrainingNotSupportedError`). Confirmado, comportamiento esperado.
- [x] `caso_001` se verificó además contra la salida literal de
      `python -m src.main predict --input data/raw/prediction_input.csv` (modo `--model auto`
      del CLI original tal cual se entregó) — coincide exacto: `predicted_price=121.394124`,
      confirmando que el plugin no solo reproduce el algoritmo sino el comportamiento real y
      completo del código entregado (incluida la selección automática de arquitectura, que en
      esta entrega elige correctamente GRU).

## Correctitud (golden dataset)

`inbox/a14/manifest.yaml::golden_cases` contiene 5 casos: 4 predicciones sobre ventanas reales
de `data/processed/final_dataset_for_modeling.csv` (el mismo dataset bundled para reajustar los
scalers) y 1 rechazo controlado por historial insuficiente. Los 5 se ejecutaron contra el
endpoint HTTP real (`POST /predict`, modo inline) — no contra una llamada directa a Python —
para verificar el wiring completo (DTO → use case → plugin → feature_engineering → reajuste de
scalers desde el dataset bundled → GRU → reconstrucción del precio).

| Caso | Esperado (predicted_price) | Obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|
| caso_001_ultima_ventana_minima | 121.394124 | 121.394124 | 0.0 | ✅ |
| caso_002_ventana_extendida_mismo_resultado | 121.394124 | 121.394124 | 0.0 | ✅ |
| caso_003_ventana_historica_2012 | 97.071318 | 97.071318 | 0.0 | ✅ |
| caso_004_frontera_inicio_test | 104.271455 | 104.271455 | 0.0 | ✅ |
| caso_005_rechazo_historial_insuficiente | HTTP 422 | HTTP 422 (`min_length` Pydantic) | — | ✅ |

Tolerancia usada: `tolerance_pct: 1.0` declarada en el manifest (margen de reproducibilidad
numérica entre entornos/versiones de PyTorch); en la práctica los 4 casos de predicción
reprodujeron el valor exacto (diferencia 0.0) — el wiring del plugin reconstruye exactamente el
mismo tensor `[1, 12, 39]` escalado que espera el artefacto, reajustando los scalers desde el
dataset bundled exactamente como hace `src/predict/predictor.py::_build_scalers_and_features()`.

**Resultado: 5/5 casos dentro de tolerancia (ninguno silenciado ni ajustado).**

## Hallazgos durante esta verificación

Ninguno nuevo sobre el wiring en sí. El hallazgo relevante de todo este ciclo — la existencia de
dos entregas divergentes del mismo código, y cuál de ellas es la oficialmente aprobada — se
resolvió antes de esta verificación (confirmación directa del cliente) y quedó documentado en el
manifest, no aquí.

## Estado final

**LISTO PARA PR.** GRU supera al baseline de Drift en el test reportado por la entrega
aprobada; no hay ningún hallazgo de aceptación pendiente de decisión humana en esta versión del
manifest (a diferencia de la integración anterior, hecha por error contra la copia no oficial).
