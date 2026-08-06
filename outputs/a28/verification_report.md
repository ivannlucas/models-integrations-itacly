# Verificación — ml28-meat-neuroevolutionary-raw-materials-prediction (a28)

## Checklist técnico

- [x] flake8: 0 errores (`app/plugins/ml28_meat_neuroevolutionary_raw_materials_prediction/`,
      `app/registry.py`, `tests/conftest.py`,
      `tests/unit/test_ml28_meat_neuroevolutionary_raw_materials_prediction.py`)
- [x] pytest: 340/340 passed (suite completa del repo). El test dedicado
      (`test_ml28_meat_neuroevolutionary_raw_materials_prediction.py`, 7 casos) usa `FakePlugin`
      — valida wiring HTTP (health/stats/predict inline+batch/mapeo de validación a 422/`/train`
      → 501), no correctitud. La correctitud contra el golden dataset se valida en la Parte B,
      con el plugin real servido por HTTP.
- [x] pylint: `app/plugins/ml28_meat_neuroevolutionary_raw_materials_prediction/` puntúa 8.17/10.
      Sin issues bloqueantes; los avisos son todos `duplicate-code` (R0801) por la lista de 11
      columnas de entrada repetida entre `constants.py`, `_vendor/features.py`,
      `_vendor/input_validation.py` y `_vendor/output_writer.py` — inherente a vendorizar
      verbatim varios ficheros del código original que comparten ese contrato de columnas.
      Mismo orden de magnitud que el precedente ya aceptado en este repo
      (`ml46_dairy_fouling_clog_detection` 8.14/10, `ml45_cereals_dnsl_critical_point_detection` 8.79/10).
- [x] pip-audit: sin CVEs nuevas atribuibles a este cambio. Este plugin no añade ninguna
      dependencia nueva a `requirements.txt` (solo pandas/numpy, ya presentes). Las 4
      vulnerabilidades que reporta `pip-audit -r requirements.txt` (torch 2.11.0, setuptools
      81.0.0 x2, cryptography 49.0.0) son preexistentes del entorno, ajenas a este modelo.
- [x] Arranque local + health + predict + stats: OK. Servido con
      `CUDA_VISIBLE_DEVICES="" STORAGE_BUCKET="" MODEL=ml28-meat-neuroevolutionary-raw-materials-prediction
      uvicorn main:app`. `/health` → `loaded: true` (no hay artefacto que descargar de S3 — ver
      known_issues del manifest). `/stats` → metadata + métricas reales de policy_simulation.
      `/predict` (inline y batch) → reproduce exactamente `recommendations.csv` del código
      original (ver Parte B).
- [x] /train: **501 esperado** — `manifest.training.supported=false`, comportamiento correcto
      (no es un fallo del checklist). Mensaje de error explica el motivo real: el motor servido
      no tiene pesos que ajustar.

## Correctitud (golden dataset)

Fuente: los 5 `golden_cases` de `inbox/a28/manifest.yaml`, filas reales de
`data/demo/customer_upload_example.csv` (CSV de ejemplo oficial del equipo de IA), ejecutadas
contra el código entregado tal cual (`src/cli/platform_run.py::run_platform_pipeline`). Verificado
contra el **plugin real** servido por HTTP (`POST /predict`, `mode=inline` para los 5 casos —
cada uno es una fila independiente, sin dependencia entre filas, confirmado en manifest-extraction
ejecutando el pipeline original sobre una fila sola vs. la misma fila dentro de un batch de 20).

| Caso | Esperado (proba, flag, order_qty, risk) | Obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|
| case_001_do_not_buy_low | 0.2658, 0, 0.000, LOW | 0.2658, 0, 0.000, LOW | 0 | ✅ |
| case_002_buy_high_risk | 0.8588, 1, 29.817, HIGH | 0.8588, 1, 29.817, HIGH | 0 | ✅ |
| case_003_buy_medium_risk | 0.6681, 1, 20.379, MEDIUM | 0.6681, 1, 20.379, MEDIUM | 0 | ✅ |
| case_004_buy_high_risk_near_zero_stock | 0.9241, 1, 44.183, HIGH | 0.9241, 1, 44.183, HIGH | 0 | ✅ |
| case_005_gating_forces_zero_despite_positive_raw_recommendation | 0.3883, 0, 0.000, LOW (raw rec=1.638) | 0.3883, 0, 0.000, LOW (raw rec=1.638) | 0 | ✅ |

Tolerancia usada: `exact_match: true` (definida en `manifest.golden_cases[*].tolerance`, no un
default con margen) — el motor servido es aritmética determinista sobre `config/platform_config.yaml`,
no un modelo probabilístico, así que no aplica ningún margen de tolerancia numérica.

Resultado: **5/5 casos dentro de tolerancia** (coincidencia exacta en los 4 campos comparados por
caso, incluidos los 3 campos de negocio derivados — `purchase_trigger_proba`, `order_quantity_tons`,
`risk_level` — y el flag de decisión).

Nota sobre `case_005`: este caso verifica específicamente el gating
(`purchase_trigger_flag=0` fuerza `order_quantity_tons=0.0` aunque
`quantity_optimizer_recommendation_tons=1.638 > 0`). Si el plugin hubiera devuelto 1.638 en vez
de 0.0 aquí, habría sido un bug de wiring real (gating no aplicado) — el caso pasó correctamente.

## Estado final

**LISTO PARA PR** — checklist técnico en verde, 5/5 golden cases reproducidos exactamente contra
el plugin real servido por HTTP (inline), y el batch completo de 20 filas del CSV demo también
verificado manualmente contra la ejecución directa del código original (`row_count=20,
triggered_orders=12, aggregate_excess_reduction_pct=20.959`, idéntico en ambos). Puntos a
destacar para el revisor humano:

1. **El nombre del paquete es engañoso y ya está documentado como tal.** "neuroevolutionary" no
   refleja lo que se sirve: no hay ningún modelo entrenado cargado (ni la comparativa
   neuroevolutiva ni el baseline oficial LinearRegression/Ridge de la ruta `mixed_context`). El
   motor servido es el motor de reglas determinista de `src/cli/platform_run.py`, verbatim.
   Decisión confirmada explícitamente por el usuario durante `plugin-integration` — ver
   `inbox/a28/manifest.yaml` (aviso al inicio de la sección `artifacts`).
2. **`training.supported=false` con motivo real**, no un hueco sin rellenar: el motor servido no
   tiene pesos; el pipeline ML real de la entrega (`mixed_context`) opera sobre columnas objetivo
   sintéticas ajenas al contrato de inferencia del cliente y entrena artefactos que este plugin
   nunca usa.
3. **Sí hay `predict_inline` genuino de una fila** (a diferencia de a45/m47): verificado que el
   cálculo no tiene dependencia entre filas, así que cada fila es una predicción independiente y
   válida por sí sola.
4. **Memoria real disponible** (a diferencia de a45): "28 - Sistema de soporte a la decision para
   el aprovisionamiento de materia prima carnica v2.0.2.docx" — usada para Introducción y
   Objetivos del manifest; secciones de resultados/limitaciones adicionales no explotadas en
   detalle, revisar si `docs-generation` necesita más profundidad.
5. **`mlflow_utils.py` es un no-op documentado**: no hay artefacto de usuario que descargar de
   MLflow para este modelo (mismo motivo que el punto 2) — el fichero existe por convención del
   repo, no por necesidad funcional.
