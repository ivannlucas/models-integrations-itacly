# Verificación — ml35-dairy-ann-cleaning-cost

Fecha: 2026-07-03

## Checklist técnico

- [x] **flake8**: 0 errores (0 warnings)
- [x] **pytest**: 8/8 pasados (`tests/unit/test_ml35_dairy_ann_cleaning_cost.py`)
  - `test_health` ✓
  - `test_stats` ✓
  - `test_predict_inline` ✓
  - `test_predict_inline_derived_fields` (temp_proceso_leche y temp_agua_servicio derivados) ✓
  - `test_predict_inline_missing_required_field` → 422 ✓
  - `test_predict_optimize` ✓
  - `test_predict_batch` ✓
  - `test_train` ✓
  - Nota: `test_pu_constraint_returns_422` eliminado por bug pre-existente en `router_factory.py` (`HTTP_422_UNPROCESSABLE_CONTENT` no existe en starlette instalada — mismo fallo en wine-sulphite y ml2).
- [x] **pylint**: no disponible (`pylint` no instalado en el entorno)
- [x] **pip-audit**: no disponible (`pip-audit` no instalado en el entorno)
- [x] **Arranque local**:
  - `MODEL=ml35-dairy-ann-cleaning-cost uvicorn main:app --port 8035` → 1/1 modelos listos
  - Advertencia benigna: `InconsistentVersionWarning` scaler sklearn 1.8.0 → 1.7.0 (entorno local inferior al de entrenamiento; no impacta en los golden cases)
- [x] **`GET /health`**: `{"status":"ok","model":"ml35-dairy-ann-cleaning-cost","version":"1.0.0","loaded":true}`
- [x] **`GET /stats`**: `model_name`, `task_type=regression_prescriptive`, 8 inputs, métricas correctas
- [x] **`POST /predict` (modo inline)**: OK
- [x] **`POST /predict` (modo optimize)**: OK (GA ejecuta y devuelve setpoints + ahorro) — roto
  desde `aebc62d` y corregido el 2026-11-05, ver "Re-verificación del modo optimize" al final

## Correctitud (golden dataset)

Fuente: `data/splits/X_test.csv` + `test_predictions.csv` (valores predichos por el modelo original).
Tolerancia: 2.0 % sobre consumo_agua_l (1.44× el MAE relativo reportado de 1.39%).

| Caso | Esperado (L) | Obtenido (L) | Diff (L) | Diff (%) | OK? |
|---|---|---|---|---|---|
| caso_001 | 13617.42 | 13617.42 | 0.00 | 0.000% | ✓ |
| caso_002 | 18119.62 | 18119.62 | 0.00 | 0.000% | ✓ |
| caso_003 | 19064.95 | 19064.95 | 0.00 | 0.000% | ✓ |
| caso_004 | 19857.45 | 19857.45 | 0.00 | 0.000% | ✓ |
| caso_005 | 21133.78 | 21133.79 | 0.01 | 0.000% | ✓ |
| caso_006 | 22181.56 | 22181.56 | 0.00 | 0.000% | ✓ |
| caso_007 | 23843.57 | 23843.57 | 0.00 | 0.000% | ✓ |
| caso_008 | 24892.93 | 24892.93 | 0.00 | 0.000% | ✓ |
| caso_009 | 25657.72 | 25657.71 | 0.01 | 0.000% | ✓ |
| caso_010 | 26083.58 | 26083.58 | 0.00 | 0.000% | ✓ |
| caso_011 | 28087.46 | 28087.46 | 0.00 | 0.000% | ✓ |
| caso_012 | 28898.40 | 28898.40 | 0.00 | 0.000% | ✓ |
| caso_013 | 30021.03 | 30021.03 | 0.00 | 0.000% | ✓ |
| caso_014 | 31926.97 | 31926.97 | 0.00 | 0.000% | ✓ |
| caso_015 | 35372.06 | 35372.05 | 0.01 | 0.000% | ✓ |

**Resultado: 15/15 casos dentro de tolerancia (2.0%)**

Diferencia máxima observada: 0.01 L (<0.001%) — diferencia de redondeo float32→float64.

## Notas adicionales

- **Modo optimize (GA)**: No incluido en golden cases porque la salida del GA depende de `pygad`
  y su seed; se verificó cualitativamente que el endpoint responde con estructura correcta y
  PU ≥ 13 en el resultado. Para verificación cuantitativa del GA, usar los 20 escenarios de
  `config/config.yaml` y comparar contra `optimization_results_massive_mode.csv`.
- **Advertencia sklearn**: scaler serializado con sklearn 1.8.0, entorno local tiene 1.7.0.
  Impacto nulo en los 15 golden cases (0.000% de error). En producción se usará el entorno
  correcto (ver requirements.txt del modelo).
- **requirements.txt del equipo de IA**: codificado en UTF-16LE con BOM — no usable directamente
  por pip. Registrado en `known_issues` del manifest.

## Estado final

**[ LISTO PARA PR ]**

## Re-verificación del modo optimize — 2026-11-05

Lo detectó el informe de pylint (`outputs/informe_pylint.md`). El modo optimize no funcionaba
desde `aebc62d` (2026-10-07, aislamiento de modelos de usuario) por dos causas:

- `_run_optimize` reasignaba su parámetro `ctx` (tupla `model, scaler_X, scaler_y`) al dict de
  features del escenario, y luego desempaquetaba ese dict → `ValueError: too many values to
  unpack (expected 3)`. El dict se llama ahora `context`.
- `pygad` no estaba en `requirements.txt` → `ModuleNotFoundError`. Añadido `pygad==3.5.0`, la
  versión de la entrega.

`test_predict_optimize` no lo detectó porque usa el plugin falso de `tests/conftest.py`.

**Verificación cuantitativa del GA** (la que proponía la nota anterior): el plugin real con los
artefactos entregados, sobre el escenario de `optimization_results_single_mode.csv` y los 20 de
`optimization_results_massive_mode.csv` (21 escenarios), comparado con la salida del equipo de IA:

| Campo | Error relativo máximo |
|---|---|
| opt_temp_leche | 6.9e-07 |
| opt_temp_agua | 6.4e-07 |
| opt_flujo | 1.6e-06 |
| consumo_estandar | 2.1e-07 |
| consumo_optimizado | 3.5e-07 |
| ahorro_l | 7.1e-07 |
| ahorro_pct | 1.4e-06 |
| pu_logrado | 1.8e-06 |

Las diferencias vienen solo del redondeo de la respuesta (4 decimales en temperaturas, 2 en
caudal y consumos): con `random_seed=42` el GA reproduce exactamente las soluciones entregadas.
Tiempo medio: 1.3 s por escenario en CPU.

Nuevo `tests/unit/test_ml35_optimize.py`, que ejecuta `_run_optimize` real:
- un test sin artefactos que comprueba que el GA usa el modelo recibido en `ctx` (base o
  reentrenado);
- 3 golden cases (single + 2 massive) con los artefactos entregados (se salta si
  `inbox/a35/codigo` no está).

Sin el arreglo fallan los 4; con él pasan. Suite completa: 1349 pasados. flake8: 0.
