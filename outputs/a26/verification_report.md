# Verificación — ml26-wine-sulfite-gru-pso-forecast (a26)

**Fecha:** 2026-10-09
**Plugin:** `app/plugins/ml26_wine_sulfite_gru_pso_forecast/`
**Manifest:** `inbox/a26/manifest.yaml` (18 golden cases)
**Código de referencia:** `inbox/a26/codigo/` (commit 62862cc, "Actualiza modelo 26 a horizonte 72h e inferencia CSV"), memoria v2.5 (06/08/2026)
**Entorno:** WSL2, Python 3.12, torch 2.x CPU (el equipo de IA fija torch 2.9.1)

> Este es el primer informe de verificación de a26 en el repositorio. KI-10 y KI-11 del manifest citaban un
> `outputs/a26/verification_report.md` que no se llegó a commitear en ninguna rama.

## Hallazgo principal: `/train` no portaba el procedimiento del equipo de IA

El `train()` integrado hacía un **ajuste fino de los pesos servidos** y **conservaba la normalización servida**.
El equipo de IA nunca definió ese procedimiento. El original (`train_sequence_pso.py`) busca hiperparámetros con
PSO y luego entrena **un modelo nuevo desde cero** con `train_sequence_with_config`: semilla 7 y z-score de X e y
ajustado sobre el train. Además, KI-04 decía que el plugin usaba `train_sequence_with_config`, y no era así.

**Corregido:** `training.py` porta ahora `train_sequence_with_config` literalmente. Solo se omite la búsqueda
PSO, que reutiliza la configuración final, igual que en ml14 y ml23.

**Hallazgo del port:** las ventanas deben ser C-contiguas, como los `.npy` del original. Con una copia no
contigua de los mismos valores, la media float32 del z-score cambia en el último bit y, tras 25 épocas, el modelo
diverge (pesos con diferencia de hasta 0,1).

## Checklist técnico (Parte A)

- [x] **flake8** (repo completo): 0 errores.
- [x] **pylint** sobre `app/plugins/ml26_wine_sulfite_gru_pso_forecast/`: 10,00/10. El proyecto usa
      `max-line-length=120`, así que se descarta C0301.
- [x] **pytest** `tests/unit/`: 908 passed. Los tests de entrenamiento de ml26 se rehacen para el port: modelo
      nuevo, normalización del usuario, modelo servido intacto y ventanas contiguas.
- [x] **Artefactos:** `gru_pso.pkl` (sha256 03fd25b6…0b16) y `best_model.json` de `inbox/a26/codigo/models/artifacts/`.
      En S3 no existen (KI-13); para esta verificación se copiaron a `artifacts/` local.
- [x] **Servidor real** (`MODEL=ml26-wine-sulfite-gru-pso-forecast uvicorn main:app`):
  - `GET /health` → 200, `loaded: true`.
  - `/predict` batch con 30 lotes de test → 200, 30 predicciones con `risk_band`.
  - `/predict` inline, los 18 golden cases en los dos modos → ver tabla.
  - Lecturas sin `stage_progress` → 422 (KI-01).
  - `/train` sin `mlflow_run_id` → 422.
- [x] **Contrato de reentrenamiento:**
  - El modelo se guarda solo en MLflow.
  - Un fallo de subida da 502.
  - Un run sin modelo da 422.
  - El modelo de un run nunca se guarda en `self` (`test_user_model_isolation.py`).

## Correctitud frente al código original (datos reales del entregable)

| Comprobación | Resultado |
|---|---|
| Preparación de datos del plugin (`sequential.csv.gz` → features + split por lote + ventanas) frente a `data/splits/{train,val,test}_seq_{X,y}.npy` | Idéntica (420/90/90 lotes, 14.108/2.971/3.096 ventanas, diferencia 0) |
| Modelo servido sobre las 3.096 ventanas de test frente a `data/predictions/gru_pso_test_predictions.csv` | Diferencia máxima 4e-6 (SO₂) / 1e-6 (riesgo) |
| Camino operativo: 30 lotes de test truncados en puntos distintos, plugin frente a `prepare_raw_model_input` + `predict_sequence` originales | Diferencia máxima 1,7e-5 mg/L / 9e-7 |
| `/train` del plugin frente a `train_sequence_with_config` original ejecutado aislado (semilla 7, CPU) | **Pesos idénticos (diferencia 0)**; val 1,56973, test 1,40037, 25 épocas (mejor la 17), 240 s |
| El mismo reentrenamiento frente al artefacto entregado | val 1,5697 frente a 1,5647; test 1,4004 frente a 1,3983; mismas épocas. Diferencia del entorno, que la memoria §9.1 anticipa |

## Golden dataset (Parte B) — 18 casos contra el endpoint HTTP real

Tolerancias de `manifest.tolerance_policy`:
- **Ventana procesada:** atol 0,01 mg/L y 0,001 de riesgo.
- **Camino operativo** (lote + lecturas en modo bodega, con `stage_progress`): 2 × MAE de test, es decir
  3,0165 mg/L y 0,0857.

| Caso | Lote | Lecturas | Esperado SO₂ | Ventana: SO₂ | Δ SO₂ / Δ riesgo | ¿OK? | Operativo: SO₂ | Δ SO₂ / Δ riesgo | ¿OK? | Banda real / predicha |
|---|---|---|---|---|---|---|---|---|---|---|
| caso_001 | LOT-00013 | 90 | 34.195343 | 34.195343 | 0.0e+00 / 0.0e+00 | ✅ | 34.195343 | 0.0e+00 / 0.0e+00 | ✅ | bajo / bajo |
| caso_002 | LOT-00042 | 68 | 21.993731 | 21.993731 | 0.0e+00 / 0.0e+00 | ✅ | 21.993731 | 0.0e+00 / 0.0e+00 | ✅ | alto / alto |
| caso_003 | LOT-00049 | 42 | 28.697739 | 28.697739 | 0.0e+00 / 0.0e+00 | ✅ | 28.697739 | 0.0e+00 / 0.0e+00 | ✅ | bajo / bajo |
| caso_004 | LOT-00065 | 84 | 22.813425 | 22.813425 | 0.0e+00 / 0.0e+00 | ✅ | 22.813425 | 0.0e+00 / 0.0e+00 | ✅ | medio / alto |
| caso_005 | LOT-00117 | 50 | 21.564476 | 21.564478 | 2.0e-06 / 1.0e-06 | ✅ | 21.564478 | 2.0e-06 / 1.0e-06 | ✅ | alto / alto |
| caso_006 | LOT-00277 | 46 | 30.575390 | 30.575390 | 0.0e+00 / 0.0e+00 | ✅ | 30.575390 | 0.0e+00 / 0.0e+00 | ✅ | medio / medio |
| caso_007 | LOT-00290 | 64 | 35.890995 | 35.890999 | 4.0e-06 / 0.0e+00 | ✅ | 35.890999 | 4.0e-06 / 0.0e+00 | ✅ | bajo / bajo |
| caso_008 | LOT-00301 | 76 | 34.954269 | 34.954269 | 0.0e+00 / 0.0e+00 | ✅ | 34.954269 | 0.0e+00 / 0.0e+00 | ✅ | bajo / bajo |
| caso_009 | LOT-00333 | 84 | 19.008781 | 19.008781 | 0.0e+00 / 0.0e+00 | ✅ | 19.008781 | 0.0e+00 / 0.0e+00 | ✅ | alto / alto |
| caso_010 | LOT-00403 | 88 | 21.952858 | 21.952858 | 0.0e+00 / 0.0e+00 | ✅ | 21.952858 | 0.0e+00 / 0.0e+00 | ✅ | alto / alto |
| caso_011 | LOT-00432 | 110 | 10.537428 | 10.537429 | 1.0e-06 / 0.0e+00 | ✅ | 10.537429 | 1.0e-06 / 0.0e+00 | ✅ | alto / alto |
| caso_012 | LOT-00475 | 66 | 24.541862 | 24.541862 | 0.0e+00 / 0.0e+00 | ✅ | 24.541862 | 0.0e+00 / 0.0e+00 | ✅ | medio / medio |
| caso_013 | LOT-00500 | 28 | 25.994129 | 25.994127 | 2.0e-06 / 0.0e+00 | ✅ | 25.994127 | 2.0e-06 / 0.0e+00 | ✅ | bajo / bajo |
| caso_014 | LOT-00526 | 26 | 18.721996 | 18.721996 | 0.0e+00 / 0.0e+00 | ✅ | 18.721996 | 0.0e+00 / 0.0e+00 | ✅ | medio / medio |
| caso_015 | LOT-00529 | 30 | 18.499367 | 18.499367 | 0.0e+00 / 0.0e+00 | ✅ | 18.499367 | 0.0e+00 / 0.0e+00 | ✅ | alto / medio |
| caso_016 | LOT-00555 | 42 | 27.656008 | 27.656008 | 0.0e+00 / 0.0e+00 | ✅ | 27.656008 | 0.0e+00 / 0.0e+00 | ✅ | alto / medio |
| caso_017 | LOT-00585 | 88 | 32.123852 | 32.123852 | 0.0e+00 / 0.0e+00 | ✅ | 32.123852 | 0.0e+00 / 0.0e+00 | ✅ | medio / medio |
| caso_018 | LOT-00591 | 48 | 27.144464 | 27.144464 | 0.0e+00 / 0.0e+00 | ✅ | 27.144464 | 0.0e+00 / 0.0e+00 | ✅ | medio / bajo |

**Resultado:**
- Ventana procesada: **18/18**.
- Camino operativo: **18/18**. Con `stage_progress` reproduce la ventana del original al bit (diferencia ≤ 4e-6),
  muy por debajo de la tolerancia de 2 × MAE.
- Banda de riesgo predicha igual a la real en 14/18 casos. Es informativo: la Tabla 18 de la memoria
  declara un 87,5 % de acierto exacto de banda.

Ningún caso se ha silenciado ni se ha ajustado ninguna tolerancia.

## Limitaciones vigentes (no son fallos de integración)

- **Datos sintéticos (KI-02).** Todas las métricas proceden del simulador; falta validar con históricos reales de bodega.
- **`stage_progress` obligatorio (KI-01).** Sin él, el código original triplica el error: MAE de SO₂ 4,82 frente
  a 1,51.
- **`/train` necesita `free_sulfite_mg_l` continuo (KI-03),** que una bodega real no mide.
- **No se repite la búsqueda PSO (KI-04).** La memoria §10 recomienda repetirla al pasar a datos reales.

## Estado final

**LISTO PARA PR** tras corregir `/train`.

**Pendiente (acción humana):** subir `gru_pso.pkl` y `best_model.json` a
`artifacts/fixed/ml26_wine_sulfite_gru_pso_forecast/` en S3 (KI-13).
