# Verificación — ml18-meat-spatial-price-forecast (a18)

Fecha: 2026-10-02
Plugin: `app/plugins/ml18_meat_spatial_price_forecast/`
Manifest: `inbox/a18/manifest.yaml`

> **[CORREGIDO 2026-10-08] Reanálisis con el código v4.2 y `/train` habilitado.**
> Una primera copia del código subida el 2026-10-08 era la versión anterior (6 features, MAPE 12,15 %).
> Con la versión v4.2 (commit 5ceeeb7, 7 features), comprobado con datos reales:
>
> | Comprobación | Resultado |
> |---|---|
> | Original `predict --forecast` sobre `dataset_modelado_desde_2008.csv` | Test MAPE 9,932 % · R² 0,8114 (= memoria) |
> | Plugin frente al original, forecast de las 400 combinaciones CCAA-Producto | 400/400 · diferencia máxima 2e-6 (redondeo a 6 decimales) |
> | 4 golden cases reconstruidos del dataset real | 4/4 (diferencia ≤ 1e-6) |
> | `/train` del plugin frente a `train()` original, mismo TF 2.21 en CPU | **Idénticos a 4 decimales**: test MAPE 9,9552 % · R² 0,8110 · 17 épocas (mejor la 9) · 48.768/11.568/11.600 ventanas |
> | Modelo reentrenado servido con `mlflow_run_id` | Carga y predice |
> | Servidor real | `/health` loaded, `/predict` batch 200 (400 forecasts), `/train` sin run 422 |
>
> La diferencia de 0,02 puntos de MAPE frente al artefacto entregado viene del entorno (TF 2.20 frente
> a 2.21, otro hardware), no del port.
>
> **Cambios en el plugin:**
> - `training.py`: port literal de `train()`.
> - `/train` habilitado y persistido solo en MLflow; un fallo de subida da 502.
> - `/predict` y `/stats` con `mlflow_run_id`; un run sin modelo da 422.
> - Se resuelve la columna de precio aunque la cabecera traiga variantes de codificación, como
>   `resolve_target_column` del original.
>
> Las fichas técnica y funcional se han regenerado. Las líneas de abajo que dicen "`/train` → 501"
> son del estado anterior.
>
> **Pendiente (acción humana):** subir los artefactos de `inbox/a18/codigo/models/artifacts/best_gru_df_2008_con_renta/`
> a `artifacts/fixed/ml18_meat_spatial_price_forecast/` en S3.

## Estado de aceptación del modelo

GRU (seleccionado por el propio equipo de IA tras comparar LSTM/GRU/baselines) obtiene MAPE de
test `9.93%`, muy por debajo del umbral de aceptación del `20%` definido en la memoria (sección
"Validación temporal y calibración de umbrales"). La propia "Ficha de Valoración del Estado
Técnico del Modelo" (v2.1, 21/09/2026) marca explícitamente la **Opción A** — desarrollo técnico
completado, sin observaciones técnicas abiertas. Ver `inbox/a18/manifest.yaml::model_status`
para el detalle y las fuentes.

## Checklist técnico

- [x] **flake8** (`app/plugins/ml18_meat_spatial_price_forecast/`, `tests/unit/test_ml18_*.py`,
      `app/registry.py`, `tests/conftest.py`): **0 errores**.
- [x] **pytest** (`tests/unit/`): **7/7 passed** (tests propios de ml18: health, stats, predict
      inline, rechazo por `min_length` a nivel de DTO, `InsufficientRowsError` → 422, predict
      batch, `train` → 501). Suite completa del repo: **380/402 passed**, sin regresiones
      atribuibles a este cambio — confirmado explícitamente con `git stash` (sin los cambios de
      ml18: 373/395 passed, los mismos 22 fallos preexistentes por `ModuleNotFoundError: torch`
      en varios plugins ya existentes del repo — `torch` no está instalado en este entorno
      sandbox; 373+7=380, ninguno de los 22 fallos involucra a ml18).
- [x] **pylint** `app/plugins/ml18_meat_spatial_price_forecast/ --disable=import-error`:
      **9.33/10**. Avisos todos `C0301` (líneas >100 car. — criterio no real del proyecto:
      `.flake8` fija `max-line-length=120` con `E501` ignorado), `C0116` (docstrings en 3
      funciones privadas triviales de `preprocessing.py`) y un `R0914` (too-many-locals) en
      `inference.py::run_inference` — en línea con el resto de plugins del repo.
- [x] **pip-audit**: ml18 no añade ninguna dependencia nueva a `requirements.txt`
      (`tensorflow`/`scikit-learn`/`pandas`/`numpy`/`joblib` ya declaradas para otros plugins
      del repo, p. ej. `ml3_wine_disease_pest_forecast` también usa TensorFlow/Keras).
- [x] **Arranque local + health/predict/stats/train — OK**, contra un servidor FastAPI real (no
      mocks): se montó una app standalone (`make_model_router` + `ModelContainer`, el mismo
      wiring que usa `main.py`) solo para `ml18-meat-spatial-price-forecast`, evitando el
      *import* eager de `app.registry` (que carga todos los plugins del repo, varios con
      dependencias pesadas no instalables en este entorno sandbox). Se instaló
      `tensorflow==2.20.0` en un venv aislado para esta prueba; el artefacto real
      `artifacts/ml18_meat_spatial_price_forecast/model.joblib` (+ `x_scaler.json` +
      `y_scaler.json`) del equipo de IA se cargó sin mocks.
  - `GET /health` → `200 {"status":"ok","model":"ml18-meat-spatial-price-forecast","loaded":true}`
  - `POST /predict` (inline, los 5 `golden_cases`) → ver tabla de correctitud abajo.
  - `GET /stats` → `200`, incluye `metrics_reported` real del manifest.
  - `POST /train` → **`501`** con el detalle de por qué (procedimiento de entrenamiento
    entregado sin contrato de datos de cliente — ver manifest).
- [x] **`/train` según `manifest.training`**: `supported: false` en el manifest → el endpoint
      devuelve `501` (`TrainingNotSupportedError`). Confirmado, comportamiento esperado.
- [x] Las 4 predicciones de `golden_cases` se verificaron además cruzando la salida del plugin
      contra la salida literal de `python -m src.main predict --data <csv> --forecast` del CLI
      original entregado, y contra `data/predictions/predictions.csv` generado ejecutando
      `--backtest` sobre el dataset completo (76.736 filas) — los tres caminos (backtest sobre
      el dataset completo, forecast sobre una ventana reducida vía CLI, y el plugin integrado)
      coinciden a precisión de máquina para los mismos meses.

## Correctitud (golden dataset)

`inbox/a18/manifest.yaml::golden_cases` contiene 5 casos: 4 predicciones sobre combinaciones
CCAA-Producto reales de `data/processed/model_ready/dataset_modelado_desde_2008.csv` (incluido
un caso con CCAA insular sin vecinas — Baleares — que ejercita el fallback documentado en la
memoria) y 1 rechazo controlado por historial insuficiente. Los 5 se ejecutaron contra el
endpoint HTTP real (`POST /predict`, modo inline) — no contra una llamada directa a Python —
para verificar el wiring completo (DTO → use case → plugin → feature engineering espacial →
escalado → GRU → desescalado).

| Caso | Esperado (predicted_price) | Obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|
| caso_001_andalucia_carne_pollo | 4.542946 | 4.542946 | 0.0 | ✅ |
| caso_002_madrid_carne_vacuno | 10.615247 | 10.615248 | 0.000001 | ✅ |
| caso_003_castilla_y_leon_carne_cerdo_frontera_test | 6.440175 | 6.440175 | 0.0 | ✅ |
| caso_004_baleares_pollo_entero_sin_vecinas | 3.680698 | 3.680698 | 0.0 | ✅ |
| caso_005_rechazo_historial_insuficiente | HTTP 422 | HTTP 422 | — | ✅ |

Tolerancia usada: `tolerance_pct: 1.0` declarada en el manifest (margen de reproducibilidad
numérica entre entornos/versiones de TensorFlow); en la práctica 3/4 casos reprodujeron el valor
exacto (diferencia 0.0) y 1/4 difiere en el sexto decimal (float32), muy por debajo de la
tolerancia.

**Resultado: 5/5 casos dentro de tolerancia (ninguno silenciado ni ajustado).**

## Hallazgos durante esta verificación

Ninguno nuevo sobre el wiring en sí — las dos observaciones de fondo (imprecisión README vs.
código en el lag espacial, y la etiqueta de caso de uso CU18 vs. CU06) se identificaron durante
`manifest-extraction` y están documentadas en `inbox/a18/manifest.yaml::known_issues`, no aquí.

## Estado final

**LISTO PARA PR.** El modelo está aceptado por el propio equipo de IA (Ficha de Valoración,
Opción A) y el plugin reproduce exactamente su comportamiento real contra el artefacto GRU
entregado — no hay ningún hallazgo de aceptación pendiente de decisión humana en este manifest.
