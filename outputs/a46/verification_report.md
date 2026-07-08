# Verificación — ml46-dairy-fouling-clog-detection (a46)

Fecha: 2026-07-08
Rama: `feature/model-46-integration`
Plugin: `app/plugins/ml46_dairy_fouling_clog_detection/`
Manifest: `inbox/a46/manifest.yaml`

## Checklist técnico

- [x] **flake8** (repo completo): 0 errores en los ficheros tocados por esta integración
      (`app/plugins/ml46_dairy_fouling_clog_detection/`, `app/registry.py`,
      `app/domain/services/exceptions.py`, `tests/unit/test_ml46_dairy_fouling_clog_detection.py`,
      `tests/conftest.py`). Los hallazgos que sí aparecen en `flake8 .` son preexistentes y
      ajenos a esta integración: código crudo de otro modelo en `inbox/a47/...` (no se despliega)
      y un import no usado preexistente en `tests/conftest.py:107` (`Ml35DairyOptimizeResp`, de
      antes de esta rama).
- [x] **pytest** (`tests/unit/`, `-p no:flask` — ver nota abajo): **264/271 passed**.
      Los 7 fallos son **todos** el mismo bug preexistente en
      `app/infrastructure/http/router_factory.py:64`
      (`status.HTTP_422_UNPROCESSABLE_CONTENT` no existe en la versión de starlette instalada;
      el nombre correcto es `HTTP_422_UNPROCESSABLE_ENTITY`). Afecta a **todo** plugin que dispare
      una `extra_predict_exceptions` (ml2, ml5 ×2, ml7, ml8, wine-sulphite y, ahora, ml46) — no es
      un fallo introducido por esta integración; se reproduce igual en `main` antes de este
      cambio. Se deja documentado para que se corrija en `app/infrastructure/` (fuera del alcance
      de `plugin-integration`, que no debe tocar esa capa). El resto de los 6 tests de ml46
      (health, stats, predict_inline, predict_batch, train, rechazo por longitud insuficiente)
      pasan.
      Nota: en este entorno, `pytest-flask` está instalado globalmente y su fixture `_monkeypatch_response_class`
      colisiona con el fixture `app` (FastAPI) de este repo — hay que ejecutar con `-p no:flask`
      o desinstalar `pytest-flask` del entorno; también es preexistente, no relacionado con ml46.
- [x] **pylint** `app/plugins/ml46_dairy_fouling_clog_detection/`: **8.05/10**. Hallazgos son
      `line-too-long`/`too-many-locals`/`too-many-branches`/`duplicate-code` — mismas categorías
      que en los plugins ya mergeados (ml30: 9.68/10, ml35: 8.89/10), inherentes a portar un
      pipeline de features/ventanas más grande que el de esos modelos. Un falso positivo esperado:
      `E1102: F.softplus is not callable` (pylint no resuelve bien `torch.nn.functional`). Cero
      hallazgos de tipo bug/correctness.
- [x] **pip-audit** `-r requirements.txt`: 2 CVEs reportadas (`torch==2.11.0` CVE-2025-3000,
      `keras==3.12.3` PYSEC-2026-73) — **preexistentes**, `requirements.txt` no se modificó en
      esta integración (ml46 solo usa dependencias ya presentes: torch, pandas, numpy,
      scikit-learn, PyYAML). 0 CVEs nuevas introducidas por ml46.
- [x] **Arranque local + health + predict + stats + train**: OK. Servidor real
      (`MODEL=ml46-dairy-fouling-clog-detection uvicorn main:app`), probado por HTTP con `curl`:
  - `GET /health` → `200 {"status":"ok","loaded":true}`
  - `POST /predict` (inline, ventana real de `caso_001` con historial completo del activo) →
    `200`, `pred_severity=0.0001760900777` vs. esperado `0.0001760901068` (diff ≈ 3e-11)
  - `POST /predict` (batch, `data/splits/test_rows.csv` completo) → `200`, 6773 ventanas
    puntuadas, 28 episodios de alerta consolidados
  - `GET /stats` → `200`, contrato de entrada/salida y métricas reales del `test_window_metrics`
  - `POST /train` (fine-tuning sobre `data/splits/val_rows.csv`, 6282 ventanas, 8 épocas, ~63 s
    en CPU) → `200` con `TrainResponse` completo (`stage_accuracy=0.9917`, etc. — mejor que el
    checkpoint base porque el fine-tune se hizo sobre val, no sobre test; solo valida que el
    endpoint funciona end-to-end, no es una comparación de métricas). **El checkpoint local se
    restauró al original de `models/artifacts/selected_model.pt` (hash md5 `8c47559b...`
    verificado) inmediatamente después de esta prueba**, para no contaminar la verificación de
    golden_cases ni lo que se entrega en `artifacts/ml46_dairy_fouling_clog_detection/`.

## Correctitud (golden dataset)

Los 13 `golden_cases` de `inbox/a46/manifest.yaml` se verificaron llamando a
`plugin.predict_batch(data_path="data/splits/test_rows.csv")` (equivalente a `POST /predict`
modo batch) y comparando cada campo de `expected` contra la fila de salida que corresponde a
`(asset_id, window_end_timestamp)`. `pred_stage` se compara por igualdad exacta; el resto de
campos con `float_rtol` tal como especifica el manifest.

| Caso | pred_stage (esp./obt.) | Máx. diff. relativa | Campo | ¿Dentro de tolerancia (1e-4)? |
|---|---|---|---|---|
| caso_001 | 0 / 0 | 1.92e-06 | p_stage1 | ✅ |
| caso_002 | 1 / 1 | 9.35e-07 | p_stage0 | ✅ |
| caso_003 | 2 / 2 | 9.96e-07 | p_clog_h | ✅ |
| caso_004 | 0 / 0 | 2.82e-06 | p_clog_h | ✅ |
| caso_005 | 1 / 1 | 1.46e-06 | p_foul_h | ✅ |
| caso_006 | 2 / 2 | 9.82e-07 | p_stage1 | ✅ |
| caso_007 | 1 / 1 | 9.68e-07 | p_foul_h | ✅ |
| caso_008 | 1 / 1 | 1.01e-06 | p_actionable_foul_h | ✅ |
| caso_009 | 1 / 1 | 9.67e-07 | p_stage0 | ✅ |
| caso_010 | 1 / 1 | 9.72e-07 | p_foul_h | ✅ |
| caso_011 | 1 / 1 | 1.04e-06 | pred_ttu_min | ✅ |
| caso_012 | 1 / 1 | 1.86e-06 | p_actionable_foul_h | ✅ |
| caso_013 | 1 / 1 | 9.35e-07 | p_stage2 | ✅ |

Tolerancia usada: `float_rtol = 1e-4` + `pred_stage` exacto (tal como fija cada golden_case en
el manifest). Diferencia real observada: máximo 2.82e-6 — tres órdenes de magnitud por debajo de
la tolerancia, consistente con ruido de redondeo de `float32`, no con una discrepancia de wiring.

**Resultado: 13/13 casos dentro de tolerancia.**

### Nota sobre `predict_inline` vs. `predict_batch`

Durante la verificación se detectaron y corrigieron dos causas reales de desviación que sólo
afectan al modo `predict_inline` (ventana única) si no se manejan correctamente — documentadas
también en `inbox/a46/manifest.yaml` → `known_issues`:

1. **`last_maintenance_type` no aparece en la Tabla 9 de la memoria** (contrato de entrada
   productiva) pero es una de las 76 features `no_clock` reales del modelo. Sin ella, se
   asume `"none"` para todas las filas, lo que se verificó que desvía `severity` ~8% y
   probabilidades raras (`p_foul_h`, `p_clog_h`) en >100% relativo en un caso de prueba real.
   Se añadió a `inputs.optional` del plugin (`constants.RAW_OPTIONAL_COLUMNS`) como corrección.
2. **Baseline de residuos por activo no visto en entrenamiento** (`estimate_prefix_baseline`):
   para activos que no estaban en `train_asset_baselines` (todo activo de producción real, y
   los propios `asset_00`/`asset_06` de test), los residuos dependen de las primeras ~8h de
   historial del activo. Con solo la ventana mínima de 120 minutos, `severity` se desvía ~8% del
   valor exacto del checkpoint; aportando el historial completo del activo, la desviación baja a
   ~1e-6 (ver comparación arriba). Documentado en
   `inputs.window.baseline_history_note` del manifest y en la descripción del campo `rows` de
   `PredictInlineRequest`.

Ambas causas se verificaron aislando el tensor de entrada (120×76) del plugin frente al
construido por el código original sin modificar (`inbox/a46/codigo/.../src/`): antes de la
corrección, diferían; después, coinciden bit a bit (`max abs diff = 0.0`) para el caso de prueba.
`predict_batch` nunca se vio afectado porque procesa el CSV completo (incluye siempre el
historial y `last_maintenance_type` si el CSV de origen los trae, como es el caso de
`data/splits/test_rows.csv`).

## Estado final

**LISTO PARA PR** (checklist técnico y correctitud en verde), con la siguiente salvedad para
revisión humana antes de mergear — no es un fallo de wiring, es una decisión de producto:

- El bug de `router_factory.py` (`HTTP_422_UNPROCESSABLE_CONTENT`) es preexistente y afecta a 6
  plugins ya en `main` además de a ml46. Recomendado corregirlo en un PR aparte de
  infraestructura (cambiar a `HTTP_422_UNPROCESSABLE_ENTITY`), no como parte de este PR de ml46.
- Ver `inbox/a46/manifest.yaml` → `known_issues` para la discrepancia mayor entre la memoria v1.2
  (100 activos, 2.29M filas) y los artefactos realmente entregados (10 activos, 198.615 filas) —
  ya confirmada contigo durante `manifest-extraction`, se usó el artefacto real como fuente de
  verdad. Cualquier cifra de negocio derivada de este modelo debe llevar advertencia de
  validación pendiente frente a telemetría real (dataset 100% sintético).
