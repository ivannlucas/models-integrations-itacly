# Verificación — ml36-dairy-dnl-co2-emissions-optimizer (a36)

Entorno aislado (fuera del repo): Python 3.11, numpy 2.4.2, pandas 3.0.1, scikit-learn 1.8.0, torch 2.14.1 (CPU), deap 1.4.4.

## Checklist técnico
- [x] flake8 (config del repo): 0 errores
- [x] pytest wiring ml36: 14/14 (global: 25/25 junto con ml34)
- [~] pylint: 9.93/10 sin line-too-long (ml34 baseline 9.74/10). Solo avisos de estilo (too-many-locals/arguments)
- [ ] pip-audit: no ejecutado (requiere red y el requirements.txt completo; sin cambios de dependencias)
- [x] Arranque (mismo router/contenedor que main.py, solo con ml36) + health + stats + predict inline/batch + optimize inline/batch + train: OK
- [x] /train: 200 con métricas (1200 filas reales, 70 épocas, 8 s); fichero inexistente → 400

## Carga real de artefactos
Modelo `.pt`, `final_model_config.json`, `scaler_X.pkl`, `scaler_Y.pkl` y `ga_policy_global.json` cargan correctamente.
Los scalers cargan con scikit-learn 1.8.0 y también con 1.5.2 (la versión fijada en requirements.txt del repo), con un
`InconsistentVersionWarning` y resultados idénticos (hash de las transformaciones del test set idéntico en ambas versiones).
Requieren numpy >= 2 (formato de pickle `numpy._core`).

## Correctitud — predict (18 casos reales del test split)
| Comparación | Resultado |
|---|---|
| Plugin vs salida del propio modelo del equipo de IA (tol. 1e-3) | **18/18**, diferencia máx. 5e-5 (redondeo a 4 decimales) |
| Plugin vs valor real del dataset (tol. 2×MAE de la memoria: 0,038 °C / 0,615 kg) | 12/18 |

Los 6 casos que superan 2×MAE contra el valor real (predict_004/007/011/015/016/017) tienen **exactamente el mismo error
que el modelo original** (diferencia plugin-original < 5e-5): es el error normal del modelo en filas individuales
(R² de CO2 = 0,895), no del plugin. No se ha ajustado la tolerancia.

Prueba global sobre las 7704 filas del test: MAE T_out 0,01909 / R² 0,999492; MAE CO2 0,30759 / R² 0,895145 —
coinciden con la memoria (Tabla 11) y con `final_test_metrics.json`. Diferencia máx. plugin vs predicciones del equipo: 4e-6.

## Correctitud — optimize (4 casos reales de validación)
| Comprobación | Resultado |
|---|---|
| Política estática + operación actual (deterministas), tol. 1e-3 | **4/4** |
| GA adaptativo/hybrid vs `realtime_decisions.csv` del equipo (semilla 43 + fila) | **4/4 idénticos** (CO2 ≤ 4e-5, setpoints ≤ 5e-5) |
| Reproducible con la misma semilla | 4/4 |
| T_out >= 72,5 °C | 4/4 |

Nota: la comprobación "CO2 recomendado <= CO2 de la política" escrita en el manifest NO es una propiedad del modelo
(en optimize_003/004 la política da menos CO2 pero incumple 72,5 °C, y el GA elige una solución más cara pero factible).
Es un comentario erróneo del manifest, no un fallo del plugin.

## Estado final
Plugin verificado a nivel técnico y numérico. REQUIERE confirmación del equipo de IA/producto en los puntos de decisión
del informe de chat antes de PR (restricción térmica, documentación inconsistente). Sin commit.

## Entorno de producción (Docker) — scalers
`Dockerfile` usa python:3.12-slim e instala `requirements.txt` tal cual. Resolviendo ese fichero con pip (--dry-run, Python 3.11)
se obtiene numpy 2.4.6, scikit-learn 1.5.2, torch 2.13.0, deap 1.4.4, pandas 3.0.6, tensorflow 2.21.0.
numpy >= 2 (requisito de los pickles) se cumple y los scalers cargan con scikit-learn 1.5.2 (aviso de versión, transformaciones
idénticas a 1.8.0). No hace falta modificar dependencias. Limitaciones: resolución hecha con Python 3.11 (la imagen usa 3.12) y
numpy no está fijado en `requirements.txt`; confirmar en el build real de CI.

## Aclaración sobre el manifest
La nota de los casos `optimize_*` decía que el CO2 recomendado era <= al de la política. Es incorrecto (corregido en el manifest):
el GA adaptativo se reproduce exactamente con la semilla 43 + índice de fila.

## Addendum — Revisión contra el código original actualizado (2026-10-14)

`inbox/a36/codigo/` se ha actualizado con el código original. El repo GitHub tenía una versión de ml36 anterior a la de Bitbucket (merge de los modelos 36/48), sin los arreglos del contrato MLflow; se alinean y se cierran los gaps.

**Contrato MLflow — [CORREGIDO en GitHub]:** `mlflow_run_id` obligatorio; un run sin modelo completo da 422 (`@require_user_model`, sin caer al modelo base y limpiando el directorio temporal); un fallo de subida da 502 (`ModelPersistenceError`); el modelo del usuario se resuelve en variables locales, nunca en `self` (peticiones concurrentes). Los 5 tests transversales que fallaban (`test_mlflow_persistence_contract`, `test_user_model_download_hygiene`) pasan.

**Restricción térmica — [RESUELTO]:** la versión GitHub rechazaba con 422 (`ThermalSafetyViolationError`) las recomendaciones con `T_out < 72,5 °C`. El original no lo hace: `run_realtime_decision_pipeline` devuelve siempre el mejor individuo del GA (la restricción solo penaliza el fitness). Se mantiene el comportamiento de Bitbucket (06/10, confirmado con el equipo de IA): `recommended_factible` es informativo y la plataforma debe validarlo. ml34 conserva su 422.

**Unidades:** `Delta_P` en bar y `Viscosity` en cP, confirmadas por el simulador (`potencia_bomba = Q · delta_p · 100000 Pa`; viscosidad ≈ 2 mPa·s).

**Inferencia (artefactos de S3, MD5 = entregados):**

| Comprobación | Resultado |
|---|---|
| Predict 18 golden vs modelo del equipo | **PASS 18/18** (dif. ≤ 5e-5, redondeo a 4 decimales) |
| Test completo (7704 filas) vía `predict_batch` | **PASS** — T_out MAE 0,019092 / R² 0,999492; CO2 MAE 0,307588 / R² 0,895145 (= memoria) |
| Optimize 4 golden (política estática y GA hybrid, semilla 43 + fila) | **PASS 4/4** (dif. ≤ 5e-5) |
| `predict_batch` optimize, 25 primeras filas de validación vs `realtime_decisions.csv` | **PASS** (dif. ≤ 5e-5) |

**Reentrenamiento — [CORREGIDO]:** `/train` hacía un *fine-tuning* de los pesos servidos con los scalers servidos, validaba con el último 15 % de filas y calculaba las métricas sobre todo el CSV (sin test aparte). Ahora es el port de `split_normalization.py` + `trainer.py` (`training.py`):

- split 70/15/15 por ciclo (`cycle_id` o reinicios de `t_ciclo`), equilibrado por estación si hay columna de estación, mes o fecha, con `default_rng(42)`;
- StandardScaler y MinMaxScaler ajustados en train; el modelo y sus scalers se suben al run de MLflow;
- MLP nuevo 1×64 ReLU, dropout 0,1, semilla 1 + 500, Adam 1e-4, MSE, batch 128, hasta 400 épocas con paciencia 30;
- métricas en el split de test en unidades reales; la respuesta añade `n_train`, `n_val`, `n_test`, `best_val_loss` y `training_time_s`.

| Comprobación con `df_pasteurizacion_master.csv` (51 389 filas) | Resultado |
|---|---|
| Split vs `train/val/test_raw.csv` | **PASS** — 35 930 / 7 755 / 7 704, mismas filas y orden (dif. 6,6e-24, round-trip CSV) |
| `scaler_X` / `scaler_Y` vs entregados | **PASS** — idénticos |
| Épocas y mejor val_loss | **PASS** — 131 épocas, 0,00081977 (= `final_training_history.json`) |
| Métricas de test vs `final_test_metrics.json` | **PASS** — T_out MAE 0,019090 / R² 0,999492; CO2 MAE 0,307590 / R² 0,895144 (entregado 0,019092 / 0,999492 / 0,307587 / 0,895145) |
| Pesos vs `MLP_Final_Best_From_Search.pt` | dif. máx. 4e-3: entradas idénticas en float32; la diferencia es de la aritmética de otra versión de torch/BLAS |
| `train()` → MLflow (simulado) | **PASS** — sube pesos, config y los scalers nuevos; un fallo de subida da `ModelPersistenceError` |

Duración: 60 s en CPU. Desviaciones deliberadas (documentadas en `training.py`): features = `FEATURE_ORDER` fijo; siempre CPU; sin `torch.use_deterministic_algorithms` (global al proceso); se descarta un último batch de 1 muestra (BatchNorm); la política global GA no se recalcula (el modo optimize usa la de los artefactos base).

**Tests:** `tests/unit/test_ml36_training.py` (nuevo, 11 tests, incluido el de reproducción con los datos entregados) y el de cableado: 24/24. Suite GitHub: 997 passed. Bitbucket (copia): 1003 passed, 3 skipped. flake8 limpio.

**Pendiente:** `/train` contra un servidor MLflow real. [CORREGIDO] Las fichas de a36 sí estaban en S3 (`ml-files/modelo-36/`): se han revisado contra el código y los datos (cifras verificadas) y se han regenerado en `.docx` con la plantilla (`datos_a36.json`), añadiendo la verificación del reentrenamiento a la ficha técnica.
