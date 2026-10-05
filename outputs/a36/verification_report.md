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
