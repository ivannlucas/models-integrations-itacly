# Verificación — ml43-cereals-dnsl-anomaly-fault-detection (CU43+CU44)

Fecha: 2026-08-05
Rama de origen del código: `feature/ML43-1-neurofuzzy-model` (repo hermano `Bitbucket`, mismo
plugin ya integrado allí; este informe cubre la integración equivalente en
`models-integrations-itacly`, siguiendo el proceso auditado del repo: manifest-extraction →
plugin-integration → verification).

## Checklist técnico

- [x] flake8 (`app/plugins/ml43_cereals_dnsl_anomaly_fault_detection`, `app/registry.py`,
  `app/domain/services/exceptions.py`): **0 errores**. Se corrigieron 3 hallazgos iniciales
  (import sin usar en `_vendor/preprocess.py`, línea en blanco final en `__init__.py`, dos
  variables sin usar en `plugin.py::train()` — ahora expuestas en `TrainResponse` en vez de
  descartadas).
- [x] pytest (`tests/unit/`, entorno con torch/fastapi/pydantic/shap/sklearn instalados
  ad-hoc): **293 passed**, 2 failed — ambos fallos son **preexistentes y no relacionados**
  (`tests/unit/test_infrastructure.py::TestModelo10ModelLoader::*`, requieren `torchvision`,
  no instalado en este sandbox; no tocan código de ml43). El nuevo
  `tests/unit/test_ml43_cereals_dnsl_anomaly_fault_detection.py` pasa 7/7.
- [x] pylint (`app/plugins/ml43_cereals_dnsl_anomaly_fault_detection`, `--disable=import-error`):
  **8.73/10**. Todos los hallazgos están en `_vendor/` (código del equipo de IA, portado tal
  cual — line-too-long, too-many-locals/arguments, wrong-import-order, naming style). Mismo
  patrón y orden de magnitud que el plugin hermano ya integrado `ml45_cereals_dnsl_critical_point_detection`
  (**8.79/10** con el mismo tipo de hallazgos en su propio `_vendor/`) — no hay regresión de
  calidad respecto al precedente de esta misma familia DNSL. Código propio del plugin
  (`plugin.py`, `preprocessing.py`, `postprocessing.py`, `model_loader.py`, `mlflow_utils.py`,
  `constants.py`, DTOs) sin hallazgos.
- [ ] pip-audit: **no completado**. Se intentó contra `requirements.txt` (repo completo, ~40
  paquetes) pero no terminó en un tiempo práctico en este sandbox (requiere resolver
  vulnerabilidades vía red para cada paquete). Pendiente de ejecutar en un entorno con acceso de
  red sin restricción de tiempo antes de merge — no bloqueante para este informe porque no se
  ha añadido ninguna dependencia nueva más allá de `shap` (ya presente en `requirements.txt`,
  añadida previamente para `ml45`).
- [x] Arranque local + health + predict + stats + train: **OK**, con matices — ver nota abajo.
- [x] `/train`: manifest declara `training.supported: true` → **200** con `TrainResponse`
  conteniendo las métricas de `training.metrics_returned` (subconjunto razonado — ver
  `plugin.py::train()` docstring: no se reproduce el `DNFLoss` completo con sus términos de
  regularización estructural, solo BCE simple sobre el mismo optimizador Adam/lr/weight_decay
  del manifest; ver "Nota sobre `train()`" abajo).

### Nota sobre el arranque local

`main.py` importa `app.registry.REGISTRY` a nivel de módulo, que a su vez importa **todos** los
plugins registrados (incluidos los de imagen/vídeo que dependen de `torchvision`/`cv2`/
`tensorflow`/`ultralytics`), independientemente del filtro `MODEL=<id>` (el filtro solo actúa
sobre `_active_entries` **después** de que `REGISTRY` ya se ha construido). Instalar el stack
completo de `requirements.txt` (incluyendo `detectron2` desde código fuente, según indica el
propio fichero) no era practicable en este sandbox. Para no dejar sin verificar el arranque real,
se montó una app FastAPI standalone que registra **únicamente** el router de ml43 con
`make_model_router` + `ModelContainer` reales (mismo mecanismo que usa `main.py`, sin tocar
`router_factory.py` ni `container.py`) y se hicieron peticiones HTTP reales:

```
GET  /models/ml43-cereals-dnsl-anomaly-fault-detection/health   -> 200 {"loaded": true, ...}
GET  /models/ml43-cereals-dnsl-anomaly-fault-detection/stats    -> 200 (StatsResponse completo)
POST /models/ml43-cereals-dnsl-anomaly-fault-detection/predict  (inline) -> 200, incluye
     corrective_actions (CU44) con Estado_del_sistema, Bloques_principales, Acciones_sugeridas
POST /models/ml43-cereals-dnsl-anomaly-fault-detection/predict  (batch, golden_cycles.csv)
     -> 200, 35 predicciones — idénticas a las de la Parte B (ver abajo)
POST /models/ml43-cereals-dnsl-anomaly-fault-detection/train    (golden_cycles.csv) -> 200
     TrainResponse con métricas reales; upload_warning informativo porque `mlflow` no está
     instalado en este sandbox ad-hoc (degradación esperada, no un fallo de wiring)
```

Los artefactos reales del checkpoint entregado (`best_dnf_model.pt`, `scaler.pkl`,
`xai_background.npy`) se colocaron en `artifacts/ml43_cereals_dnsl_anomaly_fault_detection/`
para poder ejecutar esta verificación end-to-end.

## Correctitud (golden dataset)

Los 7 `golden_cases` de `inbox/a43/manifest.yaml` (NORMAL + 6 familias de falla) se
reprodujeron llamando directamente a `Ml43CerealsDnslAnomalyFaultDetectionPlugin.predict_batch()`
(mismo código que expone `/predict`) sobre el CSV real usado para generarlos
(`inbox/a43/codigo/data/input/golden_cycles.csv`, 7 ciclos de `data/splits/test.csv`) y
comparando cada una de las 35 ventanas contra el valor esperado del manifest.

| Caso | Ventanas | Clase (todas) | Prob. — diff máxima | ¿OK? |
|---|---|---|---|---|
| cycle_2400_normal | 5/5 | ✓ | 0.00002 | ✓ |
| cycle_2407_resistencia_degradada | 5/5 | ✓ | 0.00004 | ✓ |
| cycle_2417_refractario_erosionado | 5/5 | ✓ | 0.00003 | ✓ |
| cycle_2424_sensor_descalibrado_zona1_drift | 5/5 | ✓ | 0.00004 | ✓ |
| cycle_2430_ventilador_defectuoso | 5/5 | ✓ | 0.00005 | ✓ |
| cycle_2443_aislamiento_degradado | 5/5 | ✓ | 0.00003 | ✓ |
| cycle_2519_valvula_obstruida | 5/5 | ✓ | 0.00005 | ✓ |

Tolerancia usada: `probability_atol=0.005` (definida por caso en el manifest, igual para los 7).
Resultado: **35/35 ventanas dentro de tolerancia** (diferencias reales de ~1e-5, atribuibles a
las distintas versiones de librerías entre el entorno de generación del manifest y este de
verificación — ver `inbox/a43/manifest.yaml::known_issues`, no a un error de puerto).

El caso `cycle_2519_valvula_obstruida` reproduce fielmente el falso negativo real del modelo
(las 5 ventanas se predicen "No Fallo" a pesar de que el ciclo sí tiene falla) — confirma que el
puerto no "corrige" ni oculta el comportamiento real del checkpoint entregado.

La capa XAI (CU44) se verificó cualitativamente vía el endpoint `/predict` inline en vivo (ver
arriba): genera `Estado_del_sistema`, `Bloques_principales` y `Acciones_sugeridas` coherentes con
el catálogo de `config.yaml::xai.action_config` / memoria CU44 — no se han definido golden_cases
cuantitativos para esta capa (ver `inbox/a43/manifest.yaml::known_issues`, último punto: la
memoria CU44 reporta agregados estadísticos por grupo de confusión, no valores reproducibles
ventana a ventana).

## Nota sobre `train()`

`plugin.py::train()` usa el mismo optimizador (Adam, lr=0.00175, weight_decay=1e-5) y arquitectura
(`DEFAULT_MODEL_CFG`, idéntica al checkpoint entregado) que el manifest, pero simplifica la
función de pérdida: usa `BCEWithLogitsLoss` simple sobre `anomaly_score`, sin los términos de
regularización estructural del `DNFLoss` original (diversidad de reglas, entropía de α, balance de
uso de reglas — ver `inbox/a43/manifest.yaml::training.hyperparams.dnf_loss`). Es una
simplificación deliberada y documentada (ver docstring de `train()`) para un endpoint de
reentrenamiento vía HTTP — reproducir el `DNFLoss` completo requeriría portar `src/training/loss.py`
íntegro, fuera de alcance razonable para esta integración. El umbral de decisión sí se recalcula
por búsqueda (maximizando F1 de la clase Fallo sobre el split de test), replicando el criterio
real descrito en la memoria CU43 (Sección 4.2).

## Estado final

**LISTO PARA PR**, con las siguientes salvedades a revisar por una persona antes de merge:

1. **pip-audit no ejecutado** — correr en un entorno con red antes de merge.
2. **Discrepancia memoria vs checkpoint** (ver `inbox/a43/manifest.yaml::known_issues`, primer
   punto) — la memoria CU43 v1.4 describe una configuración de modelo distinta a la
   efectivamente entregada. El código de este plugin sigue el checkpoint real (verificado), no
   la memoria. Requiere que el equipo de IA confirme si la memoria debe actualizarse o si el
   checkpoint entregado es el definitivo.
3. **`train()` simplificado** respecto al `DNFLoss` completo (ver nota arriba) — aceptable para
   un endpoint de fine-tuning ligero, pero no reproduce exactamente el procedimiento de
   entrenamiento original de 200 épocas con Optuna.
4. El servidor completo (`main.py`, todos los modelos) no se pudo arrancar en este sandbox por
   falta de `torchvision`/`cv2`/etc. de otros plugins — la verificación de arranque se hizo con
   una app standalone de un solo router (ver nota arriba). Se recomienda repetir el arranque
   completo (`python main.py` con `MODEL=ml43-cereals-dnsl-anomaly-fault-detection`) en un
   entorno con `requirements.txt` completo instalado antes de merge, aunque el mecanismo de
   carga es idéntico y ya verificado.
