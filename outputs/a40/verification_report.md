# Verificación — a40 (ml40-meat-refrigeration-aeration-fault-diagnosis)

**Plugin:** `app/plugins/ml40_meat_refrigeration_aeration_fault_diagnosis/`
**Manifest:** `inbox/a40/manifest.yaml` (23 golden cases)
**Fecha:** 2026-07-13 · Rama: `feature/model-40-integration`
**Entorno de verificación:** WSL2, Python 3.10, scikit-learn 1.7.0, pandas 2.x

## Checklist técnico (Parte A)

- [x] **flake8**: 0 errores en todo el repo (`flake8 .` exit 0). Se corrigió un F401 preexistente
  en `tests/conftest.py` (import sin uso de ml35, ajeno a a40) que rompía el pipeline.
- [x] **pytest**: 296/296 passed (`pytest tests/unit/ -p no:flask --cov=app`), incluyendo los 8
  tests nuevos de `test_ml40_meat_refrigeration_aeration_fault_diagnosis.py` (health, stats,
  inline, batch, train, validación Pydantic min_length y mapeo 422 de las 2 excepciones de
  dominio nuevas). Los tests unitarios validan wiring con `FakePlugin` (patrón del repo); la
  correctitud real se valida en la Parte B. DTOs con 100 % de cobertura.
- [x] **pylint** plugin: **9.54/10** (referencias del repo: ml46 = 8.05, ml35 = 8.89). Restan
  avisos C0301 (líneas >100) y R0914 (too-many-locals en el port fiel de las reglas
  neurosimbólicas), por debajo del umbral de ruido de los plugins ya integrados.
- [x] **pip-audit**: 2 vulnerabilidades **preexistentes** en dependencias del repo, ninguna
  introducida por ml40 (que no añade dependencias nuevas):
  `torch 2.11.0` (CVE-2025-3000, sin fix publicado) y `keras 3.12.3` (PYSEC-2026-73, fix 3.13.1).
- [x] **Arranque local + endpoints**: `main.py` completo no arranca en este entorno WSL por
  dependencias pesadas de *otros* plugins (`timm` de ml4 no instalado — limitación conocida del
  entorno local, no de ml40). Se verificó con servidor mínimo usando el `ModelContainer` y
  `make_model_router` reales (mismo wiring que `main.py`):
  - `GET /health` → 200 `{"status":"ok","loaded":true}`
  - `POST /predict` inline (ciclo crudo de aireado de 200 min, dataset real) → 200,
    `prediction=0 NORMAL, confidence=0.9672` (etiqueta real: 0) ✔
  - `POST /predict` inline con 80 filas de refrigeración → **422** con mensaje de histórico
    insuficiente (`InsufficientCycleHistoryError`) ✔
  - `GET /stats` → 200 con contrato completo de entradas/salidas y métricas reportadas ✔
- [x] **/train** (`manifest.training.supported: true`) → **200** con `TrainResponse` y las
  métricas de `metrics_returned`: reentrenado el sistema aireado sobre el dataset crudo completo
  (30.000 filas, 150 runs): `accuracy=1.0, f1_macro=1.0, precision_macro=1.0, recall_macro=1.0`
  (idéntico a la memoria Tabla 4). Artefactos de usuario guardados como `user_aireado_model.pkl`
  + `user_aireado_stats.yaml` **sin sobrescribir los artefactos fijos**.

## Correctitud (golden dataset, Parte B)

**Modo de verificación:** `POST /predict` batch con el split de test completo de cada sistema
(mismo modo con el que el pipeline original genera sus métricas), comparando el diagnóstico
consolidado por `run_id`.

**Fuente de los expected:** re-ejecución del código original entregado (sin modificar) sobre los
splits y artefactos actuales — los `data/predictions/evaluacion_test_*.csv` entregados estaban
obsoletos para aireado (ver "Hallazgos"). Consistencia agregada contra la memoria Tabla 4:
accuracy por run **1.0000 aireado (30/30)** y **0.9462 refrigeración (246/260)** vs. 1.00 y
0.94-0.95 declarados.

**Tolerancia usada:** `prediction` exacta + `prediction_name` exacto; `confidence` con
`atol=0.005` (media por run de `max(predict_proba)`, determinista dado el artefacto; el margen
cubre diferencias de deserialización entre versiones de sklearn). Derivada del carácter
determinista del pipeline — más estricta que el 5 % por defecto.

| Caso | Run | Esperado | Obtenido | Δ conf | ¿OK? |
|---|---|---|---|---|---|
| air_001 | 28 | 0 NORMAL · 0.956912 | 0 NORMAL · 0.956912 | 4.2e-07 | ✔ |
| air_002 | 40 | 0 NORMAL · 0.959454 | 0 NORMAL · 0.959454 | 2.0e-07 | ✔ |
| air_003 | 13 | 1 ENCOSTRAMIENTO · 0.981020 | 1 ENCOSTRAMIENTO · 0.981020 | 1.5e-07 | ✔ |
| air_004 | 25 | 1 ENCOSTRAMIENTO · 0.966762 | 1 ENCOSTRAMIENTO · 0.966762 | 3.1e-07 | ✔ |
| air_005 | 30 | 2 SATURACION_HIELO · 0.982461 | 2 SATURACION_HIELO · 0.982461 | 2.6e-07 | ✔ |
| air_006 | 38 | 2 SATURACION_HIELO · 0.988564 | 2 SATURACION_HIELO · 0.988564 | 9.7e-08 | ✔ |
| air_007 | 19 | 3 FALLO_VENTILADOR · 0.959579 | 3 FALLO_VENTILADOR · 0.959579 | 8.0e-08 | ✔ |
| air_008 | 27 | 3 FALLO_VENTILADOR · 0.961430 | 3 FALLO_VENTILADOR · 0.961430 | 2.5e-07 | ✔ |
| ref_001 | 1 | 0 NORMAL · 0.953922 | 0 NORMAL · 0.953922 | 2.5e-07 | ✔ |
| ref_002 | 104 | 1 COND_FOUL_MILD · 0.913105 | 1 COND_FOUL_MILD · 0.913105 | 4.2e-07 | ✔ |
| ref_003 | 201 | 2 COND_FOUL_SEVERE · 0.574579 | 2 COND_FOUL_SEVERE · 0.574579 | 3.6e-07 | ✔ |
| ref_004 | 303 | 3 EVAP_FAN_DEG · 0.999154 | 3 EVAP_FAN_DEG · 0.999154 | 8.6e-08 | ✔ |
| ref_005 | 404 | 4 EVAP_FAN_FAIL · 0.999873 | 4 EVAP_FAN_FAIL · 0.999873 | 4.6e-07 | ✔ |
| ref_006 | 501 | 5 UNDERCHARGE_MILD · 0.968246 | 5 UNDERCHARGE_MILD · 0.968246 | 5.0e-07 | ✔ |
| ref_007 | 603 | 6 UNDERCHARGE_SEVERE · 0.974143 | 6 UNDERCHARGE_SEVERE · 0.974143 | 2.4e-07 | ✔ |
| ref_008 | 700 | 7 OVERCHARGE · 0.961318 | 7 OVERCHARGE · 0.961318 | 1.6e-07 | ✔ |
| ref_009 | 803 | 8 SENSOR_DRIFT_PLUS · 0.994084 | 8 SENSOR_DRIFT_PLUS · 0.994084 | 4.5e-07 | ✔ |
| ref_010 | 903 | 9 SENSOR_DRIFT_MINUS · 0.992502 | 9 SENSOR_DRIFT_MINUS · 0.992502 | 2.7e-07 | ✔ |
| ref_011 | 1003 | 10 COMP_INEFFICIENCY · 0.963926 | 10 COMP_INEFFICIENCY · 0.963926 | 3.6e-07 | ✔ |
| ref_012 | 1100 | 11 NON_CONDENSABLES · 0.677540 | 11 NON_CONDENSABLES · 0.677540 | 3.8e-07 | ✔ |
| ref_013 | 1202 | 12 UNDERCHARGE_AND_COND_FOUL · 0.917532 | 12 UNDERCHARGE_AND_COND_FOUL · 0.917532 | 4.5e-07 | ✔ |
| ref_014 | 200 | 11 NON_CONDENSABLES (true=2) · 0.581579 | 11 NON_CONDENSABLES · 0.581579 | 4.2e-07 | ✔ |
| ref_015 | 1135 | 2 COND_FOUL_SEVERE (true=11) · 0.602035 | 2 COND_FOUL_SEVERE · 0.602035 | 4.5e-07 | ✔ |

**Resultado: 23/23 casos dentro de tolerancia** (Δ máx de confianza ≈ 5·10⁻⁷ — ruido de
representación decimal). ref_014/ref_015 verifican la reproducción de las *confusiones reales*
del modelo entregado (expected = predicción del modelo, no la verdad), no su acierto.

## Hallazgos (requieren atención humana — no bloquean la reproducción)

1. **Evaluaciones entregadas obsoletas (aireado).** El último commit del equipo de IA regeneró
   `aireado_model.pkl` (346 KB → 11.6 MB) y los splits (30 runs × 200 pasos), pero
   `data/predictions/evaluacion_test_aireado.csv` quedó con los run_id del split anterior (60
   runs × 100 pasos, ya inexistentes). Los golden cases se re-derivaron ejecutando el pipeline
   original sobre el split actual. Para refrigeración el CSV entregado sí coincide valor a valor.
   → Pedir al equipo de IA que regenere o retire los CSV obsoletos.
2. **Regla NC/CF dependiente del lote (verificado empíricamente).** La regla neurosimbólica que
   desambigua NON_CONDENSABLES ↔ COND_FOUL_SEVERE usa la *media del lote* de
   `early_P_dis_error`/`T_cond_approach`. El run 1100 (true NON_CONDENSABLES) se diagnostica
   **11 (correcto)** con el split completo pero **2 (incorrecto)** si se envía aislado. El
   diagnóstico de esas dos clases depende de la composición del lote — diseño del equipo de IA
   portado fielmente. → Recomendar una versión por-ciclo de la regla para producción.
3. **Versiones de sklearn de los .pkl.** Artefactos serializados con sklearn 1.9.0
   (aireado/scaler) y 1.8.0 (refrigeración); el entorno de verificación (1.7.0) los carga con
   `InconsistentVersionWarning` pero reproduce exactamente todas las métricas. Los RF además
   predicen con NaN (filas de warm-up), lo que exige **sklearn ≥ 1.4** (requirements del repo:
   `>=1.3.0`). → Fijar sklearn ≥ 1.8 en el despliegue.
4. **Artefacto de 171 MB.** `refrigeracion_model.pkl` no debe commitearse (añadido `/artifacts/`
   a `.gitignore`); en producción irá a `s3://<STORAGE_BUCKET>/artifacts/fixed/ml40_meat_refrigeration_aeration_fault_diagnosis/`.
5. **Anomalía del input de ejemplo de aireado (modo raw).** Con `input_aireado.csv` de ejemplo,
   13/15 runs NORMAL salen ENCOSTRAMIENTO (según `predictions_aireado.csv` entregado por el
   propio equipo). Es coherente con los umbrales calibrados entregados
   (`encostramiento_risk=0.805`) frente a los defaults del código (0.90), o con una distribución
   distinta del input de ejemplo. No afecta al split de test (acc 1.00) ni a los golden cases.
   → Preguntar al equipo de IA qué umbral debe servirse en producción para el modo raw.
6. **CVEs preexistentes** (torch, keras) sin relación con ml40 — gestionar a nivel de repo.

## Estado final

**LISTO PARA PR** (con los hallazgos 1, 2 y 5 marcados para decisión humana / seguimiento con el
equipo de IA — ninguno afecta a la reproducción verificada del comportamiento entregado).
