# Verificación — ml38-wine-machinery-fault-detection (a38, DNLS-VINO)

- **Plugin**: `app/plugins/ml38_wine_machinery_fault_detection/` · `MODEL_ID` `ml38-wine-machinery-fault-detection`
- **Manifest**: `inbox/a38/manifest.yaml` (21 golden cases, 15 known issues)
- **Rama**: `feature/model-a38-integration` (sin commitear, pendiente de revisión humana)
- **Fecha**: 2026-11-17
- **Entorno**: WSL Ubuntu, Python 3.12.3, venv `itacyl-a38` alineado con `requirements.txt` del repo
  (scikit-learn 1.5.2, keras 3.15.1, tensorflow 2.21.0, torch 2.13.0+cpu, mlflow 3.16.1; `pip check` limpio).
  El servidor corre Keras con **backend torch** (lo fija ml1 al importarse desde `registry.py`).
- **Artefactos servidos**: los 4 `.pkl` del equipo, bit a bit (md5 en el manifest), copiados a
  `artifacts/ml38_wine_machinery_fault_detection/`. **Falta subirlos a
  `s3://<STORAGE_BUCKET>/artifacts/fixed/ml38_wine_machinery_fault_detection/`**: es una acción
  operativa del responsable; el agente no escribe en S3.

## Checklist técnico

- [x] **flake8**: 0 errores en el código de ml38 (plugin, tests, registry, conftest, evidencias).
  El comando del pipeline sobre todo el repo da exit 1, pero no por ml38:
  - 2 errores ya presentes en `main` (`outputs/a08/.../generar_datos_prueba.py` E128,
    `outputs/a26/.../generar_datos_prueba.py` E741). No se han tocado.
  - Los errores dentro de `inbox/a38/codigo/` son del código del equipo de IA, que no se commitea.
  - Detalle en `evidence/flake8_a38.txt`.
- [x] **pytest**: `tests/unit/` → **1534 passed, 44 skipped, 0 failed**, cobertura total `app` 66 %.
  - ml38: 28 tests, todos verdes con backend torch y con backend tensorflow (ejecutados por
    separado). Incluyen el test golden contra los artefactos reales, la equivalencia de la pérdida
    física con la fórmula original y un `train()` real con MLflow simulado.
  - Cobertura de ml38: `inference` 100 %, `training` 94 %, `preprocessing` 90 %, `plugin` 77 %;
    el resto del plugin, 95-100 %.
  - Los tests de regresión transversales también pasan con ml38: higiene de descargas de MLflow y
    ausencia de estado en `self`. ml38 se ha añadido a `test_user_model_isolation.py`.
  - Detalle en `evidence/pytest_a38_final.txt`.
  - La primera pasada de la suite completa sacó 2 fallos, ya corregidos:
    1. Un test mío mal planteado por fuga espectral.
    2. Un `TypeError` real en la pérdida física cuando se llama con arrays numpy bajo backend
       torch. La pérdida ahora convierte sus entradas a tensores del backend activo.
- [x] **pylint** `app/plugins/ml38_wine_machinery_fault_detection/ --disable=import-error`: **10.00/10**
  (también el fichero de tests).
- [x] **pip-audit** `-r requirements.txt`: **sin CVEs nuevas**. Solo aparece `setuptools==80.9.0`
  (PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w), un pin que ya está en `main`. El plugin **no añade
  dependencias**. `pyarrow`, necesario para leer parquet, llega por `mlflow` (KI-15). Detalle en
  `evidence/pipaudit_req.txt`.
- [x] **Arranque local** (`MODEL=ml38-wine-machinery-fault-detection uvicorn main:app`):
  - `/health` global → `{"ml38-…": true}` y `/models/ml38-…/health` → `loaded: true`.
  - El log avisa una vez por fichero de que los `.pkl` vienen de scikit-learn 1.7.2 (KI-03).
- [x] **/stats** → 200, con las métricas declaradas (OOF/CV). Con `mlflow_run_id` añade el bloque
  `mlflow` con los params y métricas del run.
- [x] **/predict**: inline (24 features) 200; raw (pareja de CSV) 200; batch CSV 200; batch parquet 200.
- [x] **/train** → **200 con `TrainResponse`** (`training.supported: true`):
  - Reentreno completo sobre las 877 filas: 23 s de entrenamiento, 26 s la petición.
  - El modelo se guarda **solo** en el run de MLflow (backend sqlite local de prueba):
    `model/{family_rf,family_scaler,severity_models,severity_scalers}.pkl`.
  - Predecir con ese `mlflow_run_id` usa el modelo del usuario (200).
  - El modelo base sigue sirviendo exactamente lo mismo después del reentreno.

### Comportamiento de errores (servidor real)

| Petición | Esperado | Obtenido |
|---|---|---|
| raw `Motor-4` al 50 % (configuración no documentada) | 422 `UnsupportedOperatingConfigurationError` | 422 ✅ |
| raw `motor: Motor-1` | 422 (schema) | 422 ✅ |
| raw `window_index: 99` | 422 `FeatureContractError` | 422 ✅ |
| batch sin la columna `vib_band_low` | 422 `FeatureContractError` | 422 ✅ |
| batch con un NaN en una fila | 200; esa fila con `error` y el resto predichas | 200 ✅ (fila 1: `Valores vacíos o no numéricos: ['vib_rms']`) |
| `/train` sin `mlflow_run_id` / con `"  "` | 422 | 422 / 422 ✅ |
| `/train` sin columna requerida | 400 | 400 ✅ |
| `/predict` con un run de MLflow vacío | 422 `UserModelUnavailableError` (sin caer al modelo base) | 422 ✅ |

## Correctitud (golden dataset)

Los 21 casos del manifest se lanzaron como `/predict` inline contra el servidor real
(`evidence/verify_ml38.py`; resultados completos en `evidence/ml38_verification_results.json`).
Origen de los casos: `data/splits/fold_assignments.csv` + `feature_dataset.parquet` (inputs
reales) y `data/predictions/predictions.csv` (salida del modelo final entregado). La selección es
reproducible con `evidence/select_golden.py`, que regenera el bloque del manifest idéntico.

**Tolerancia**:
- `family`, `severity` y `severity_label`: coincidencia exacta.
- `severity_confidence`: |Δ| ≤ **0,05**. Es el **valor por defecto** del skill (5 % de la escala
  de probabilidad), porque `metrics_reported` solo tiene accuracy y F1, no un error de regresión
  del que derivar una tolerancia. Se señala explícitamente.

| Caso | sample_id (fold) | Condición real | Esperado (familia / severidad / confianza) | Obtenido | Δ confianza (abs) | ¿OK? |
|---|---|---|---|---|---|---|
| caso_001 | 609 (f1) | align angular 2 — Motor-4@70 | alignment / significativo / 0.719051 | alignment / significativo / 0.719051 | 0.0e+00 | ✅ |
| caso_002 | 621 (f3) | align angular 5 — Motor-4@70 | alignment / significativo / 0.771486 | alignment / significativo / 0.771486 | 0.0e+00 | ✅ |
| caso_003 | 644 (f5) | align combination 4 — Motor-4@70 | alignment / significativo / 0.995969 | alignment / significativo / 0.995969 | 0.0e+00 | ✅ |
| caso_004 | 15 (f1) | bearing bpfi 3 — Motor-2@100 | bearing / significativo / 0.981412 | bearing / significativo / 0.981412 | 0.0e+00 | ✅ |
| caso_005 | 0 (f3) | bearing bpfi 1 — Motor-2@100 | bearing / leve / 0.036167 | bearing / leve / 0.036167 | 3.7e-09 | ✅ |
| caso_006 | 10 (f5) | bearing bpfi 2 — Motor-2@100 | bearing / significativo / 0.974504 | bearing / significativo / 0.974504 | 1.1e-16 | ✅ |
| caso_007 | 678 (f1) | cavitation discharge 1 — Motor-4@70 | cavitation / leve / 0.124776 | cavitation / leve / 0.124776 | 7.5e-09 | ✅ |
| caso_008 | 683 (f3) | cavitation discharge 2 — Motor-4@70 | cavitation / significativo / 0.767227 | cavitation / significativo / 0.767227 | 6.0e-08 | ✅ |
| caso_009 | 688 (f2) | cavitation discharge 3 — Motor-4@70 | cavitation / significativo / 0.945706 | cavitation / significativo / 0.945706 | 6.0e-08 | ✅ |
| caso_010 | 64 (f1) | broken rotor bar — Motor-2@100 | electrical / leve / 0.004450 | electrical / leve / 0.004450 | 4.7e-10 | ✅ |
| caso_011 | 201 (f3) | stator short 1 — Motor-2@100 | electrical / leve / 0.033683 | electrical / leve / 0.033683 | 3.7e-09 | ✅ |
| caso_012 | 601 (f5) | stator short 2 — Motor-2@75 | electrical / significativo / 0.755249 | electrical / significativo / 0.755249 | 0.0e+00 | ✅ |
| caso_013 | 69 (f1) | healthy 1 — Motor-2@100 | healthy / sin fallo / null | healthy / sin fallo / null | — | ✅ |
| caso_014 | 105 (f3) | healthy 2 — Motor-2@100 | healthy / sin fallo / null | healthy / sin fallo / null | — | ✅ |
| caso_015 | 135 (f5) | healthy 3 — Motor-2@100 | healthy / sin fallo / null | healthy / sin fallo / null | — | ✅ |
| caso_016 | 172 (f1) | impeller 2 — Motor-2@100 | mechanical_other / significativo / 0.983082 | mechanical_other / significativo / 0.983082 | 6.0e-08 | ✅ |
| caso_017 | 175 (f3) | impeller 3 — Motor-2@100 | mechanical_other / significativo / 0.937330 | mechanical_other / significativo / 0.937330 | 0.0e+00 | ✅ |
| caso_018 | 179 (f5) | loose foot motor — Motor-2@100 | mechanical_other / significativo / 0.999475 | mechanical_other / significativo / 0.999475 | 0.0e+00 | ✅ |
| caso_019 | 852 (f1) | unbalance motor 5 — Motor-4@70 | unbalance / significativo / 0.911190 | unbalance / significativo / 0.911190 | 0.0e+00 | ✅ |
| caso_020 | 842 (f3) | unbalance motor 3 — Motor-4@70 | unbalance / significativo / 0.978215 | unbalance / significativo / 0.978215 | 1.1e-16 | ✅ |
| caso_021 | 857 (f5) | unbalance motor 6 — Motor-4@70 | unbalance / significativo / 0.987687 | unbalance / significativo / 0.987687 | 1.1e-16 | ✅ |

Tolerancia usada: familia/severidad exactas; `severity_confidence` |Δ| ≤ 0,05 (**default del skill**, no
derivable de `metrics_reported`). Diferencia máxima observada: **6,0e-8**.
**Resultado: 21/21 casos dentro de tolerancia.**

### Evidencia adicional (no sustituye a los golden cases)

- **Paridad con todo el dataset**: `/predict` batch con las 877 filas (CSV y parquet) da 877/877
  familias, 877/877 severidades, el mismo patrón de nulos y |Δ confianza| máx. 1,2e-7 frente a
  `predictions.csv`.
- **Independencia del backend**: con los `.pkl` cargados fuera del servidor
  (`evidence/artifact_backend_parity.py`), torch y tensorflow dan el mismo resultado (877/877).
- **Modo raw — comprobación de referencia** (`reference_checks.raw_example_entrega`, no es un golden
  case): el ejemplo de la entrega (Motor-4, 70 %) reproduce **exactamente** el JSON del equipo
  (`mechanical_other`, severidad 0, confianza 0,015418630093336105, Δ = 0).
  - La respuesta lleva además 2 avisos en `input_warnings`: las señales tienen 4096 muestras y el
    modelo se entrenó con ventanas de 240000 y 300000 (KI-05).
  - La condición real del ejemplo es *align angular 3* (familia alignment). El modelo falla aquí
    por la longitud de la señal, no por la integración.
- **Fidelidad del reentrenamiento** (`evidence/family_cv_parity.py`):
  - En el mismo entorno, el port del plugin y el `train_family_model` original del equipo dan
    métricas idénticas (diferencia < 1e-12), y los folds GroupKFold coinciden con
    `fold_assignments.csv`.
  - El `/train` real da CV de familia macro-F1 **0,6064** y accuracy **0,6168**, frente a los
    **0,6070 / 0,6180** de la Tabla 9.
  - Causa: solo el pin `scikit-learn==1.5.2` del repo. Con la 1.7.2 del equipo, su propio código
    da 0,607003 exacto (KI-14).
  - Los MLP de severidad varían entre ejecuciones, como declara la memoria (§6.3). En este
    reentreno: electrical 0,20 de macro-F1 frente a 0,0 entregado; alignment 0,49 frente a 0,37.

## Hallazgos que requieren decisión o seguimiento (no bloquean la integración)

1. **Modo raw con señales de longitud distinta a la de entrenamiento (KI-05)**. Hoy el plugin
   predice y avisa en `input_warnings`, no rechaza. Decidir si debe dar 422.
2. **Canal de la señal (KI-06)**. El modelo solo ha visto el canal 1 de vibración y la corriente
   de fase 1. La memoria sugiere que vale cualquier `chN`, incluidos los de tensión. El plugin no
   puede validarlo a partir de un CSV; está documentado en `/stats` y en el DTO.
3. **Pin de scikit-learn (KI-03, KI-14)**. Los `.pkl` son de la 1.7.2 y el repo fija la 1.5.2 por
   el servicio de explicabilidad. Hoy la inferencia es idéntica, pero los reentrenos difieren
   ligeramente y la compatibilidad de pickles entre versiones no está garantizada.
4. **Rendimiento intrínseco del modelo (KI-08)**, no de la integración:
   - Familia electrical: F1 OOF 0,17.
   - Severidad electrical: CV 0,0 en la entrega.
   - Severidad clase 0: F1 0,21.
   - Accuracy y macro-F1 OOF ≈ 0,60. Es el objetivo que fija la memoria, pero modesto.
5. **Dominio (KI-10)**. Banco de ensayo NLN-EMP, no maquinaria de bodega. El KPI CU02 (+15 % de
   vida útil) **no está demostrado: requiere validación con datos reales antes de desplegar**.
6. **S3**: copiar los 4 artefactos a `artifacts/fixed/ml38_wine_machinery_fault_detection/` antes
   de desplegar.

## Estado final

**LISTO PARA PR** en lo técnico: checklist en verde, 21/21 golden dentro de tolerancia y paridad
877/877. **Requiere revisión humana** de los puntos 1-3 de arriba antes de abrir el PR, y de los
puntos 4-5 antes de cualquier uso operativo. No se ha abierto PR ni hecho merge.
