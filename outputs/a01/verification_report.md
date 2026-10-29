# Verificación — ml1-wine-pest-disease-detection (a01)

- Plugin: `app/plugins/ml1_wine_pest_disease_detection/` · rama `fix/a01-ml1-review-fixes`
- Manifest: `inbox/a01/manifest.yaml` (20 golden cases, `training.supported: true`)
- Fecha: 2026-10-28 · Entorno: WSL2, Python 3.12.14, keras 3.15.1 (`KERAS_BACKEND=torch`), torch 2.13.0,
  ultralytics 8.4.171, scikit-learn 1.5.2, CPU.
- Datos: carpeta `data/` completa del equipo de IA (`data/splits/test_split.csv` con 1470 filas,
  `IMAGENES_RESIZED224`, `YOLO_CROP_CACHE`, `train_rf.csv`/`test_rf.csv`) y los 4 `.joblib` entregados,
  en `inbox/a01/codigo/` (no se commitea).

> **Resumen.** Esta revisión sustituye a la de la integración (PR #24), cuyo «19/20 golden, reproduce
> fielmente el pipeline» era incorrecto: EfficientNetB0 se cargaba sin error pero con la arquitectura
> equivocada. Corregido ese punto, el plugin reproduce **exactamente** las métricas auditadas del equipo
> (accuracy 0.9469, F1 macro 0.8807, F1 weighted 0.9463 sobre las 1470 filas de test) y su propio
> `predictions.csv`. Golden 18/20; los dos fallos son errores del modelo. **Listo para PR**, con las
> acciones humanas de la última sección (subir el RF re-serializado a S3 y dimensionar el pod).

## Checklist técnico

- [x] **flake8** (plugin + tests de ml1): 0 errores.
- [x] **pytest** — ficheros de ml1 + aislamiento + higiene + contrato MLflow juntos: 300 passed. Suite
  completa: ver «Suite completa» al final.
- [x] **pylint** `app/plugins/ml1_wine_pest_disease_detection/`: ver «Suite completa». Avisos ya existentes:
  `line-too-long`, `E1101` de cv2 (falso positivo de la extensión C, igual que ml5), `too-many-locals`.
- [ ] **pip-audit**: no ejecutado en esta revisión (no se añaden dependencias).
- [x] **Arranque real** `MODEL=ml1-wine-pest-disease-detection` vía `main.py` + `TestClient`, artefactos
  reales en `artifacts/ml1_wine_pest_disease_detection/`, sin ningún parche: carga en 6.9 s,
  `/health` global 200, `/models/ml1…/health` 200 `loaded: true`, `/stats` 200.
- [x] `/predict` inline con imagen → 200 `HIBRIDO_ENSAMBLE`; inline sin imagen → 200
  `IMAGEN_NO_DISPONIBLE` + riesgo del RF; batch con el `input.csv` del equipo → 200; batch con el formato
  de `test_split.csv` (columnas extra) → 200, 50/50 filas sin error.
- [x] `/train`: sin cambios de contrato respecto a PR #24 (200 con métricas, 422 sin `mlflow_run_id`, 502 si
  falla MLflow). Ahora sube a MLflow solo el RF de riesgo. No se ha repetido contra un MLflow real.

## Relación con la verificación de PR #29 (main, mismo 28/10/2026)

PR #29 verificó el plugin **antes** de esta corrección contra el mismo corpus. Su parte tabular sigue en pie
(RF reproducido al 5.º decimal contra las 500.000 filas de `SENSORES.csv`). Su parte CNN no: midió 0.9463 con
imagen en bruto y lo dio por fiel al 0.9469 auditado, pero ese 0.9463 es exactamente lo que da el plugin con
EfficientNet roto sobre imagen en bruto; el modelo del equipo, sobre esa misma entrada, da 0.9401 (tabla de abajo).
La cercanía era casual: anular un backbone malo sube la accuracy. Sus F1 por clase (OIDIO_RACIMO 0.69, EMPOASCA
0.83) tampoco son los del modelo real. Corregido en el manifest y en las fichas de `outputs/a01/`.

## Correcciones de esta revisión

| # | Hallazgo | Corrección | Comprobación |
|---|---|---|---|
| 1 | **EfficientNetB0 sin la capa `Rescaling(1/sqrt(std))`** que Keras 3 añade con `weights='imagenet'` (`src/training/train.py:84-85`). Predecía ISARIOPSIS en las 1470 filas; ensemble 0.9599 en vez de 0.9469. | `model_loader._efficientnet_b0_imagenet_topology`: misma topología vía `clone_model`, sin descargar ImageNet. | Salida idéntica (Δ=0.0) a `EfficientNetB0(weights='imagenet')`; test completo = auditado. Tests nuevos en `tests/unit/test_ml1_model_loader.py` (fallan con el código anterior). |
| 2 | RF serializados con scikit-learn 1.8.0; el repo fija 1.5.2. | `rf_riesgo_expert_no_gases.joblib` re-entrenado con 1.5.2 según `train_tabular.py` sobre `train_rf.csv`. | 200 árboles idénticos al entregado; Δ máx. de predicción 5.6e-16 en `test_rf.csv`; carga sin `InconsistentVersionWarning`. |
| 3 | `rf_clase` cargado sin usarse; ambos RF obligatorios para arrancar. | Ya no se carga `rf_clase`; `rf_riesgo` opcional como en `predictor.py:137-144` (sin él, las peticiones sin imagen → 422). | Test `test_missing_rf_riesgo_does_not_block_load`. |
| 4 | Batch: cualquier columna extra hacía fallar la fila. | Solo se leen los sensores, como `predictor.py:317`. | Test `test_predict_batch_ignores_non_sensor_columns` + batch real de 50 filas. |
| 5 | `min_samples_split` (10/5) en `constants.py` y manifest, ausente del original. | Eliminado. | `get_params()` de los RF entregados = `train_tabular.py` (sin cambio en predicciones). |
| 6 | Rutas `inbox/a1`, `outputs/a1`. | Renombrado a `a01` en git, docstrings, `/stats` y manifest. | — |

## Correctitud — test completo (1470 filas de `test_split.csv`)

| Entrada | Accuracy | F1 macro | F1 weighted | Auditado |
|---|---|---|---|---|
| Recortes cacheados del equipo (`Cached_Path`) | **0.9469** | **0.8807** | **0.9463** | 0.9469 / 0.8807 / 0.9463 — idéntico |
| Imagen 224 + recorte YOLO del plugin (ruta de producción) | 0.9483 | 0.8833 | 0.9477 | — |
| Foto a resolución completa (`data/raw/IMAGENES`) | 0.9401 | 0.8787 | 0.9393 | — |

Recall por clase (ruta de producción) frente a la Tabla 4 de la memoria: las 15 clases dentro de ±0.03
(ALTICA 1.000/1.00, BLACK_ROT 0.907/0.90, BOTRYTIS 0.907/0.88, EMPOASCA 0.588/0.59, ERINOSIS 0.943/0.94,
ESCA_HOJA 1.000/1.00, ESCA_RACIMO 0.692/0.69, HEALTHY_HOJA 0.987/0.99, HEALTHY_RACIMO 0.935/0.93,
ISARIOPSIS 1.000/1.00, LOBESIA 0.600/0.60, MILDIU 0.929/0.94, OIDIO 0.879/0.88, OIDIO_RACIMO 0.615/0.62,
RED_MITE 0.983/0.98). Riesgo: MAE 0.189.

Fidelidad pieza a pieza frente al `VidPredictor` original: recorte YOLO, MobileNetV3 y DenseNet121
idénticos bit a bit; cabeza de riesgo idéntica en los 20 golden.

## Correctitud — golden dataset (HTTP, app real)

Los 20 casos están en `test_split.csv` (nombre de fichero + 9 sensores), con etiqueta y riesgo idénticos.

| Caso | Esperado | Obtenido | Confianza | Riesgo esperado | Riesgo obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|---|---|---|
| caso_01 | ALTICA | ALTICA | 0.6941 | 0.071 | 0.3263 | 0.255 | ✅ |
| caso_02 | BLACK_ROT | BLACK_ROT | 0.4841 | 0.077 | 0.3423 | 0.265 | ✅ |
| caso_03 | BOTRYTIS | BOTRYTIS | 0.3997 | 0.016 | 0.3184 | 0.302 | ✅ |
| caso_04 | EMPOASCA | EMPOASCA | 0.3992 | 0.456 | 0.2237 | 0.232 | ✅ |
| caso_05 | ERINOSIS | ERINOSIS | 0.6675 | 0.604 | 0.3233 | 0.281 | ✅ |
| caso_06 | ESCA_HOJA | ESCA_HOJA | 0.8752 | 0.373 | 0.0621 | 0.311 | ✅ |
| caso_07 | ESCA_RACIMO | ESCA_RACIMO | 0.3909 | 0.354 | 0.3408 | 0.013 | ✅ |
| caso_08 | HEALTHY_HOJA | HEALTHY_HOJA | 0.7670 | 0.233 | 0.2113 | 0.022 | ✅ |
| caso_09 | HEALTHY_RACIMO | HEALTHY_RACIMO | 0.6547 | 0.180 | 0.2785 | 0.099 | ✅ |
| caso_10 | ISARIOPSIS | ISARIOPSIS | 0.5306 | 0.029 | 0.3099 | 0.281 | ✅ |
| caso_11 | LOBESIA | LOBESIA | 0.3954 | 0.227 | 0.7934 | **0.566** | ❌ riesgo |
| caso_12 | MILDIU | MILDIU | 0.6735 | 0.133 | 0.2812 | 0.148 | ✅ |
| caso_13 | OIDIO | **HEALTHY_HOJA** | 0.4156 | 0.041 | 0.2290 | 0.188 | ❌ clase |
| caso_14 | OIDIO_RACIMO | OIDIO_RACIMO | 0.3960 | 0.059 | 0.2085 | 0.150 | ✅ |
| caso_15 | RED_MITE | RED_MITE | 0.3674 | 0.524 | 0.3282 | 0.196 | ✅ |
| caso_16 | MILDIU | MILDIU | 0.6703 | 0.053 | 0.1373 | 0.084 | ✅ |
| caso_17 | OIDIO | **OIDIO_RACIMO** | 0.3113 | 0.049 | 0.2773 | 0.228 | ❌ clase |
| caso_18 | BLACK_ROT | BLACK_ROT | 0.3819 | 0.099 | 0.3399 | 0.241 | ✅ |
| caso_19 | BOTRYTIS | BOTRYTIS | 0.5668 | 0.013 | 0.2399 | 0.227 | ✅ |
| caso_20 | ESCA_HOJA | ESCA_HOJA | 0.7557 | 0.246 | 0.1318 | 0.114 | ✅ |

Tolerancia usada: clase exacta; riesgo 2×MAE de la memoria (0.189) = 0.378.
Resultado: **18/20 clase**, **19/20 riesgo**.

- **caso_13 y caso_17** son errores del modelo, no de la integración: el ensemble con los pesos del equipo da
  la misma clase, y el plugin reproduce el test completo exacto. OIDIO/OIDIO_RACIMO son de las clases con peor
  recall de la memoria. caso_13 no aparecía antes porque el 19/20 anterior se midió con EfficientNet roto.
- **caso_11** y, en general, el riesgo: la tolerancia 2×MAE deja pasar casi cualquier valor. Con el test
  completo, R²=0.076 y una constante (la mediana) da MAE 0.183, mejor que el modelo (0.189). El riesgo
  sintético depende de la clase, pero la cabeza desacoplada solo ve sensores que no dependen de ella
  (`scripts/data/create_dataset.py`). Es un límite del modelo; documentado en `known_issues` y debe ir a la
  ficha funcional.

`predictions.csv` del equipo (`data/processed/to_predict/input.csv`), reproducido por batch:

| Fila | Equipo | Plugin |
|---|---|---|
| 0 (Downy Mildew_116) | MILDIU 64.87 % / 0.3585 | MILDIU 0.6488 / 0.3585 |
| 1 (sin imagen) | 0.2985 | 0.2985 |
| 2 (sin imagen) | 0.177 | 0.1770 |

## RF entregados

| | Entregado (sklearn 1.8.0) | Re-serializado (1.5.2) | Manifest |
|---|---|---|---|
| RF clase: accuracy / F1 macro (`test_rf.csv`) | 0.11593 / 0.1180 | ya no se sirve | 0.11593 / 0.1180 |
| RF riesgo: MAE / R² | 0.1881 / 0.205 | idéntico (Δ ≤ 5.6e-16) | 0.188 |
| `get_params()` frente a `train_tabular.py` | idénticos | idénticos | — |

## Recursos para el despliegue (medido con la app real)

| Medida | Valor |
|---|---|
| Artefactos (`artifacts/ml1_wine_pest_disease_detection/` = carpeta de S3) | **574 MB** — RF de riesgo 499 MB, DenseNet 32 MB, EfficientNet 28 MB, MobileNet 8 MB, YOLO 6 MB |
| RSS tras importar `main` (torch + keras + ultralytics + registro) | ~1.0 GB |
| Tiempo de carga del modelo | 4.5–6.9 s |
| RSS tras la carga | **2.3 GB** (el modelo aporta ~1.25 GB; el RF de riesgo solo, ~0.7 GB) |
| Tras las primeras peticiones con imagen | 2.9 GB |
| 200 peticiones inline + 3 batch de 50, secuencial | pico 2.9 GB, ~0.64 s por imagen en CPU |
| Lo mismo con 4 peticiones concurrentes | pico **3.2 GB** |

Recomendación: límite de memoria del pod **≥ 4 GiB** (3 GiB tiene riesgo de OOM con concurrencia).

## Pendiente de acción humana

1. **Subir a S3** `artifacts/fixed/ml1_wine_pest_disease_detection/rf_riesgo_expert_no_gases.joblib`
   re-serializado (md5 `8e82c993d2937e46c2cebb7ac8f526e4`), sustituyendo al de sklearn 1.8.0, y retirar de
   esa carpeta `rf_clase_expert_no_gases.joblib` si está. Las credenciales de `.env` de este entorno no son
   válidas (`InvalidAccessKeyId`), así que no se ha podido hacer desde aquí.
2. **Dimensionar el pod** con ≥ 4 GiB de memoria.
3. **Fichas**: siguen sin generarse para a01. La ficha funcional debe reflejar la limitación del riesgo.
4. `/train` contra el MLflow real de la plataforma (no repetido en esta revisión).

## Suite completa

- `pytest tests/unit/ --cov=app`: **1344 passed**, 0 fallos, 0 skips; cobertura total de `app/` 65 %.
  Sin segfault (el fichero nuevo ya no importa `keras` antes que el plugin, así que la sesión usa el backend
  torch de producción).
- `pylint app/plugins/ml1_wine_pest_disease_detection/`: **8.56/10** (8.55 antes de esta revisión).
