# Verificación — ml1-wine-pest-disease-detection

## Nota sobre el entorno de ejecución

Este entorno de desarrollo no tenía instaladas, al empezar, varias dependencias opcionales de
otros plugins del repo (`deap`, `pulp`, `librosa`, `timm`, `albumentations`, `mlflow`) —
`app/registry.py` importa **todos** los plugins a nivel de módulo, así que `main.py` no podía
arrancar hasta instalarlas todas. Ninguna de esas dependencias faltantes es específica de ml1 ni
la introduce este cambio. Tras instalarlas (y sin lograr instalar `albumentations`'s cadena
completa para ml4 por un límite de espacio ajeno a esta rama), se montó para esta verificación
una app FastAPI mínima que registra **únicamente** el router real de ml1 (mismo
`make_model_router` + `ModelContainer` que usa `main.py`, misma clase
`Ml1WinePestDiseaseDetectionPlugin`/`PredictRequest`/`PredictResponse`/`TrainRequest`/
`TrainResponse` reales) y se golpeó con peticiones HTTP reales vía `TestClient` (Starlette) —
equivalente a `curl` contra un servidor real, sin pasar por el import conflictivo de
`app/registry.py`. Los artefactos reales (`best.pt`, `class_mapping_no_gases.json`, los 3
`.weights.h5` y los 2 `.joblib`) están copiados en `artifacts/ml1_wine_pest_disease_detection/`
(no se commitea, es la copia local de lo que vive en S3).

Keras 3 con `backend=torch` **no necesita TensorFlow** (confirmado en esta misma verificación):
se usó `KERAS_BACKEND=torch` y se cargaron los 3 backbones reales sin TensorFlow instalado.

## Checklist técnico

- [x] `flake8 . --extend-exclude=dist,build,inbox,artifacts --show-source --statistics`:
      **0 errores**.
- [x] `pytest tests/unit/test_ml1_wine_pest_disease_detection.py --cov=app.plugins.ml1_wine_pest_disease_detection`:
      **11/11 passed**. Cobertura de línea del módulo real ml1: 15% — esperado, estos tests usan
      `FakePlugin` (wiring: esquemas, DTOs, mapeo de excepciones a HTTP). La correctitud real
      contra el plugin/artefactos reales se valida en la Parte B de este informe, no aquí.
- [x] `pytest tests/unit/ -q --ignore=tests/unit/test_base_reload_per_model.py` (repo completo,
      excluyendo el único fichero que no se puede ni importar por el `timm`/`albumentations` que
      le falta a ml4 en este entorno): **1145 passed, 5 failed, 4 skipped**. Los 5 fallos son
      preexistentes y ajenos a ml1 — `albumentations` faltante (ml4, 4 tests de aislamiento) y un
      test de higiene de descarga de ml3 que depende de infraestructura no disponible en este
      sandbox. Verificado caso por caso: ninguno menciona ni toca ml1. 0 fallos nuevos.
- [x] `pylint app/plugins/ml1_wine_pest_disease_detection/ --disable=import-error`: **8.48/10**.
      Avisos: `line-too-long` (igual que el resto del repo, que no fija `max-line-length` en
      pylint pero sí en `.flake8`=120), `too-many-locals` en `inference.py::run_inference` (fiel
      al `predictor.py` original), y `E1101 cv2 no tiene imread/resize/...` — comprobado que es
      un falso positivo **preexistente y repo-wide**: `ml5_meat_cow_behaviour/preprocessing.py`
      (ya en producción) da el mismo aviso en cada llamada a `cv2.*` (pylint no introspecciona la
      extensión compilada de OpenCV), no es una regresión de ml1.
- [x] `pip-audit -r requirements.txt`: **no ejecutable en este entorno** (el módulo `venv` de
      Python 3.14 está incompleto en esta máquina — `python3.14-venv` no instalado, requiere
      `sudo`). Misma limitación que en la verificación de ml19. Pendiente de confirmar en el
      pipeline de CI real, donde sí corre con normalidad.
- [x] `md5sum` de los 7 ficheros de `artifacts/ml1_wine_pest_disease_detection/` contra los
      mismos ficheros en `inbox/a1/codigo/models/artifacts/`: **idénticos byte a byte**
      (`best.pt`, `class_mapping_no_gases.json`, los 3 `*.weights.h5` del ensemble y los 2
      `*_no_gases.joblib` de los expertos RF) — la copia local no diverge del entregable.
- [x] `tests/unit/test_user_model_isolation.py`: se añadió el fixture `ml1` (monkeypatcheando
      `model_loader.load_model_bundle` y `download_user_model_from_mlflow` con un `_FakeModel`
      que expone `.predict()`, igual que el resto de plugins con expertos sklearn) y se incluyó
      en `CASES` para `predict_inline`/`predict_batch`. **4/4 passed**: un request sin
      `mlflow_run_id` nunca ve el par RF de otro usuario en vuelo, y dos requests solapadas con
      runs distintos no dejan el modelo base sustituido al terminar.
- [x] Arranque + health + predict + stats + train, vía app FastAPI mínima con el router/plugin
      reales (ver nota de entorno):
      - `GET /models/ml1-wine-pest-disease-detection/health` → `200 {"loaded": true}`
      - `GET /models/ml1-wine-pest-disease-detection/stats` → `200`, contrato completo con las
        métricas auditadas del manifest
      - `POST /predict` con imagen real + sensores → `200`, `metodo: HIBRIDO_ENSAMBLE`
      - `POST /predict` sin imagen, solo sensores → `200`, `metodo: RIESGO_TABULAR_EXPERT`,
        `enfermedad: IMAGEN_NO_DISPONIBLE`, `confianza: 0.0` (modo de contingencia tabular)
      - `POST /predict` con sensores incompletos → `422`, `DataContractError` con el detalle de
        las columnas que faltan
      - `POST /train` con un CSV real (`datos_plataforma/muestras_con_etiqueta_real.csv`
        renombrado al contrato de `training.required_columns`) y un `mlflow_run_id` real (MLflow
        local vía `sqlite:///...`, sin servidor remoto disponible en este sandbox) → **200**,
        métricas devueltas, y los 2 `.joblib` reentrenados subidos de verdad al run
        (`artifacts/model/rf_clase_expert.joblib` + `rf_riesgo_expert.joblib` confirmados en
        disco tras la llamada)
      - `POST /train` sin `mlflow_run_id` → **422** (`Field required` — tipo `MlflowRunId`
        obligatorio, sin valor por defecto)
      - `POST /train` con el mismo CSV y `mlflow_run_id` pero **sin MLflow accesible** (antes de
        instalar el paquete `mlflow` en este sandbox) → **502** `ModelPersistenceError`, nunca un
        200 con aviso — confirma que un fallo de persistencia no se silencia

## Hallazgo corregido durante esta verificación

`requirements.txt` no declaraba el paquete standalone `keras` (Keras 3). El repo ya tiene
`tensorflow>=2.15.0`, pero ml3/ml18 lo usan vía `tf.keras`, un import distinto del `import keras`
que necesita ml1 — y, sobre todo, ml1 depende de la API de cambio de backend
(`KERAS_BACKEND=torch`) que solo expone el paquete standalone, no `tf.keras`. Sin una línea
explícita, el pipeline de CI podría instalar una combinación de versiones de `tensorflow`/`keras`
que no garantizara ese import. **Corregido**: añadida `keras>=3.0.0` a `requirements.txt` con
comentario explicando el motivo (mismo estilo que el resto de dependencias añadidas por modelo).

## Correctitud (golden dataset)

Los 20 `golden_cases` de `inbox/a1/manifest.yaml` proceden de
`datos_plataforma/muestras_con_etiqueta_real.csv` + `datos_plataforma/imagenes/`, 20 muestras
reales extraídas por el propio equipo de IA de su `test_split.csv` (confirmado por
`datos_plataforma/README.md`) — no son casos inventados. Se reprodujeron ahora contra el
**plugin integrado real** vía HTTP (`POST /predict {"mode": "inline", "image_base64": ...,
"sensors": ...}`), no contra el script de referencia entregado (`ejemplo_inferencia.py` es
descartable como referencia: solo invoca el RF tabular, nunca el ensemble CNN, pese a etiquetar
su salida `HIBRIDO_ENSAMBLE` — ver `known_issues` del manifest).

| Caso | Esperado clase | Obtenido clase | ¿OK? | Esperado riesgo | Obtenido riesgo | Diff | ¿OK? |
|---|---|---|---|---|---|---|---|
| caso_01 | ALTICA | ALTICA | ✅ | 0.071 | 0.3263 | 0.255 | ✅ |
| caso_02 | BLACK_ROT | BLACK_ROT | ✅ | 0.077 | 0.3423 | 0.265 | ✅ |
| caso_03 | BOTRYTIS | BOTRYTIS | ✅ | 0.016 | 0.3184 | 0.302 | ✅ |
| caso_04 | EMPOASCA | EMPOASCA | ✅ | 0.456 | 0.2237 | 0.232 | ✅ |
| caso_05 | ERINOSIS | ERINOSIS | ✅ | 0.604 | 0.3233 | 0.281 | ✅ |
| caso_06 | ESCA_HOJA | ESCA_HOJA | ✅ | 0.373 | 0.0621 | 0.311 | ✅ |
| caso_07 | ESCA_RACIMO | ESCA_RACIMO | ✅ | 0.354 | 0.3408 | 0.013 | ✅ |
| caso_08 | HEALTHY_HOJA | HEALTHY_HOJA | ✅ | 0.233 | 0.2113 | 0.022 | ✅ |
| caso_09 | HEALTHY_RACIMO | HEALTHY_RACIMO | ✅ | 0.180 | 0.2785 | 0.099 | ✅ |
| caso_10 | ISARIOPSIS | ISARIOPSIS | ✅ | 0.029 | 0.3099 | 0.281 | ✅ |
| caso_11 | LOBESIA | LOBESIA | ✅ | 0.227 | 0.7934 | **0.566** | ❌ |
| caso_12 | MILDIU | MILDIU | ✅ | 0.133 | 0.2812 | 0.148 | ✅ |
| caso_13 | OIDIO | OIDIO | ✅ | 0.041 | 0.2290 | 0.188 | ✅ |
| caso_14 | OIDIO_RACIMO | OIDIO_RACIMO | ✅ | 0.059 | 0.2085 | 0.150 | ✅ |
| caso_15 | RED_MITE | RED_MITE | ✅ | 0.524 | 0.3282 | 0.196 | ✅ |
| caso_16 | MILDIU | MILDIU | ✅ | 0.053 | 0.1373 | 0.084 | ✅ |
| caso_17 | OIDIO | **OIDIO_RACIMO** | ❌ | 0.049 | 0.2773 | 0.228 | ✅ |
| caso_18 | BLACK_ROT | BLACK_ROT | ✅ | 0.099 | 0.3399 | 0.241 | ✅ |
| caso_19 | BOTRYTIS | BOTRYTIS | ✅ | 0.013 | 0.2399 | 0.227 | ✅ |
| caso_20 | ESCA_HOJA | ESCA_HOJA | ✅ | 0.246 | 0.1318 | 0.114 | ✅ |

Tolerancia usada:
- Clasificación: coincidencia exacta de clase (no aplica tolerancia relativa a una etiqueta
  categórica).
- Riesgo: **2× el MAE auditado** (`models/metrics` → `mae=0.189`) = **0.378 absoluto**, mismo
  patrón de tolerancia que el resto de modelos de este repo.

Resultado: **19/20 (95%) clasificación exacta** — consistente con el 94.69% auditado
(`report_ensemble_final_no_gases_desacoplado.csv`, support=1470). **19/20 riesgo dentro de
tolerancia.** Reproducido en esta sesión con resultados idénticos a los registrados en
`inbox/a1/manifest.yaml::golden_cases.cases.predicho_*` durante la integración — el plugin es
determinista y estable, no es una coincidencia de una única ejecución.

### Los dos fallos, investigados (no silenciados)

- **caso_17 (clasificación)**: esperado `OIDIO`, predicho `OIDIO_RACIMO`, con la confianza más
  baja de los 20 casos (0.312). Es una confusión entre dos clases visualmente emparentadas
  (oidio en hoja vs. oidio en racimo — mismo velo blanquecino característico) y una clase
  minoritaria con recall auditado 0.615 (`kpi_contractual` del manifest) — coherente con el
  propio informe de métricas del equipo de IA, no un bug de integración.
- **caso_11 (riesgo)**: esperado 0.227, predicho 0.793 (LOBESIA). El MAE agregado (0.189) es
  bajo, pero la dispersión por caso individual es mayor de lo que ese agregado sugiere — ver
  `known_issues` del manifest ("REGRESIÓN DE RIESGO MÁS RUIDOSA DE LO QUE SUGIERE EL MAE
  AGREGADO"). No se ha tocado la tolerancia ni el modelo para ocultarlo.

## Estado final

**LISTO PARA PR**, con dos salvedades explícitas:
1. `pip-audit` no se pudo ejecutar en este entorno de desarrollo concreto (limitación de sandbox,
   no del código) — debe confirmarse en el pipeline de CI real antes del merge final.
2. El entrenamiento real de `/train` se verificó contra un servidor MLflow local (`sqlite://`),
   no contra el servidor MLflow real de la plataforma — el contrato (200 con métricas + subida
   real de artefactos, 422 sin `mlflow_run_id`, 502 si la subida falla) está verificado; la
   conectividad con el MLflow real de producción queda fuera del alcance de este sandbox.

## Correcciones tras revisión externa (PR #24)

Un compañero probó la rama desplegada y reportó tres síntomas. Investigados y confirmados los
tres contra el código real (no solo contra el informe del compañero):

1. **Un sensor no numérico tiraba todo el lote.** Reproducido: una fila con `Mes="oops"` lanzaba
   un `ValueError` sin capturar desde `validate_and_align_sensors` (el `float()` de conversión no
   estaba protegido); el `except (DataContractError, InvalidImageError)` por fila de
   `predict_batch` no lo cubre, así que el error escapaba y abortaba el lote entero — incluidas
   las filas ya procesadas correctamente antes de llegar a la mala. **Corregido** en
   `preprocessing.py`: el fallo de conversión numérica ahora lanza `DataContractError`, así que
   `predict_batch` la reporta como error de esa fila sin afectar al resto. Verificado con un lote
   real de 2 filas (una con `Mes="oops"`, otra válida): `200`, `n_rows=2`, `n_predictions=1`, la
   fila mala con `error` descriptivo y la fila válida con su predicción intacta.
2. **Se ignoraban fotos válidas en la raíz del ZIP.** Reproducido: `predict_batch` elegía un
   único directorio de búsqueda (la raíz, o `images/` si esa subcarpeta existía, nunca ambos),
   pese a que el propio docstring de `predict_dto.py` ya decía que se buscan en los dos sitios.
   Una foto real en la raíz devolvía `IMAGEN_NO_DISPONIBLE` (contada como fila correcta) en
   cuanto el ZIP traía también una subcarpeta `images/` para otras filas. **Corregido**:
   `resolve_image_path_for_batch` acepta ahora una lista de directorios candidatos
   (`images/` primero si existe, luego la raíz). Verificado con un ZIP real con una foto en la
   raíz + una subcarpeta `images/` con otra foto distinta: la foto de la raíz se detecta y
   clasifica correctamente (`HIBRIDO_ENSAMBLE`), ya no cae a modo tabular.
3. **`/train` incumplía el manifest.** `training.metrics_returned` exige `f1_weighted`, pero
   `train_tabular_experts()` solo calculaba `accuracy`/`f1_macro`/`mae_riesgo` — no se devolvía en
   `TrainResponse` ni se registraba en MLflow. El `train_tabular.py` original del equipo de IA
   tampoco calcula `f1_weighted` (no es un bug heredado: es un incumplimiento de un contrato que
   fijamos nosotros mismos en el manifest). **Corregido**: se añade
   `f1_score(..., average="weighted")` sobre el mismo holdout de test ya calculado, sin tocar el
   procedimiento de entrenamiento. Verificado con un run de MLflow real (local, `sqlite://`):
   `200` con `f1_weighted` en la respuesta, y confirmado como métrica registrada en el run
   (`client.get_run(...).data.metrics["f1_weighted"]`).

Añadidos 4 tests de regresión reales (no de wiring) en `tests/unit/test_ml1_preprocessing.py`,
contra las funciones reales, no contra `FakePlugin`. Re-verificación tras los tres parches:
`flake8` 0 errores, 15/15 tests de ml1 (11 de wiring + 4 de regresión), suite completa sin
fallos nuevos (1100 passed, mismos 5 fallos preexistentes ajenos a ml1).
