# Verificación — lacteo-xai-puntos-criticos-control (m48-dnsl-fallas-maquinaria-pasteurizado)

Leyenda: **PASS** = ejecutado y correcto · **FAIL** = ejecutado y fallido · **BLOCKED** = no completable por dependencia externa/infraestructura · **NOT VERIFIED** = no validado completamente · **KNOWN ISSUE** = limitación o discrepancia documentada.

Fecha de verificación: 2026-10-05 (v1.1 del equipo) y **re-verificación 2026-10-07 con la v1.3** (artefactos del Entregable 47 v2.0). Donde una sección dice «v1.3» el resultado se ha vuelto a ejecutar con los artefactos nuevos; el resto procede de la verificación inicial y se indica cuando no se ha repetido. Este informe no abre PR, no hace commit ni sube artefactos.

## Identificación

| Campo | Valor |
|---|---|
| Plugin | `app/plugins/m48_dnsl_fallas_maquinaria_pasteurizado/` |
| `model_id` runtime / prefijo | `m48-dnsl-fallas-maquinaria-pasteurizado` / `/models/m48-dnsl-fallas-maquinaria-pasteurizado` |
| Manifest | `inbox/a48/manifest.yaml` (model_id de manifest: `lacteo-xai-puntos-criticos-control`) |
| Memoria / entrega | Entregable 48 **v1.3** (10/07/2026), que sustituye a la v1.1 revisada inicialmente (la v1.2 existe, pero no se recibió) |
| Tipo | Clasificación multietiqueta 4 componentes × 3 estados (DNSL, 1D-CNN) + XAI (Grad-CAM, SHAP opcional, CCP, motor prescriptivo) |
| Relación con m47 | Plugin independiente y autocontenido, con artefactos propios (no depende de los de m47). **Según el equipo de IA**, en la v1.3 los artefactos del 48 son los del Entregable 47 v2.0 (hashes abajo); en las v1.1/v1.2 eran pesos de un entrenamiento anterior. Esa identidad con el 47 viene declarada por el equipo y no se ha comprobado nosotros (los artefactos del plugin m47 no están en este repo). No afecta a la integración. |

## Artefactos

Ubicación local: `artifacts/a48_dnsl_fallas_maquinaria_pasteurizado/` (en `.gitignore`).

- Artefactos **v1.3** (MD5 verificados contra los que declara el equipo):
  - `neurosymbolic_cnn.pth` `5b087a2c16777300521b08e7514fa0e8`
  - `scaler_cnn_dns.pkl` `1149dc17cb04e243b7acc5b4cbacb980` — **regenerado por nosotros con scikit-learn 1.5.2** a partir del entregado (`ddae8685a8571db6502364a43c7f701a`, sklearn 1.6.1); ver Scaler
  - `ts1_mean_train.pkl` `fb5a036109839ee9ffe3fc3e3ec70a5a`
  - `feature_columns.pkl` `517503ee6313305c7ffec69321c92620` (sin cambios)
- `shap_background.npy` (50 ciclos de test ya escalados con el scaler v1.3, 3,4 MB) + `shap_background_meta.json`: **regenerados con los artefactos v1.3** (generados por nosotros; no se incluye el CSV original de 101 MB).
- Los artefactos de la v1.1 (pesos `f79fd471…`) quedan sustituidos localmente; hay copia fuera del repo.
- Estado: carga **PASS** con warning de versión del scaler (ver Scaler). Subida a S3 de los artefactos v1.3: **realizada por el usuario el 2026-10-07** en `s3://xai/artifacts/fixed/a48_dnsl_fallas_maquinaria_pasteurizado/` (ver «Validación de cierre»).

## Entorno

Verificación de referencia: **Python 3.12 con los requirements reales del repo** (el Dockerfile y el pipeline de Bitbucket usan Python 3.12), instalados en un venv aislado (`C:\Users\paahh\venv_m48_py312`) con `pip install -r requirements.txt` resuelto por uv. El entorno global no se modificó.

| Paquete | Versión (entorno de referencia, Python 3.12.15) |
|---|---|
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.5.2 (pin del repo) |
| torch | 2.13.0+cpu (repo: `torch>=2.13.0`) |
| torchvision | 0.28.0+cpu (pin del repo) |
| shap | 0.52.0 (repo: `shap>=0.51.0`) |
| scipy / pydantic / fastapi | 1.18.1 / 2.9.2 / 0.136.1 |

Notas:
- Se usaron wheels CPU de torch (índice CPU de PyTorch); el Docker de producción resolverá el wheel por defecto de PyPI. La versión de torch coincide con la requerida (2.13.0); la variante CPU/CUDA no se ha probado en la imagen real.
- Una primera integración se hizo en un venv Python 3.11.4 (torch 2.5.1+cpu, numpy 1.26.4, pandas 2.2.3, shap 0.42.0). **No es el entorno de referencia**; se usó solo para contrastar el comportamiento de SHAP (ver Resultados SHAP).
- `requirements.txt` raíz: **sin modificar, idéntico al original** (`git diff --quiet requirements.txt` → sin diferencias). Contiene `shap>=0.51.0`; no se añade ningún pin de shap.
- La imagen Docker no se ha construido; la verificación es sobre un venv equivalente, no sobre la imagen.

## Checklist técnico

- [x] **flake8: PASS** — 0 errores en `app/plugins/m48_*`, `tests/unit/test_m48_*`, `predict_model_use_case.py`, `registry.py`. En todo el repo solo queda `tests/conftest.py:1667 E305`, que ya existía antes de m48 (verificado contra la versión sin cambios) y es ajeno a m48. Se corrigieron antes 6 imports sin usar de `plugin.py`.
- [x] **pytest: PASS — 556/556 passed** (`pytest tests/unit/ --cov=app`, Python 3.12, entorno de referencia, 217 s). Incluye los 15 tests nuevos de m48.
- [~] **Cobertura de m48: 39 %** (937 sentencias, 570 sin cubrir). **KNOWN ISSUE.** `plugin.py`, `postprocessing.py` y `mlflow_utils.py` aparecen al 0 % en pytest porque los tests unitarios usan un `FakePlugin` (patrón del repo) y solo validan wiring. `xai.py` 77 %. Esto **no** significa que el plugin real no se haya ejecutado: se ejercitó con scripts y pruebas HTTP contra el plugin real (secciones siguientes), pero esas pruebas no están integradas en pytest ni cuentan para la cobertura.
- [x] **pylint: PASS (sin errores) — 8,14/10.** Solo con mensajes E/W: 9,98/10 (1 `W0718` broad-except en `plugin.py`, mismo patrón que m47; 1 `W0612` corregido después). El resto son convenciones/refactor (líneas largas, docstrings, nº de variables/argumentos en `xai.py`). Referencias: m47 8,50; ml46 8,12.
- [~] **pip-audit: 1 CVE, KNOWN ISSUE preexistente.** `setuptools==80.9.0` (PYSEC-2026-3447, fix 83.0.0). Esa versión está **fijada por el `requirements.txt` del repo** y no la introduce m48. `torch 2.13.0+cpu` y `torchvision 0.28.0+cpu` **no pudieron auditarse** (son builds CPU locales no encontrados en PyPI): **NOT VERIFIED** para estos dos paquetes. Nota: se auditó el entorno instalado con el `pip-audit` del venv, no con `pip-audit -r requirements.txt`, que habría reconstruido un entorno completo.
- [x] **Arranque local + health + predict + stats: PASS.** `MODEL=m48-dnsl-fallas-maquinaria-pasteurizado uvicorn main:app` (Python 3.12, solo m48 cargado). Se usó uvicorn en el puerto 8000 en lugar de `python main.py` (que usa `reload=True`).
- [ ] **`/train` con MLflow real: BLOCKED.** Ver sección Reentrenamiento.

### Pruebas HTTP contra el plugin real (servidor arrancado, curl)

| Prueba | Resultado |
|---|---|
| `GET /health` | **PASS** — `status: ok`, `loaded: true` |
| `GET /stats` | **PASS** — `m48-dnsl-fallas-maquinaria-pasteurizado`, v1.0.0, métricas de referencia del test del equipo |
| `POST /predict` inline (CSV del ciclo 123, `apply_digital_twin=true`, `include_shap=true`) | **PASS** (ejecución con artefactos v1.1 del 05/10) — `[2,0,0,0]`; SHAP FS1 0,2288 / FS1 0,3224 / TS2 0,2639 / TS2 0,3334. Con artefactos v1.3 el HTTP no se ha repetido: se comprobó el mismo plugin por llamada directa (ver Resultados XAI) |
| `POST /predict` batch con XAI (`include_xai`, `n_samples=10`) | **PASS** — 20 ciclos, 7 CCP, 10 ciclos analizados |
| `POST /predict` batch sin XAI | **PASS** — 20 ciclos, `xai: null` |
| `POST /predict` inválido (`include_shap` sin `include_xai`) | **PASS** — HTTP 422 |
| Inline con arrays de sensores (ciclo 123) | **PASS** — `[2,0,0,0]` (vía TestClient sobre el plugin real) |

### Cambio compartido: `app/application/use_cases/predict_model_use_case.py` (+4 líneas)

Propaga `apply_digital_twin`, `include_xai`, `include_shap` y `n_samples` a `predict_batch` solo si el plugin los declara.
- **PASS** — Un plugin con la firma estándar `(data_path, mlflow_run_id)` recibe solo esos dos kwargs, con y sin las opciones en la petición.
- **PASS** — Un plugin que declara las cuatro opciones las recibe con los valores de la petición (comprobado también en batch real de m48 por HTTP).
- **PASS** — Sigue el patrón existente de ml34/ml40 (`inspect.signature`). Ningún otro plugin del repo declara esos nombres de parámetro (búsqueda en `app/plugins/*/plugin.py`), y los 556 tests pasan.
- **KNOWN ISSUE (menor)**: se usa `getattr(request, opt)` sin valor por defecto; si en el futuro un plugin declarase estas opciones sin que su request las tenga, lanzaría `AttributeError`. Hoy solo m48 las declara y su request las define. No se ha refactorizado por indicación.
- Se añadieron además el registro en `app/registry.py` y las fábricas fake de m48 en `tests/conftest.py`.

## Correctitud (golden dataset)

Fuente: 20 ciclos de `test_cycles` (`cycle_splits.json`), seleccionados con semilla 42 para cubrir las 12 parejas (componente, clase); esperado = etiqueta real de `profile.txt` mapeada con `target_mapping.py`. El modelo **no** se ejecutó para elegirlos. Ejecutados vía `predict_batch` del plugin real, `apply_digital_twin=true`, Python 3.12.

Orden de valores: [Fouling, Válvula, Bomba, Acumulador] (0 = Sano, 1 = Warning, 2 = Crítico).

| Caso | Esperado | Obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|
| cycle_1737 | [0, 1, 1, 0] | [0, 1, 1, 0] | 0 | PASS |
| cycle_1087 | [1, 1, 2, 1] | [1, 1, 2, 1] | 0 | PASS |
| cycle_0073 | [2, 0, 0, 0] | [2, 0, 0, 0] | 0 | PASS |
| cycle_1078 | [1, 2, 2, 1] | [1, 2, 2, 1] | 0 | PASS |
| cycle_0840 | [1, 0, 0, 2] | [1, 0, 0, 2] | 0 | PASS |
| cycle_0544 | [2, 1, 1, 1] | [2, 1, 1, 1] | 0 | PASS |
| cycle_1618 | [0, 0, 0, 2] | [0, 0, 0, 2] | 0 | PASS |
| cycle_1116 | [1, 0, 1, 1] | [1, 0, 1, 1] | 0 | PASS |
| cycle_1809 | [0, 2, 2, 1] | [0, 2, 2, 1] | 0 | PASS |
| cycle_0398 | [2, 1, 1, 1] | [2, 1, 1, 1] | 0 | PASS |
| cycle_0281 | [2, 1, 1, 0] | [2, 1, 1, 0] | 0 | PASS |
| cycle_1966 | [0, 1, 2, 1] | [0, 1, 2, 1] | 0 | PASS |
| cycle_0949 | [1, 2, 2, 0] | [1, 2, 2, 0] | 0 | PASS |
| cycle_1612 | [0, 0, 0, 2] | [0, 0, 0, 2] | 0 | PASS |
| cycle_1264 | [1, 1, 1, 1] | [1, 1, 1, 1] | 0 | PASS |
| cycle_1084 | [1, 2, 2, 1] | [1, 2, 2, 1] | 0 | PASS |
| cycle_1832 | [0, 1, 2, 1] | [0, 1, 2, 1] | 0 | PASS |
| cycle_0482 | [2, 2, 2, 1] | [2, 2, 2, 1] | 0 | PASS |
| cycle_0721 | [2, 1, 0, 2] | [2, 1, 0, 2] | 0 | PASS |
| cycle_1730 | [0, 1, 1, 0] | [0, 1, 1, 0] | 0 | PASS |

Tolerancia usada: **igualdad exacta de clase** (las salidas son clases discretas 0/1/2, por lo que no aplica una tolerancia porcentual). Referencia de rendimiento esperada de la memoria/artefactos: exact_match 0,9879 sobre 331 ciclos de test, con errores solo Sano→Warning en Bomba y Acumulador.
Resultado: **20/20 casos dentro de tolerancia — PASS.** Ejecutado con los artefactos v1.1 (2026-10-05) y **repetido con los artefactos v1.3 (2026-10-07, Python 3.12, scikit-learn 1.5.2, shap 0.52.0): 20/20 PASS**; las clases son las mismas y cambian solo las confianzas (la tabla de abajo es la de las clases, idéntica en ambas versiones).

Limitaciones de esta muestra: 20 ciclos no son una validación estadística (con un exact_match de 0,988 se esperaría ≈0,2 fallos en 20 casos); verifica el cableado y el preprocesado, no sustituye las métricas de test del equipo. Los datos de entrada se extrajeron de `data/processed/hydraulic_10hz_raw.csv` (601 filas por ciclo, el modelo trunca a 600). **Sin gemelo digital los resultados no son válidos para datos UCI** (los golden requieren `apply_digital_twin=true`).

## Resultados XAI (artefactos v1.3)

Referencia: ficheros `data/predictions/xai_report_cycle.csv`, `xai_report.csv` y `xai_report_shap.csv` de la **v1.3**, regenerados por el equipo con el modelo del Entregable 47 v2.0. Ejecutado con Python 3.12, scikit-learn 1.5.2, shap 0.52.0, `apply_digital_twin=true`.

| Elemento | Resultado |
|---|---|
| **Local, ciclo 123** (validación, no test): predicción, confianza (0,9604 / 1,0000 / 0,9999 / 0,9890), ventana Grad-CAM (9,1–11,1 / 8,7–10,5 / 37,6–39,9 / 9,0–11,0), intensidad, acción, urgencia, intervalo, riesgo, sensor más relevante e importancia SHAP | **PASS** — las **13 columnas** de `xai_report_cycle.csv` coinciden (numéricas con tolerancia 1e-4) |
| **Global, 50 ciclos de test** (331 ciclos de test como entrada, `n_samples=50`, `include_shap=true`) | **PASS** — los 50 `ciclos_analizados` coinciden; las 6 filas CCP (componente, sensor crítico, importancia, ventanas en muestras y segundos, intensidad, severidad) coinciden; las 28 importancias SHAP coinciden con diferencia máxima **0,0**. Tarda ≈10 min en CPU |
| Grad-CAM sin SHAP (`include_shap=false`) | **PASS** — `sensor_mas_relevante = null`, `riesgo_incluye_shap = false`; riesgo calculado sin el bonus SHAP por diseño |
| Verificación anterior con v1.1 (2026-10-05) | Ciclo 123 coincidía con las referencias de la v1.1; el análisis global **no** se pudo reproducir entonces (ver SHAP) |

## Resultados SHAP

- **PASS (v1.3, shap 0.52.0):** el plugin reproduce exactamente el SHAP local (ciclo 123) y el global (50 ciclos) del equipo.
- **Causa de la discrepancia inicial (resuelta, confirmada por el equipo de IA):** el `requirements.txt` de la v1.1 fijaba `shap==0.42.0` por error (entró en un merge del 09/06/2026; antes decía `shap>=0.42.0`). Los CSV de v1.1 y v1.2 se generaron con shap 0.51.0 y la v1.3 fija `shap==0.51.0`. El equipo confirma que 0.52.0 da salidas idénticas, y nosotros lo hemos comprobado con el plugin. Con shap 0.42.0 (probado por nosotros en la verificación inicial) el SHAP es distinto; por eso **no debe usarse 0.42.0** (el repo exige `shap>=0.51.0`).
- Predicción, confianza, ventana Grad-CAM y acción no dependen de la versión de shap; el riesgo con SHAP incorpora un bonus basado en SHAP y puede variar con la versión.
- **Tablas 1 y 2 de la memoria: RESUELTO.** Según el equipo, las de la v1.1 salieron de una ejecución con un fallo de muestreo (tomaba los 50 ciclos de menor Cycle_ID); la v1.2 lo corrigió con una muestra aleatoria con semilla 42 y la v1.3 regenera tablas, CSV y figuras. Con la v1.3 el análisis global del plugin coincide exactamente con el CSV (arriba). Ya no es un known issue.
- Estabilidad (verificación inicial, v1.1, shap 0.42.0): el sensor top del ciclo 123 fue estable entre 5 semillas salvo en Bomba. No se repitió con v1.3.

## Scaler y carga de artefactos

- **PASS — `scaler_cnn_dns.pkl` regenerado con scikit-learn 1.5.2 (decisión tomada el 2026-10-07).** El entregado en la v1.3 estaba serializado con 1.6.1 y el repo pinea 1.5.2 (`InconsistentVersionWarning`). Se cargó con 1.5.2 y se volvió a serializar con `joblib` bajo 1.5.2: `mean_`, `scale_`, `var_` y `n_samples_seen_` son idénticos (comparación exacta), el nuevo pickle carga **sin warning** (comprobado con `-W error::UserWarning`; el original sí lo emite) y tiene MD5 `1149dc17cb04e243b7acc5b4cbacb980` (el entregado: `ddae8685a8571db6502364a43c7f701a`; copia del original guardada fuera del repo). Con el scaler regenerado se repitieron **20/20 golden PASS** y las **13 columnas del ciclo 123 PASS**; el fondo SHAP no se regeneró porque los valores del scaler son idénticos, y el análisis global (10 min) no se repitió. `requirements.txt` sin cambios. **Hay que informar al equipo de IA** de que el artefacto desplegado difiere del entregado en la serialización (no en los valores).
- **PASS** — `feature_columns.pkl` (28 columnas, orden esperado), `ts1_mean_train.pkl` (45,3110734), `neurosymbolic_cnn.pth` (state_dict, `weights_only=True`), `shap_background.npy` (50, 28, 600) float32.
- Los `.pkl`/`.pth` v1.3 se han comprobado por MD5 frente a los hashes que declara el equipo.

## Reentrenamiento y `/train`

- **`/train` con servidor MLflow real: BLOCKED.** `POST /train` real devuelve **HTTP 500** porque no hay servidor MLflow accesible (`mlflow.mlflow:5000`, error de resolución de nombre). El entrenamiento se ejecuta y falla solo la subida a MLflow. No es un fallo atribuible al plugin, pero **tampoco está demostrado el flujo end-to-end**: falta validarlo con un servidor MLflow real.
- **`/train` con la subida a MLflow sustituida: PASS parcial (solo cableado).** Con `upload_artifacts_to_mlflow` sustituido por una función de prueba, `/train` devuelve HTTP 200 con `TrainResponse` (`exact_match`, `accuracy`, `f1_macro`, `recall_macro`, `n_train`, `n_test`, `training_time_s`) y prepara los 6 artefactos para subir (modelo, scaler, columnas, ts1, fondo SHAP y metadatos). Esto verifica el contrato HTTP y el empaquetado, no la subida real.
- **Smoke test con 21 ciclos: PASS funcional, sin valor estadístico.** Entrena (≈22 épocas, early stopping) y devuelve métricas con n_test = 4. Esas métricas (exact_match 0,25, etc.) **no son métricas de validación del modelo** y no deben citarse como tales.
- **Reentrenamiento completo (2205 ciclos, hasta 300 épocas): NOT VERIFIED.** Se lanzó y se **detuvo manualmente en la época ≈40 de 300**; **no hay métricas de reentrenamiento a escala completa** y no se completaron las 300 épocas. Pendiente de ejecutar cuando se decida.
- El `trainer.py` replica el pipeline y los hiperparámetros del equipo (Adam, lr 0,00222, dropout 0,2022, lambda máx. 3,7585, 300 épocas, paciencia 15, split 70/15/15 por ciclo, augmentation por ruido en train), pero **no se ha comprobado que reproduzca las métricas del equipo**.

## Validación de cierre: S3, plataforma, MLflow real, Docker (intento posterior)

Se intentó validar los puntos pendientes sin modificar código, requirements ni artefactos y sin sustituir infraestructura por mocks. Los resultados anteriores de este informe se conservan; solo cambia lo comprobado aquí.

**Comprobaciones de entorno realizadas (solo lectura):**
- `aws` CLI: **no instalado** en este equipo (`aws: command not found`).
- Variables `STORAGE_BUCKET`, `CUSTOM_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `CUSTOM_REGION`, `MLFLOW_TRACKING_URI`: **ninguna definida** en el entorno; no existe fichero `.env` en el repo (solo `.env.example`) ni `~/.aws`. No se buscaron credenciales fuera de esos lugares.
- Conectividad: `https://s3.datagia-int.es` responde (HTTP 404 en la raíz sin autenticar, es decir, el servidor es alcanzable); `http://mlflow.mlflow:5000` **no resuelve** (HTTP 000, sin conexión).
- `docker`: **no instalado** en este equipo.

| Validación | Estado | Evidencia |
|---|---|---|
| Artefactos en `s3://xai/artifacts/fixed/a48_dnsl_fallas_maquinaria_pasteurizado/` | **PASS** (nombres y tamaños; evidencia del usuario) | Los 6 ficheros esperados están en S3 con los mismos tamaños que los locales (v1.3, scaler regenerado). Evidencia aportada por el usuario (salida de `aws s3 ls/cp --endpoint-url https://s3.datagia-int.es` pegada el 2026-10-07, no ejecutada desde este equipo): antes de la subida el prefijo contenía los 6 ficheros de la v1.1 (05/10/2026; p. ej. `neurosymbolic_cnn.pth` 591482 B y `scaler_cnn_dns.pkl` 1975 B); se subieron los 6 ficheros de `artifacts/a48_dnsl_fallas_maquinaria_pasteurizado/` y el listado posterior (2026-10-07 15:43) muestra 300 / 592347 / 1815 / 3360128 / 585 / 117 bytes, iguales a los locales. Solo se comparan nombres y tamaños; no se comparó MD5/ETag. Desde este equipo sigue sin haber CLI ni credenciales, por lo que no lo hemos ejecutado nosotros. |
| Carga real de artefactos desde S3 | **NOT VERIFIED** | Los artefactos ya están en S3, pero falta arrancar el plugin con `STORAGE_BUCKET` definido y `artifacts/a48_*` local vacío para demostrar la descarga. Análisis estático (no es prueba): `plugin.load()` llama a `ArtifactStore(ARTIFACT_FOLDER_NAME).download_all_if_needed()` solo si `STORAGE_BUCKET` está definido; esto lista `artifacts/fixed/<carpeta>/` en el bucket y descarga a `artifacts/<carpeta>/` los ficheros que falten o cuyo tamaño difiera, y después el plugin carga siempre desde ese directorio local. Como el directorio local ya contiene los 6 ficheros, una ejecución con S3 configurado **no demostraría el origen S3** salvo que se vacíe el directorio local antes. El prefijo descargado incluye `shap_background.npy` y `shap_background_meta.json` porque se baja todo el prefijo. Los artefactos permanecen en disco tras la carga (mismo patrón que m47). |
| E2E plataforma/orquestador → m48 | **NOT VERIFIED** | Lo que sí está verificado (PASS, secciones anteriores): HTTP real contra `main.py` con el registro → router → use case → plugin m48 → XAI/SHAP → respuesta, y la propagación de `include_xai`, `include_shap`, `n_samples` y `apply_digital_twin` sin afectar a plugins con firma estándar. La plataforma real no está integrada con m48: una búsqueda de `m48-dnsl`/`m48_dnsl` en los repos hermanos `retech-lote2-xai-plataforma`, `retech-lote2-xai-explicabilidad` y `retech-lote2-xai-odd-detection` da **0 referencias** (las skills `front-integration`, `explainability-integration` y `odd-integration` no se han aplicado a m48), y además no hay S3 ni MLflow accesibles. |
| `/train` contra MLflow real | **BLOCKED** | `MLFLOW_TRACKING_URI` no definido; el valor por defecto `http://mlflow.mlflow:5000` no resuelve desde este equipo. Sin cambios respecto a la verificación anterior (HTTP 500 por la subida). No se creó ningún run ni se comprobaron parámetros, métricas o artefactos en MLflow. |
| Entrenamiento completo | **NOT VERIFIED** | No existe evidencia real de un entrenamiento completo: ni checkpoint final, ni métricas, ni run de MLflow, ni informe. Solo hay el smoke test de 21 ciclos y un entrenamiento interrumpido en la época ≈40 (log con épocas hasta la 40). No se extrapola nada de esos resultados. |
| Docker / imagen real | **NOT VERIFIED** | Docker no está instalado; no se construyó ni arrancó ninguna imagen. Se mantiene como evidencia indirecta el venv Python 3.12 con los requirements del repo. El `Dockerfile` usa `python:3.12-slim` (revisado, no ejecutado). |
| pip-audit | **KNOWN ISSUE** (sin cambios) | No se volvió a ejecutar. `setuptools==80.9.0` / `PYSEC-2026-3447` sigue siendo preexistente (fijado por el repo, no introducido por m48). `torch`/`torchvision` (builds CPU locales) siguen sin poder auditarse (**NOT VERIFIED**). |
| SHAP | **PASS** (actualizado 07/10) | Con `shap 0.52.0` el plugin reproduce el SHAP local y global de la v1.3. El pin `shap==0.42.0` de la v1.1 era un error del equipo, ya corregido en la v1.3 (`shap==0.51.0`). |

**Qué se necesita para cerrar estos puntos** (no se ha hecho nada de esto):
1. Un entorno con credenciales S3 (`STORAGE_BUCKET=xai`, `CUSTOM_S3_ENDPOINT`, claves, región) para ejecutar el plugin con el directorio `artifacts/a48_*` vacío y demostrar la descarga desde S3 (la subida ya está hecha).
2. Un `MLFLOW_TRACKING_URI` accesible para `/train` real.
3. Docker (o la imagen ya construida por el pipeline) para el arranque en el entorno real.
4. La integración de m48 en plataforma/explicabilidad/ODD si se quiere el E2E completo.

## Cambios realizados fuera del plugin

| Fichero | Cambio |
|---|---|
| `app/application/use_cases/predict_model_use_case.py` | +4 líneas: propagación condicional de las opciones XAI en batch (revisado arriba) |
| `app/registry.py` | `ModelEntry` de m48 (con tipos de train) |
| `tests/conftest.py` | fábricas fake y entrada de registro de m48 |
| `tests/unit/test_m48_dnsl_fallas_maquinaria_pasteurizado.py` | 15 tests nuevos (wiring + lógica XAI) |
| `requirements.txt` | **sin cambios** (idéntico al original) |

## Known issues y pendientes

1. **BLOCKED — `/train` con MLflow real** (falta servidor MLflow; solo verificado con la subida sustituida).
2. **NOT VERIFIED — reentrenamiento completo de 300 épocas** (detenido en la época ≈40; sin métricas a escala completa).
3. **KNOWN ISSUE — CVE `PYSEC-2026-3447` en `setuptools==80.9.0`**, versión fijada por el repo y no introducida por m48. **NOT VERIFIED** la auditoría de `torch`/`torchvision` (builds CPU locales).
4. **KNOWN ISSUE — cobertura de m48 39 %** en pytest (`plugin.py` al 0 % por el uso de `FakePlugin`); la verificación del plugin real se hizo con scripts y HTTP/E2E, fuera de pytest. Sería recomendable llevar esas pruebas a tests de integración (decisión pendiente).
5. **NOT VERIFIED — imagen Docker real y wheel de torch de producción** (se verificó un venv Python 3.12 con wheels CPU, no la imagen).
6. **RESUELTO (v1.3) — Tablas 1/2, «v1.2» y riesgo_score:** explicados y corregidos por el equipo de IA; el análisis global del plugin reproduce los CSV de la v1.3 (ver Resultados XAI).
7. **RESUELTO (v1.3) — `shap==0.42.0` del requirements v1.1:** error del equipo (merge del 09/06/2026); la v1.3 fija `shap==0.51.0` y el repo exige `shap>=0.51.0`. El plugin se ha verificado con 0.52.0.
8. **RESUELTO (v1.3) — código original con pandas 3:** corregido por el equipo; el plugin no lo usa.
8b. **RESUELTO — scikit-learn del scaler:** regenerado con 1.5.2 (sin warning, valores idénticos); ver Scaler. Informar al equipo de IA.
8c. **KNOWN ISSUE — versiones del equipo frente al repo:** el equipo ejecuta en Python 3.12 con torch 2.5.1, numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1 y shap 0.51.0; el repo resuelve torch 2.13, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.5.2 y shap 0.52.0 (el plugin está verificado con estas últimas).
8d. **Pendiente con IA:** el equipo pide el fichero golden del Entregable 47 (con los ciclos «20, 23, 210») para revisarlo, porque no reconocen ese listado; hay que enviárselo.
9. **KNOWN ISSUE — datos de laboratorio UCI (aceite), no de leche real.** Las métricas son optimistas para producción; con datos UCI hay que aplicar `apply_digital_twin=true`, con datos de planta debe ser `false`. El valor por defecto de la plataforma es `APPLY_DIGITAL_TWIN` (por defecto `false`).
10. **KNOWN ISSUE — coste de SHAP en CPU:** ≈5–8 s por ciclo en local y ≈60 s para 20 ciclos con SHAP global; por eso es opcional.
11. **Limitaciones de la muestra golden:** 20 ciclos (ver arriba), todos del split de test; el golden XAI es un único ciclo (123, de validación).
12. **Información (no bloqueante) — pesos de a48 vs. m47:** el equipo declara que en la v1.3 son los del Entregable 47 v2.0 (no comprobado por nosotros). No afecta a la integración: m48 usa sus propios artefactos de forma autocontenida y no depende de los de m47.
13. **Artefactos en S3: subidos (PASS por nombres y tamaños, evidencia del usuario).** Pendiente: la **carga real desde S3** y la integración E2E con la plataforma (**NOT VERIFIED**, ver «Validación de cierre»).

## Resumen de resultados

| Verificación | Estado |
|---|---|
| Golden (20 casos), v1.1 y v1.3 | **PASS** |
| XAI local ciclo 123 (13 columnas, v1.3) | **PASS** |
| SHAP local y global con shap 0.52.0 (plugin = CSV v1.3 del equipo) | **PASS** |
| Análisis global CCP + SHAP, 50 ciclos de test (6 filas CCP y 28 SHAP idénticas a la v1.3) | **PASS** |
| Tablas 1/2 de la memoria vs `xai_report.csv` global | **RESUELTO** (v1.3) |
| pytest 556/556 (artefactos v1.1; no repetido el suite completo con v1.3) y 15/15 tests de m48 repetidos con v1.3 | **PASS** |
| flake8 (plugin) / pylint 8,14 | **PASS** |
| Cobertura m48 (39 %) | **KNOWN ISSUE** |
| HTTP: health, stats, predict inline/batch/XAI, 422 | **PASS** |
| Cambio compartido en `predict_model_use_case.py` | **PASS** |
| Carga de artefactos v1.3 y scaler regenerado con sklearn 1.5.2 (sin warning; golden 20/20 y ciclo 123 repetidos) | **PASS** |
| pip-audit (setuptools preexistente) | **KNOWN ISSUE** |
| pip-audit torch/torchvision | **NOT VERIFIED** |
| `/train` con MLflow real | **BLOCKED** |
| `/train` con subida sustituida (contrato HTTP) | **PASS** (solo cableado) |
| Smoke test de reentrenamiento (21 ciclos) | **PASS** funcional, sin valor estadístico |
| Reentrenamiento completo (300 épocas) | **NOT VERIFIED** |
| Imagen Docker real y wheel de torch de producción | **NOT VERIFIED** |
| Artefactos presentes en S3 (`s3://xai/artifacts/fixed/a48_...`) | **PASS** (6 ficheros, tamaños iguales a los locales; evidencia del usuario, sin MD5) |
| Carga real de artefactos desde S3 | **NOT VERIFIED** |
| E2E plataforma/orquestador real → m48 (m48 no está integrado en esos repos) | **NOT VERIFIED** |
| CVE preexistente `setuptools==80.9.0` (fijado por el repo) | **KNOWN ISSUE** |
| Requirements del equipo (`shap==0.42.0` en v1.1) | **RESUELTO** (v1.3 fija 0.51.0) |
| Código original del equipo con pandas 3 | **RESUELTO** (v1.3) |
| Datos UCI/laboratorio y coste de SHAP en CPU | **KNOWN ISSUE** |

**FAIL: ninguno.** Información no bloqueante: identidad de pesos m47/m48 declarada por el equipo, no comprobada por nosotros (known issue 12).

## Estado final

**REQUIERE REVISIÓN — ver detalle arriba.**

El plugin es correcto en lo verificado con los artefactos **v1.3** (golden, XAI local y global, SHAP, wiring, HTTP) y no hay ninguna verificación fallida. Las discrepancias con el equipo de IA de la verificación inicial (Tablas 1/2, versión de shap, pandas 3, pesos) están resueltas en la v1.3. **No está cerrada la integración**: los artefactos v1.3 ya están subidos a S3 (6 ficheros, tamaños verificados por el usuario); queda abierto `/train` con MLflow real (BLOCKED por falta de acceso), y la carga real desde S3, el E2E con la plataforma, el reentrenamiento completo y la imagen Docker/wheel de producción (NOT VERIFIED). El scaler se regeneró con scikit-learn 1.5.2. La subida a S3 la realizó el usuario, no este equipo de verificación.
