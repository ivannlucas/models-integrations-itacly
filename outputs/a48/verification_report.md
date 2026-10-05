# Verificación — lacteo-xai-puntos-criticos-control (m48-dnsl-fallas-maquinaria-pasteurizado)

Leyenda: **PASS** = ejecutado y correcto · **FAIL** = ejecutado y fallido · **BLOCKED** = no completable por dependencia externa/infraestructura · **NOT VERIFIED** = no validado completamente · **KNOWN ISSUE** = limitación o discrepancia documentada.

Fecha de verificación: 2026-10-05. Este informe no abre PR, no hace commit ni sube artefactos.

## Identificación

| Campo | Valor |
|---|---|
| Plugin | `app/plugins/m48_dnsl_fallas_maquinaria_pasteurizado/` |
| `model_id` runtime / prefijo | `m48-dnsl-fallas-maquinaria-pasteurizado` / `/models/m48-dnsl-fallas-maquinaria-pasteurizado` |
| Manifest | `inbox/a48/manifest.yaml` (model_id de manifest: `lacteo-xai-puntos-criticos-control`) |
| Memoria | Entregable 48 v1.1 (11/06/2026) |
| Tipo | Clasificación multietiqueta 4 componentes × 3 estados (DNSL, 1D-CNN) + XAI (Grad-CAM, SHAP opcional, CCP, motor prescriptivo) |
| Relación con m47 | Plugin independiente y autocontenido, con artefactos propios; no reutiliza los de m47. Información secundaria: no se ha establecido formalmente si los pesos de m48 son idénticos o distintos a los de m47 (los splits de a48 y a47 difieren). Esto no afecta a la integración actual ni es un requisito de la misma. |

## Artefactos

Ubicación local: `artifacts/a48_dnsl_fallas_maquinaria_pasteurizado/` (en `.gitignore`; la subida a S3 **no se ha realizado**).

- `neurosymbolic_cnn.pth`, `scaler_cnn_dns.pkl`, `feature_columns.pkl`, `ts1_mean_train.pkl` (entregados por el equipo de IA).
- `shap_background.npy` (50 ciclos de test ya escalados, 3,4 MB) + `shap_background_meta.json` (generado por nosotros; no se incluye el CSV original de 101 MB).
- Estado: carga **PASS** (ver Scaler). Subida a S3: **pendiente, no ejecutada**.

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
| `POST /predict` inline (CSV del ciclo 123, `apply_digital_twin=true`, `include_shap=true`) | **PASS** — `[2,0,0,0]`; SHAP FS1 0,2288 / FS1 0,3224 / TS2 0,2639 / TS2 0,3334 |
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
Resultado: **20/20 casos dentro de tolerancia — PASS.**

Limitaciones de esta muestra: 20 ciclos no son una validación estadística (con un exact_match de 0,988 se esperaría ≈0,2 fallos en 20 casos); verifica el cableado y el preprocesado, no sustituye las métricas de test del equipo. Los datos de entrada se extrajeron de `data/processed/hydraulic_10hz_raw.csv` (601 filas por ciclo, el modelo trunca a 600). **Sin gemelo digital los resultados no son válidos para datos UCI** (los golden requieren `apply_digital_twin=true`).

## Resultados XAI (ciclo 123, Tabla 3 de la memoria)

El ciclo 123 pertenece al split de validación, no al de test. Referencia: Tabla 3 de la memoria / `xai_report_cycle.csv` entregado.

| Elemento | Resultado |
|---|---|
| Predicción por componente (Crítico, Sano, Sano, Sano) y confianza (0,9964 / 1,0000 / 1,0000 / 0,9995) | **PASS** — coincide con la referencia (4 decimales) |
| Ventana Grad-CAM (s) e intensidad: 9,1–11,0 (0,8651), 8,6–10,7 (0,8946), 37,4–39,9 (0,8940), 9,1–11,0 (0,8790) | **PASS** — coincide |
| Acción, urgencia, intervalo, riesgo (1,0 CRÍTICO; 0,0 NORMAL ×3) | **PASS** — coincide |
| Grad-CAM sin SHAP (`include_shap=false`) | **PASS** — `sensor_mas_relevante = null`, `riesgo_incluye_shap = false`; riesgo de Fouling 0,9964 (sin bonus SHAP, distinto de 1,0 con SHAP por diseño) |
| Análisis global CCP (batch 20 ciclos, con y sin SHAP) | **PASS** funcional (8 filas CCP; con SHAP 28 filas SHAP, 59–64 s en CPU). No existe referencia numérica reproducible para el global (la muestra y los datos difieren), por lo que solo se verificó estructura y ejecución |

## Resultados SHAP

- **PASS (Python 3.12, shap 0.52.0, entorno de referencia):** el plugin da para el ciclo 123 FS1 0,2288, FS1 0,3224, TS2 0,2639, TS2 0,3334, **idéntico** al `xai_report_cycle.csv` y a la Tabla 3.
- **Código original del equipo con shap 0.52.0** (ejecutado en un venv Python 3.12 auxiliar con pandas 2.3.3, porque el código original falla con pandas 3): da exactamente el mismo resultado que el plugin y que el CSV entregado.
- **Con shap 0.42.0 (Python 3.11):** tanto el código original como el plugin dan Fouling VS1 0,355, Válvula PS3 0,571, Bomba PS3 0,352, Acumulador PS3 0,450. Con 0,42.0 el SHAP difiere. En el ciclo 123, predicción, confianza, ventana Grad-CAM y acción coinciden con la referencia en ambas versiones, porque no dependen de SHAP; el riesgo calculado con SHAP incorpora por diseño un bonus basado en SHAP, de modo que en general puede variar con la versión (en el ciclo 123 los niveles de riesgo coincidieron).
- **Conclusión:** la discrepancia observada inicialmente se debía a la **versión de shap**, no a un error del port. El `xai_report_cycle.csv` entregado se generó con shap ≥ 0,51, aunque el `requirements.txt` del equipo indica `shap==0.42.0`. El port se considera fiel. **No hay ninguna discrepancia SHAP abierta atribuible al plugin**; lo que queda es documental (el requirements entregado por el equipo declara una versión de shap que no reproduce sus propios CSV, ver known issue 7).
- Estabilidad: con shap 0.42.0 el top-sensor del ciclo 123 fue estable entre 5 semillas en Fouling, Válvula y Acumulador y varió en Bomba (PS3/FS1); no se repitió ese barrido con 0,52.0.
- **KNOWN ISSUE (para confirmar con IA, no bloqueante):** qué versión exacta de shap usaron para generar los CSV, dado que su requirements dice 0.42.0.
- Las Tablas 1 y 2 de la memoria (análisis global) siguen sin coincidir con el `xai_report.csv` entregado (p. ej. Fouling: memoria FS1 ventana 94–111; CSV TS2 ventanas 103–293 y 507–583). Con la explicación de la versión de shap solo se ha comprobado el caso local del ciclo 123; **no se ha reproducido el análisis global**, por lo que esta discrepancia de las Tablas 1/2 sigue **sin explicar** y pendiente de validar con el equipo de IA. Es una discrepancia documental entre la memoria y el CSV entregado; **no se atribuye a un fallo del plugin**, que reproduce el CSV en el único caso verificado (ciclo 123) y coincide con el código original del equipo.

## Scaler y carga de artefactos

- **PASS** — `scaler_cnn_dns.pkl` carga con scikit-learn 1.5.2 (pin del repo), numpy 2.5.3 y pandas 3.0.6 sin errores ni avisos; `n_features_in_ = 28`. Su `_sklearn_version` interno es 1.5.2 (no 1.6.1 como figura en el requirements del equipo). scikit-learn 1.6.x no se ha probado.
- **PASS** — `feature_columns.pkl` (28 columnas, orden esperado), `ts1_mean_train.pkl` (45,311), `neurosymbolic_cnn.pth` (36 tensores, `weights_only=True`).
- **PASS** — `shap_background.npy` (50, 28, 600) float32 carga y se usa en el SHAP local.

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
| Artefactos en `s3://xai/artifacts/fixed/a48_dnsl_fallas_maquinaria_pasteurizado/` | **BLOCKED** | No hay CLI de AWS ni credenciales en el entorno, por lo que no se pudo listar el prefijo ni comparar nombres/tamaños. No se sabe si los artefactos están subidos; no se subió nada. |
| Carga real de artefactos desde S3 | **NOT VERIFIED** | Depende del punto anterior. Análisis estático (no es prueba): `plugin.load()` llama a `ArtifactStore(ARTIFACT_FOLDER_NAME).download_all_if_needed()` solo si `STORAGE_BUCKET` está definido; esto lista `artifacts/fixed/<carpeta>/` en el bucket y descarga a `artifacts/<carpeta>/` los ficheros que falten o cuyo tamaño difiera, y después el plugin carga siempre desde ese directorio local. Como el directorio local ya contiene los 6 ficheros, una ejecución con S3 configurado **no demostraría el origen S3** salvo que se vacíe el directorio local antes. El prefijo descargado incluye `shap_background.npy` y `shap_background_meta.json` porque se baja todo el prefijo. Los artefactos permanecen en disco tras la carga (mismo patrón que m47). |
| E2E plataforma/orquestador → m48 | **NOT VERIFIED** | Lo que sí está verificado (PASS, secciones anteriores): HTTP real contra `main.py` con el registro → router → use case → plugin m48 → XAI/SHAP → respuesta, y la propagación de `include_xai`, `include_shap`, `n_samples` y `apply_digital_twin` sin afectar a plugins con firma estándar. La plataforma real no está integrada con m48: una búsqueda de `m48-dnsl`/`m48_dnsl` en los repos hermanos `retech-lote2-xai-plataforma`, `retech-lote2-xai-explicabilidad` y `retech-lote2-xai-odd-detection` da **0 referencias** (las skills `front-integration`, `explainability-integration` y `odd-integration` no se han aplicado a m48), y además no hay S3 ni MLflow accesibles. |
| `/train` contra MLflow real | **BLOCKED** | `MLFLOW_TRACKING_URI` no definido; el valor por defecto `http://mlflow.mlflow:5000` no resuelve desde este equipo. Sin cambios respecto a la verificación anterior (HTTP 500 por la subida). No se creó ningún run ni se comprobaron parámetros, métricas o artefactos en MLflow. |
| Entrenamiento completo | **NOT VERIFIED** | No existe evidencia real de un entrenamiento completo: ni checkpoint final, ni métricas, ni run de MLflow, ni informe. Solo hay el smoke test de 21 ciclos y un entrenamiento interrumpido en la época ≈40 (log con épocas hasta la 40). No se extrapola nada de esos resultados. |
| Docker / imagen real | **NOT VERIFIED** | Docker no está instalado; no se construyó ni arrancó ninguna imagen. Se mantiene como evidencia indirecta el venv Python 3.12 con los requirements del repo. El `Dockerfile` usa `python:3.12-slim` (revisado, no ejecutado). |
| pip-audit | **KNOWN ISSUE** (sin cambios) | No se volvió a ejecutar. `setuptools==80.9.0` / `PYSEC-2026-3447` sigue siendo preexistente (fijado por el repo, no introducido por m48). `torch`/`torchvision` (builds CPU locales) siguen sin poder auditarse (**NOT VERIFIED**). |
| SHAP | **PASS / KNOWN ISSUE** (sin cambios) | Con `shap 0.52.0` el plugin reproduce el CSV (ya verificado). La discrepancia con el `shap==0.42.0` del requirements del equipo sigue como KNOWN ISSUE documental. |

**Qué se necesita para cerrar estos puntos** (no se ha hecho nada de esto):
1. Un entorno con credenciales S3 (`STORAGE_BUCKET=xai`, `CUSTOM_S3_ENDPOINT`, claves, región) y el AWS CLI, para listar el prefijo, comparar tamaños y, tras la subida si falta algo, ejecutar el plugin con el directorio `artifacts/a48_*` vacío para demostrar la descarga.
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
6. **KNOWN ISSUE — Tablas 1/2 de la memoria** frente a `xai_report.csv` (análisis global): sin explicar y sin reproducir; pendiente con el equipo de IA. Igualmente la mención a una «v1.2» inexistente en la memoria v1.1.
7. **KNOWN ISSUE — el requirements del equipo** indica `shap==0.42.0` pero los CSV se generaron con shap ≥ 0,51; confirmar versión con IA.
8. **KNOWN ISSUE — el código original del equipo no funciona con pandas 3** (`KeyError: 'Cycle_ID'` en `feature_engineering`); el plugin no lo usa y funciona con pandas 3.0.6.
9. **KNOWN ISSUE — datos de laboratorio UCI (aceite), no de leche real.** Las métricas son optimistas para producción; con datos UCI hay que aplicar `apply_digital_twin=true`, con datos de planta debe ser `false`. El valor por defecto de la plataforma es `APPLY_DIGITAL_TWIN` (por defecto `false`).
10. **KNOWN ISSUE — coste de SHAP en CPU:** ≈5–8 s por ciclo en local y ≈60 s para 20 ciclos con SHAP global; por eso es opcional.
11. **Limitaciones de la muestra golden:** 20 ciclos (ver arriba), todos del split de test; el golden XAI es un único ciclo (123, de validación).
12. **Información (no bloqueante) — pesos de a48 vs. m47:** no se ha establecido formalmente si son idénticos o distintos. No afecta a la integración actual: m48 usa sus propios artefactos de forma autocontenida y no depende de los de m47.
13. **BLOCKED / pendiente — artefactos en S3:** no se pudo comprobar si están subidos (sin AWS CLI ni credenciales en este equipo) y no se subió nada. La carga real desde S3 y la integración E2E con la plataforma están **NOT VERIFIED** (ver «Validación de cierre»).

## Resumen de resultados

| Verificación | Estado |
|---|---|
| Golden (20 casos) | **PASS** |
| XAI ciclo 123 (predicción, confianza, ventana, riesgo, acción) | **PASS** |
| SHAP ciclo 123 con shap 0.52.0 (plugin = código original = CSV del equipo) | **PASS** |
| Análisis global CCP (estructura y ejecución) | **PASS** |
| Tablas 1/2 de la memoria vs `xai_report.csv` global | **KNOWN ISSUE** |
| pytest 556/556 | **PASS** |
| flake8 (plugin) / pylint 8,14 | **PASS** |
| Cobertura m48 (39 %) | **KNOWN ISSUE** |
| HTTP: health, stats, predict inline/batch/XAI, 422 | **PASS** |
| Cambio compartido en `predict_model_use_case.py` | **PASS** |
| Carga de artefactos y scaler (sklearn 1.5.2) | **PASS** |
| pip-audit (setuptools preexistente) | **KNOWN ISSUE** |
| pip-audit torch/torchvision | **NOT VERIFIED** |
| `/train` con MLflow real | **BLOCKED** |
| `/train` con subida sustituida (contrato HTTP) | **PASS** (solo cableado) |
| Smoke test de reentrenamiento (21 ciclos) | **PASS** funcional, sin valor estadístico |
| Reentrenamiento completo (300 épocas) | **NOT VERIFIED** |
| Imagen Docker real y wheel de torch de producción | **NOT VERIFIED** |
| Artefactos presentes en S3 (`s3://xai/artifacts/fixed/a48_...`) | **BLOCKED** (sin CLI ni credenciales) |
| Carga real de artefactos desde S3 | **NOT VERIFIED** |
| E2E plataforma/orquestador real → m48 (m48 no está integrado en esos repos) | **NOT VERIFIED** |
| CVE preexistente `setuptools==80.9.0` (fijado por el repo) | **KNOWN ISSUE** |
| Requirements del equipo (`shap==0.42.0`) vs versión necesaria para reproducir sus CSV (`shap>=0.51`) | **KNOWN ISSUE** (documental, pendiente con IA) |
| Código original del equipo con pandas 3 | **KNOWN ISSUE** (no afecta al plugin) |
| Datos UCI/laboratorio y coste de SHAP en CPU | **KNOWN ISSUE** |

**FAIL: ninguno.** Información no bloqueante: identidad de pesos m47/m48 no establecida formalmente (known issue 12).

## Estado final

**REQUIERE REVISIÓN — ver detalle arriba.**

El plugin es correcto en lo verificado (golden, XAI, SHAP, wiring, tests, HTTP) y no hay ninguna verificación fallida. **No está cerrada la integración**: quedan abiertos `/train` con MLflow real y los artefactos en S3 (BLOCKED por falta de acceso en este equipo), y la carga real desde S3, el E2E con la plataforma, el reentrenamiento completo y la imagen Docker/wheel de producción (NOT VERIFIED). Además hay discrepancias documentales pendientes con el equipo de IA (Tablas 1/2, versión de shap). No se ha subido ningún artefacto a S3.
