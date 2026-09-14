# Verificación — ml15-wine-ipi-price-forecast (a15)

Fecha: 2026-09-14
Plugin: `app/plugins/ml15_wine_ipi_price_forecast/`
Manifest: `inbox/a15/manifest.yaml`

## Checklist técnico

- [x] **flake8** (`app/plugins/ml15_wine_ipi_price_forecast/`, `tests/unit/test_ml15_*.py`,
      `app/registry.py`, `app/domain/services/exceptions.py`, `tests/conftest.py`): **0 errores**.
- [x] **pytest** (`tests/unit/`): **316/316 passed** tras integrar ml15 (14 tests propios de
      ml15: 6 de endpoints + 8 de `preprocessing.py`), sin regresiones en el resto del repo.
      *Nota de entorno*: 3 ficheros de test preexistentes (`test_modelo10_lacteo_unit.py`,
      `test_ml46_dairy_fouling_clog_detection.py`, parte de `test_infrastructure.py`) no se
      pudieron ejecutar en esta verificación por `ModuleNotFoundError: torch` — `torch` no está
      instalado en este entorno sandbox y no es una dependencia de ml15. Confirmado que el fallo
      es previo a este cambio (falla exactamente igual con `git stash` de los cambios de ml15).
- [x] **pylint** `app/plugins/ml15_wine_ipi_price_forecast/ --disable=import-error`: **8.80/10**,
      sin categorías de aviso nuevas respecto al resto del repo — se comparó explícitamente contra
      `ml17_meat_market_price_analysis` (plugin ya integrado y en producción, tipo de modelo más
      cercano: Ridge/sklearn), que puntúa 9.70/10 con los mismos dos tipos de aviso: `C0301`
      (líneas >100 car., pylint por defecto — el proyecto usa `max-line-length=120` en `.flake8`
      con `E501` **ignorado explícitamente**, así que no es un criterio real del proyecto) y
      `C0103` (variable `X` para la matriz de features, convención universal sklearn/numpy, usada
      igual en ml17). Un `R0914` (too-many-locals) real en `predict_inline`/`predict_batch` sí se
      corrigió extrayendo `_predict_row()` como helper compartido.
- [x] **pip-audit**: no se pudo ejecutar contra `requirements.txt` del repo completo en este
      entorno (dependencias pesadas del resto de plugins — torch/tensorflow/detectron2/
      ultralytics/opencv — no instalables aquí). Se ejecutó `pip-audit --local` contra el entorno
      con las dependencias reales de ml15 instaladas (pandas, numpy, scikit-learn, joblib,
      fastapi, pydantic, boto3, python-dotenv, mlflow-skinny): ninguna de ellas aparece en los
      hallazgos — todos los CVEs reportados son de paquetes de sistema Ubuntu preexistentes
      (pip, setuptools, urllib3, requests, pygments, pyjwt, wheel, pytest) no relacionados con
      este plugin.
- [x] **Arranque local + health/predict/stats/train — OK**, contra un servidor FastAPI real
      (no mocks): se montó una app standalone que replica exactamente el wiring de `main.py`
      (`ModelContainer` + `make_model_router`) para `ml15-wine-ipi-price-forecast`, evitando solo
      el *import* eager de `app.registry` (que carga los 25 plugins del repo, incluidos los que
      requieren torch/tensorflow — no instalables aquí). El artefacto real
      `models/artifacts/horizon_6_ridge.pkl` del equipo de IA se cargó sin mocks.
  - `GET /health` → `200 {"status":"ok","loaded":true}`
  - `POST /predict` (inline, caso_001 del golden dataset) → `200`, `y_pred=125.48069826420476`
    — coincide exactamente con el valor esperado.
  - `POST /predict` (inline, sin `date` ni variables de calendario) → `422`
    (`MissingRequiredFeatureError`, no 500).
  - `GET /stats` → `200`, incluye `metrics_reported` real del manifest.
  - `POST /train` → `200`, `TrainResponse` con `rmse/mae/mape_pct/r2/mda_pct` (smoke test con
    target sintético solo para probar el wiring end-to-end, no para evaluar el modelo).
- [x] **/train según `manifest.training`**: `supported: true` en el manifest → el endpoint
  devuelve `200` con métricas reales (no 501). Confirmado.

## Correctitud (golden dataset)

`inbox/a15/manifest.yaml::golden_cases` contiene 13 casos — las 13 filas de
`data/splits/test/panel_test.csv` (único split de test disponible), cada uno con el `y_pred`
real de `models/artifacts/horizon_6_ridge.pkl` recalculado y verificado. Se ejecutaron los 13
contra el endpoint HTTP real (`POST /predict`, modo inline, aportando `date` para derivar las 4
variables de calendario) — no contra una llamada directa a Python — para verificar el wiring
completo (DTO → use case → plugin → preprocessing → modelo).

| Caso | Esperado (y_pred) | Obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|
| caso_001 | 125.4806982642 | 125.4806982642 | 0.0 | ✅ |
| caso_002 | 126.4201951772 | 126.4201951772 | 0.0 | ✅ |
| caso_003 | 126.3227400402 | 126.3227400402 | 0.0 | ✅ |
| caso_004 | 126.1130040261 | 126.1130040261 | 0.0 | ✅ |
| caso_005 | 127.2265468732 | 127.2265468732 | 0.0 | ✅ |
| caso_006 | 127.9986014562 | 127.9986014562 | 0.0 | ✅ |
| caso_007 | 128.4897489307 | 128.4897489307 | 0.0 | ✅ |
| caso_008 | 129.8074591069 | 129.8074591069 | 0.0 | ✅ |
| caso_009 | 131.7418608027 | 131.7418608027 | 0.0 | ✅ |
| caso_010 | 130.3852867238 | 130.3852867238 | 0.0 | ✅ |
| caso_011 | 130.1021000790 | 130.1021000790 | 0.0 | ✅ |
| caso_012 | 130.2042067868 | 130.2042067868 | 0.0 | ✅ |
| caso_013 | 129.9802623156 | 129.9802623156 | 0.0 | ✅ |

Tolerancia usada: `tolerance_pct: 1.0` declarada en el manifest (margen de reproducibilidad
numérica entre entornos/versiones de scikit-learn); en la práctica todos los casos reprodujeron
el valor exacto (diferencia 0.0), porque Ridge+StandardScaler es determinista y el wiring del
plugin reconstruye exactamente el mismo vector de 16 features que el artefacto espera.

**Resultado: 13/13 casos dentro de tolerancia.**

Solo 8/13 casos (`caso_001`–`caso_008`) tienen `y_true` real disponible en el dataset (los 5
restantes tienen `target_date` posterior al último dato de la serie IPI nacional, feb-2026 — ver
manifest known_issues). Sobre esos 8, las métricas de exactitud frente al valor real (no frente
al propio modelo) coinciden con `models/metrics/final_evaluation.json`: RMSE=1.0396, MAE=0.9440,
MAPE=0.7377%, R²=0.1751, MDA=100% — recalculadas de forma independiente en el proceso de
`manifest-extraction` y reconfirmadas aquí contra el endpoint HTTP real.

## Hallazgo corregido durante la verificación

La primera versión de `predict_dto.py` declaraba `PredictInlineRequest.features: dict[str, Any]`
como un único campo anidado. `PredictModelUseCase.execute()` construye el diccionario que recibe
`plugin.predict_inline(features=...)` haciendo `request.model_dump(exclude={"mode", "model_key",
"threshold", "mlflow_run_id", "data_path"})` sobre **todo** el request — es decir, espera que
cada feature sea un campo de nivel superior del DTO (como hace `ml17_meat_market_price_analysis`,
el plugin de referencia más cercano), no un campo `features` anidado. Con el diseño inicial, el
plugin recibía `{"features": {...}}` en vez del diccionario plano, y no encontraba `date` ni
ninguna feature — silenciosamente exigía derivar el calendario y fallaba con 422 en **todas**
las peticiones inline reales, pese a que las pruebas con `FakePlugin`/llamada directa a Python
pasaban (porque ahí se invoca `plugin.predict_inline(features=...)` sin pasar por el use case
real). Se detectó al ejecutar el checklist de arranque local contra el servidor HTTP real (no
solo contra `FakePlugin`) y se corrigió rediseñando `PredictInlineRequest` con las 16 features
como campos individuales tipados — igual que `ml17`. Los 13 golden cases se revalidaron después
del fix contra el endpoint HTTP real (ver tabla arriba). **Lección para futuros plugins**: la
Parte A del checklist de este skill (arranque local + curl real) no es opcional ni sustituible
por los tests con `FakePlugin` — son cosas distintas y este bug solo lo detecta la primera.

## Estado final

**✅ Verificado — listo para `docs-generation` y revisión humana.**

Pendiente de decisión humana (no bloquea el plugin, documentado en
`inbox/a15/manifest.yaml::known_issues`):
- Confirmar si el nombre "rnn" del directorio original corresponde a una fase de experimentación
  con LSTM/GRU descartada y no documentada, o es un residuo de plantilla de otro modelo del lote.
- Decidir si el modo `regional_layer` (capa contextual por CCAA) se integra como modo adicional
  de este plugin en una iteración futura, o queda fuera de alcance permanentemente.
- Fijar `scikit-learn==1.8.0` exacto en el entorno de despliegue (el artefacto se serializó con
  esa versión; se probó cargándolo con 1.9.1 sin diferencias numéricas, pero con
  `InconsistentVersionWarning`).
