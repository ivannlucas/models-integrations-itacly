# Verification report — a33 (ml33-cereals-reuse-strategy-optimizer)

**Fecha:** 2026-08-27
**Modelo:** Optimizador MILP determinista (`scipy.optimize.milp`, HiGHS) para la asignación de
lotes de subproducto cerealista a estrategia de reuso (Animal feed / Composting / Biochar /
Biomass combustion), minimizando emisiones de CO2 simuladas bajo capacidades de planta que se
resetean cada `lots_per_day` lotes.
**Relación con ml31:** dominio de negocio afín (cereales, reducción de residuos) pero problema,
dataset y contrato técnicamente distintos de `ml31_cereals_residue_optimizer` (reasignación de
superficie de cultivo vs. asignación de lotes). Integrado como plugin **nuevo e independiente**
por decisión explícita del usuario — ver `inbox/.../manifest.yaml` known_issues.

## Entorno de verificación (nota de alcance)

Este repo trae 48 modelos con una pila de dependencias combinada muy pesada (torch, tensorflow,
ultralytics, detectron2, mlflow, shap...). No se instaló la pila completa en este pase — se
instaló un entorno mínimo (`fastapi`, `pydantic`, `pandas`, `numpy`, `scipy`, `scikit-learn`,
`joblib`, `boto3`, `python-dotenv`, `pytest`, `flake8`, `pylint`, `pip-audit`, `uvicorn`)
suficiente para este plugin, que no tiene dependencias ML pesadas. Esto es **suficiente y
representativo para ml33** (0 dependencias nuevas fuera de scipy, ya transitiva vía
scikit-learn) pero implica que **no se re-ejecutó** la suite de otros modelos que sí requieren
torch/tensorflow. Detalle marcado explícitamente en cada sección de abajo.

## Parte A — Checklist técnico

- [x] **flake8** (`app/plugins/ml33_cereals_reuse_strategy_optimizer/`, `tests/unit/test_ml33_*.py`,
  `app/registry.py`, `tests/conftest.py`, usando el `.flake8` del repo): **0 errores** tras corregir
  un E203 en `optimizer.py` (slice con espacio antes de `:`).
- [x] **pytest** (`tests/unit/`, `--continue-on-collection-errors` porque 2 ficheros de otros
  modelos requieren `torch`, no instalado en este pase): **320 passed** (7 de ellas son las
  nuevas `test_ml33_cereals_reuse_strategy_optimizer.py`), **13 failed + 1 error**, los 14 casos
  **todos** en `test_ml46_dairy_fouling_clog_detection.py`, `test_infrastructure.py`
  (`TestModelo10ModelLoader`) y `test_modelo10_lacteo_unit.py` por `ModuleNotFoundError: No
  module named 'torch'` — preexistente, ajeno a este cambio (confirmado re-ejecutando
  `test_ml31_cereals_residue_optimizer.py` sin regresiones: 8/8 en verde). Cobertura de
  `app/plugins/ml33_cereals_reuse_strategy_optimizer/` vía estos tests de wiring: DTOs 100%,
  `plugin.py`/`optimizer.py` 0% — **esperado**: estos tests usan `FakePlugin` (patrón del
  propio `conftest.py`: "validan wiring, no correctitud"); la correctitud real se valida en la
  Parte B contra el plugin real.
- [x] **pylint** (`app/plugins/ml33_cereals_reuse_strategy_optimizer/`, `--disable=import-error`):
  **8.58/10** — mismas categorías de aviso que el resto del repo (C0301 line-too-long en
  strings descriptivos largos, R09xx de complejidad en el solver), sin `.pylintrc` propio del
  repo que las suprima. Referencia: `ml31_cereals_residue_optimizer` (ya en producción) puntúa
  9.18/10 con los mismos tipos de aviso. No se consideran issues nuevos.
- [x] **pip-audit** (`requirements.txt` completo, resolviendo los 64 paquetes incl. torch/
  tensorflow/ultralytics vía metadatos, sin necesidad de instalarlos): 4 CVEs conocidas en 3
  paquetes — `torch==2.11.0` (PYSEC-2025-194, fix 2.13.0), `setuptools` transitivo 81.0.0
  (PYSEC-2026-3447, fix 83.0.0), `cryptography` transitivo 49.0.0 (PYSEC-2026-3552, fix 50.0.0).
  **Ninguna involucra `scipy`** (única dependencia nueva de este plugin, sin CVEs conocidas).
  Las 3 vulnerabilidades son preexistentes y de paquetes usados por otros modelos — fuera del
  alcance de esta integración; no se ha tocado su versión para no romper otros plugins ya en
  producción. Señalado para que el equipo lo priorice de forma centralizada.
- [x] **Arranque local + health + predict + stats**: no se pudo usar literalmente
  `MODEL=ml33-cereals-reuse-strategy-optimizer python main.py`, porque `app/registry.py` importa
  los 22 plugins ya registrados a nivel de módulo (torch/tensorflow/etc. no instalados en este
  pase) **incluso** con `MODEL` filtrando cuál se *carga* en runtime. En su lugar se montó un
  servidor `uvicorn` real, standalone, que instancia el plugin real
  (`Ml33CerealsReuseStrategyOptimizerPlugin`, no un fake) sobre el `router_factory`/
  `ModelContainer` reales — mismo código de producción, solo sin pasar por `app/registry.py`.
  Resultado:
  - `GET /health` → `{"status":"ok","model":"ml33-cereals-reuse-strategy-optimizer","loaded":true}`
  - `GET /stats` → 200, `model_name` correcto, `metrics` con las cifras del manifest.
  - `POST /predict` (inline, golden case `bloque_001_filas_1_15`, 15 lotes) → reproduce
    **exactamente** la lista de 15 estrategias esperada (ver Parte B).
  - `POST /predict` (batch, CSV real de 10 000 filas) → reproduce **exactamente** la
    distribución/emisión total esperadas (ver Parte B). Nota de rendimiento: la llamada tardó
    varios minutos en este entorno de desarrollo mono-worker (667 bloques × solve MILP
    secuencial dentro de una única petición síncrona). No es un defecto de corrección, pero es
    una característica a tener en cuenta para CSVs muy grandes en producción — ver "Pendiente
    antes de PR".
- [x] **`/train`**: `POST /train` → **501** (`TrainingNotSupportedError`). Correcto:
  `manifest.training.supported: false` (solver MILP exacto, sin pesos entrenables).

## Parte B — Correctitud contra el golden dataset (manifest, 3 casos)

Ejecutado el motor real (`ExactEmissionsOptimizer` vía `Ml33CerealsReuseStrategyOptimizerPlugin`)
tanto en llamada directa como a través del servidor HTTP real. **Tolerancia: 0% (exact match)**
— es un solver MILP determinista sin semilla aleatoria; no hay margen de error numérico legítimo
que enmascarar.

| Golden case | Vía | Esperado | Obtenido | ¿OK? |
|---|---|---|---|---|
| `bloque_001_filas_1_15` (15 lotes, bloque 1) | HTTP `/predict` inline real | `[Animal feed, Composting, Biomass combustion, Composting, Animal feed, Biomass combustion, Biomass combustion, Biomass combustion, Animal feed, Composting, Biochar, Biochar, Biochar, Biochar, Composting]`, `ai_is_fallback=false` en las 15 | Idéntico, `capacity_fallback_count=0` | ✅ exacto |
| `bloque_002_filas_16_30` (15 lotes, bloque 2) | Motor real (`optimizer.py`) sobre el CSV del delivery | `[Composting, Composting, Biochar, Biomass combustion, Composting, Biomass combustion, Biomass combustion, Composting, Animal feed, Composting, Biochar, Animal feed, Biomass combustion, Animal feed, Animal feed]` | Idéntico, 0 mismatches | ✅ exacto |
| `batch_full_test_split_distribution` (10 000 filas) | HTTP `/predict` batch real, CSV completo `dataset_optimization_cereal_co2_test_raw.csv` | `n_rows=10000`; distribución `{composting: 37.18, animal_feed: 27.27, biomass_combustion: 17.86, biochar: 17.69}`; `total_estimated_emissions_kg=30893908.4`; `capacity_fallback_count=0` | `n_rows=10000`; distribución idéntica; `total_estimated_emissions_kg=30893908.4023` (redondea a 30893908.4); `capacity_fallback_count=0` | ✅ exacto |

Adicionalmente, se comparó el motor real contra **las 10 000 filas completas** del split de test
(no solo los 2 bloques del manifest) llamando directamente a `ExactEmissionsOptimizer` con las
mismas 10 000 filas de `dataset_optimization_cereal_co2_test_raw.csv` y comparando fila a fila
contra `ai_assigned_strategy` de `data/predictions/inference_with_constraints.csv` (salida real
y auditada del delivery original): **0 discrepancias en 10 000/10 000 filas**. Esta es la
comprobación de correctitud más fuerte disponible — no solo reproduce los golden cases
puntuales del manifest, sino el pipeline completo del equipo de IA, bit a bit.

**Resultado Parte B: 3/3 golden cases dentro de tolerancia (exact match), más 10 000/10 000 filas
del split de test completo verificadas contra la salida auditada del delivery original.**

## Desviación deliberada del código fuente original

`src/predict/exact_optimizer.py` (delivery original) envuelve la llamada al solver en un
context manager (`_suppress_native_stdout`) que redirige el file descriptor 1 (stdout) a nivel
de sistema operativo para silenciar el banner nativo de HiGHS. Esta técnica es segura en un
script CLI mono-hilo, pero es **insegura en un servidor ASGI multi-hilo/concurrente**: dos
peticiones resolviendo un bloque en paralelo podrían dejar el fd 1 del proceso apuntando
permanentemente a `/dev/null` tras la primera que termine (ver razonamiento completo en el
docstring de `optimizer.py`). Se ha **omitido deliberadamente** en el puerto — no cambia ningún
resultado numérico (confirmado por la Parte B), solo implica que ocasionalmente puede aparecer
una línea de log nativa de HiGHS (`HighsMipSolverData::...`) en el stdout del proceso. Cosmético,
no funcional.

## Dependencias

- `scipy>=1.11.0` añadida explícitamente a `requirements.txt` (antes solo transitiva vía
  `scikit-learn`; ahora importada directamente por `scipy.optimize.milp`). Sin CVEs conocidas
  (ver pip-audit arriba).
- Sin nuevas dependencias de ML pesadas: el motor desplegado no usa `neat-python` (solo el
  benchmark retenido del delivery original lo usa, y no se ha portado — ver manifest).

## Pendiente antes de PR (revisión humana)

1. **Sin memoria .docx**: este delivery no trae `entregable/*.docx` — el manifest y este
   informe se construyeron desde código real + README.md + JSONs de métricas auditados. Si
   existe una memoria oficial fuera de este repo, debe revisarse antes de `docs-generation`
   (puede tener secciones — título de negocio, autoría, contexto de proyecto — que el código no
   expresa).
2. **Latencia de `predict_batch` en CSVs grandes**: varios minutos para 10 000 filas en este
   entorno de desarrollo (single-worker, 667 soluciones MILP secuenciales dentro de una única
   petición HTTP síncrona). Evaluar si el timeout del API gateway de producción lo tolera, o si
   conviene paralelizar bloques / mover a un flujo asíncrono para CSVs de este tamaño.
3. **Vulnerabilidades preexistentes** (`torch`, `setuptools`, `cryptography` transitivos) no
   corregidas aquí por estar fuera del alcance de esta integración y compartidas con otros
   modelos ya en producción — priorizar de forma centralizada.
4. No se re-ejecutó la suite completa de modelos que requieren `torch`/`tensorflow` en este
   pase (ver nota de alcance arriba) — no hay motivo para esperar impacto (ml33 no toca ningún
   fichero compartido salvo `app/registry.py` y `tests/conftest.py`, ambos solo con adiciones
   aisladas), pero queda pendiente de que el pipeline CI real (con el entorno completo) lo
   confirme.

## Estado final

**LISTO PARA PR** — checklist técnico en verde para todo lo verificable en este entorno,
correctitud exacta (0 discrepancias) contra el golden dataset completo del manifest y contra las
10 000 filas del split de test original. Puntos 1–4 de arriba son para revisión humana, no
bloqueantes de corrección.
