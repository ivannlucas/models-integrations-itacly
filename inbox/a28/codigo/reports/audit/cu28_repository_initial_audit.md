# Auditoría inicial de hardening del repositorio CU28

Fecha: 2026-06-24
Rama: `hardening/cu28-remove-legacy-stale-artifacts`
Scope oficial: `mixed_context`
Fecha de referencia oficial: `2026-05-18`

## Criterio de clasificación

La clasificación se realizó sobre ficheros versionados (`git ls-files`) y
referencias desde código, CLI, tests, documentación y manifiestos. Las rutas
agrupadas en una misma fila reciben el mismo tratamiento porque comparten
origen, función y estado de vigencia.

## Inventario y decisión

| Ruta o conjunto | Ficheros | Categoría | Decisión |
|---|---:|---|---|
| `config/` | 3 | OFICIAL | Conservar como configuración canónica. |
| `src/` usado por `src.main`, reproducción y plataforma | 77 | NECESARIO PARA REPRODUCIBILIDAD | Conservar. |
| `models/metrics/official/` | 8 | OFICIAL | Conservar como inputs recalculables de métricas. |
| `models/metrics/summary/*latest__mixed_context*` y `metrics_summary__mixed_context.*` | 10 | OFICIAL | Conservar como única familia de summaries. |
| Artefactos y métricas de la corrida `mixed_context_20260518_seed42_smoke` referenciados por el manifest | 15 | NECESARIO PARA REPRODUCIBILIDAD | Conservar como evidencia de ejecución, sin tratarlos como summaries oficiales adicionales. |
| `reports/official/` | 3 | OFICIAL | Conservar; el informe Markdown y la fórmula se derivan de artefactos canónicos. |
| `reports/audit/cu28_metrics_consistency_report.md` y script asociado | 2 | OFICIAL | Conservar y revalidar. |
| `notebooks/00` a `notebooks/07` y sus renders EDA | 80 | NECESARIO PARA REPRODUCIBILIDAD | Conservar y regenerar con la corrida oficial. |
| Datos raw, procesados, splits y predicciones incluidos en manifiestos | 60 | NECESARIO PARA REPRODUCIBILIDAD | Conservar. |
| `tests/` | 25 | TEST | Conservar; ampliar con controles anti-legacy. |
| `legacy/deprecated_before_platform_reset/` | 430 | LEGACY ELIMINABLE | Eliminar. No tiene dependencias operativas fuera de `legacy/` y contiene datos, modelos, notebooks y documentos sustituidos. |
| `legacy/metrics_snapshots/` | 9 | LEGACY ELIMINABLE | Eliminar. Contiene summaries históricos que contradicen la familia oficial. |
| `data/interim/external/legacy_raw_cache/` | 7 | LEGACY ELIMINABLE | Eliminar del control de versiones. Los siete ficheros son copias SHA-256 idénticas de `data/raw/external/`. Cambiar el ETL para usar un cache regenerable no versionado. |
| Artefactos y métricas de modelo con run ID `20260604T092638Z` | 12 | LEGACY ELIMINABLE | Eliminar. Son resultados anteriores a la corrida congelada. |
| `models/metrics/summary/*_comparison.csv` y `*_comparison.json` | 4 | LEGACY ELIMINABLE | Eliminar y dejar de generarlos; duplican los summaries `latest` oficiales. |
| `models/metrics/policy_simulation_*_(period|scenario)_policy_metrics.csv` fuera de `official/` | 4 | LEGACY ELIMINABLE | Eliminar y dejar de generarlos; duplican los CSV oficiales. |
| `dist/cu28_data_blob_20260604.*` | 3 | LEGACY ELIMINABLE | Eliminar. Conservar únicamente el bundle/manifiesto fechado `20260518`. |
| `docs/audit/` previo | 5 | LEGACY ELIMINABLE | Eliminar cuando quede sustituido por auditorías reproducibles vigentes. Incluye documentos de trabajo y una definición de ruta oficial ya obsoleta. |
| `scripts/data_processing.py`, `get_stats.py`, `policy_simulation.py`, `predict.py`, `train.py` | 5 | LEGACY ELIMINABLE | Eliminar. Son wrappers no referenciados; la CLI canónica es `python -m src.main`. |
| `data/metrics/README.md` | 1 | LEGACY ELIMINABLE | Eliminar junto con la ruta histórica vacía. |
| `.pytest_cache/`, `__pycache__/`, checkpoints, logs y outputs locales | regenerable | TEMPORAL / CACHE / BUILD | Eliminar localmente y bloquear en `.gitignore`. |
| `internal_archive/not_for_delivery/` | 0 | LEGACY ARCHIVABLE FUERA DE ENTREGA | No crear: ningún candidato requiere conservarse fuera de Git. |
| Candidatos sin función demostrable | 0 | AMBIGUO | No quedan candidatos ambiguos tras revisar referencias y manifiestos. |

## Evidencia de duplicación

- `legacy/` contiene 439 ficheros versionados y ocupa 539,39 MiB.
- Sus únicas referencias externas son notas documentales que lo declaran
  histórico; no existe import, lectura de pipeline, test ni entrada de
  manifiesto que dependa de esa carpeta.
- Los siete ficheros de `data/interim/external/legacy_raw_cache/` tienen el
  mismo SHA-256 que sus equivalentes bajo `data/raw/external/`.
- Existen simultáneamente artefactos de junio de 2026 y artefactos de la
  corrida oficial congelada, además de copias históricas de summaries y policy.
- El pipeline escribe hoy dos copias de cada comparison summary y dos copias
  de cada CSV de policy; se corregirá la generación para publicar una sola
  familia oficial.

## Valores obsoletos localizados

Los patrones prohibidos aparecen en snapshots históricos, artefactos de junio
de 2026 y en la auditoría manual inicial de consistencia. También aparecen
coincidencias numéricas legítimas en hiperparámetros, datos fila a fila o
diagnósticos; esas coincidencias no se tratarán como métricas CU28 obsoletas.

Tratamiento:

1. eliminar snapshots y artefactos anteriores;
2. retirar documentos manuales sustituidos;
3. excluir de los controles los usos legítimos no métricos;
4. añadir una auditoría contextual que falle solo cuando el valor se presente
   como resultado vigente, peso canónico o referencia oficial contradictoria.

## Fuentes de verdad que deben sobrevivir

- `reports/official/cu28_metrics_official__mixed_context.md`
- `reports/official/synthetic_procurement_need_formula__mixed_context.json`
- `models/metrics/official/`
- `models/metrics/summary/`
- `reports/audit/cu28_metrics_consistency_report.md`
- `scripts/audit_cu28_metrics_consistency.py`
- `config/config.yaml`
- `reproducibility_manifest__mixed_context.json`

## Resultado de la auditoría inicial

Estado: **CANDIDATOS IDENTIFICADOS**

La limpieza es segura con las decisiones anteriores. El estado final solo
podrá marcarse PASS después de eliminar los candidatos, regenerar artefactos y
ejecutar tests, auditoría de métricas, auditoría de hardening y reproducción
smoke.
