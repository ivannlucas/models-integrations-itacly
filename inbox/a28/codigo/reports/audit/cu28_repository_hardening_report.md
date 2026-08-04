# Auditoría de hardening del repositorio CU28

- Fecha de ejecución: `2026-06-26`
- Rama objetivo: `hardening/cu28-remove-legacy-stale-artifacts`
- Scope: `mixed_context`
- Fecha de referencia: `2026-05-18`
- Checks superados: `15/15`
- Resultado final: **PASS**

## Resultado de checks

| Check | Estado | Detalle |
|---|---|---|
| artefactos oficiales requeridos | PASS | todos presentes |
| bundle oficial 20260518 | PASS | dist\cu28_data_blob_20260518.zip |
| identidad canónica mixed_context | PASS | match |
| latest ligado al manifest y corrida oficial | PASS | hashes y run IDs válidos |
| estructura versionada sin legacy/snapshots | PASS | sin rutas obsoletas |
| familia única de summaries oficiales | PASS | whitelist exacta |
| dist contiene solo sidecars 20260518 | PASS | sidecars=['dist/cu28_data_blob_20260518.manifest.json', 'dist/cu28_data_blob_20260518.sha256'] |
| contenido vigente sin valores obsoletos | PASS | sin coincidencias contextuales |
| notebooks oficiales sin outputs embebidos | PASS | notebooks=8; outputs=0 |
| un único informe oficial de métricas | PASS | reports=['cu28_metrics_official__mixed_context.md'] |
| informe oficial deriva de JSON oficiales | PASS | valores oficiales presentes |
| ruta oficial documentada inequívocamente | PASS | contrato completo |
| README de entrega incluido en data blob | PASS | docs_snapshot_count=17 |
| rutas no entregables bloqueadas | PASS | exclusiones presentes |
| auditoría de consistencia de métricas | PASS | PASS |

## Ficheros eliminados

| Grupo | Cantidad | Tratamiento |
|---|---:|---|
| `legacy/deprecated_before_platform_reset/` | 430 | eliminado |
| `legacy/metrics_snapshots/` | 9 | eliminado |
| `data/interim/external/legacy_raw_cache/` | 7 | eliminado; duplicados SHA-256 de raw oficial |
| `artefactos y métricas de modelo run 20260604` | 12 | eliminado |
| `summaries *_comparison.{csv,json}` | 4 | eliminado y generación desactivada |
| `policy CSV históricos fuera de official/` | 4 | eliminado y generación desactivada |
| `dist/cu28_data_blob_20260604.*` | 3 | eliminado |
| `docs/audit/ manual previo` | 5 | eliminado por estar sustituido |
| `wrappers scripts/{data_processing,get_stats,policy_simulation,predict,train}.py` | 5 | eliminado |
| `data/metrics/README.md` | 1 | eliminado junto con la ruta histórica vacía |

## Ficheros movidos a internal_archive/not_for_delivery

- Ninguno. No se identificó contenido que justificara conservar una copia no entregable.

## Ficheros oficiales conservados

- `config/config.yaml`
- `DELIVERY_README_CU28.md`
- `reproducibility_manifest__mixed_context.json`
- `data_blob_manifest.json`
- `models/artifacts/model_manifest__mixed_context.json`
- `models/artifacts/upstream_predictor_latest__mixed_context.pkl`
- `models/artifacts/purchase_trigger_latest__mixed_context.pkl`
- `models/artifacts/quantity_optimizer_latest__mixed_context.pkl`
- `models/metrics/summary/baseline_comparison_latest__mixed_context.csv`
- `models/metrics/summary/baseline_comparison_latest__mixed_context.json`
- `models/metrics/summary/neuroevolution_comparison_latest__mixed_context.csv`
- `models/metrics/summary/neuroevolution_comparison_latest__mixed_context.json`
- `models/metrics/summary/trigger_metrics_latest__mixed_context.json`
- `models/metrics/summary/quantity_optimizer_latest__mixed_context.json`
- `models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.csv`
- `models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.json`
- `models/metrics/summary/policy_simulation_latest__mixed_context.csv`
- `models/metrics/summary/policy_simulation_latest__mixed_context.json`
- `models/metrics/summary/metrics_summary__mixed_context.csv`
- `models/metrics/summary/metrics_summary__mixed_context.json`
- `models/metrics/official/policy_simulation_period_latest__mixed_context.csv`
- `models/metrics/official/policy_simulation_scenario_latest__mixed_context.csv`
- `models/metrics/official/purchase_trigger_predictions_train__mixed_context.csv`
- `models/metrics/official/purchase_trigger_predictions_validation__mixed_context.csv`
- `models/metrics/official/purchase_trigger_predictions_test__mixed_context.csv`
- `models/metrics/official/quantity_optimizer_predictions_train__mixed_context.csv`
- `models/metrics/official/quantity_optimizer_predictions_validation__mixed_context.csv`
- `models/metrics/official/quantity_optimizer_predictions_test__mixed_context.csv`
- `reports/official/cu28_metrics_official__mixed_context.md`
- `reports/official/synthetic_procurement_need_formula__mixed_context.json`
- `reports/official/synthetic_procurement_need_formula__mixed_context.md`
- `reports/audit/cu28_metrics_consistency_report.md`
- `reports/audit/cu28_doc_metrics_alignment.json`
- `reports/audit/cu28_doc_metrics_alignment.md`
- `scripts/audit_cu28_metrics_consistency.py`
- `scripts/audit_doc_metrics_alignment.py`
- `dist/cu28_data_blob_20260518.manifest.json`
- `dist/cu28_data_blob_20260518.sha256`

## Valores obsoletos encontrados y tratamiento

- Métricas de regresión anteriores: eliminadas con snapshots y runs previos.
- Matriz de confusión anterior: eliminada de documentación y snapshots.
- Pesos anteriores de synthetic_procurement_need: eliminados de defaults y documentación vigente.
- Reducción y agregados de policy anteriores: eliminados de reports y summaries históricos.
- Coincidencias numéricas legítimas en datos fila a fila o hiperparámetros no se clasifican como métricas obsoletas.

## Rutas excluidas de entrega

- `outputs/`
- `data/interim/external/source_cache/`
- `tmp/`
- `temp/`
- `cache/`
- `.ipynb_checkpoints/`
- `__pycache__/`
- `models/metrics/experiments/`
- `models/artifacts/experiments/`
- `internal_archive/not_for_delivery/`

## Resumen de inventario

- Ficheros versionados auditados: `560`
- Summary files permitidos: `12`
- Artefactos oficiales requeridos: `38`
