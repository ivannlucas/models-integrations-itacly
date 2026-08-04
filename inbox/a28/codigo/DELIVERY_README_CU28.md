# Entrega blindada CU28 / NEUROCARN-OPT

## Identidad oficial

- Rama de entrega: `hardening/cu28-remove-legacy-stale-artifacts`
- Scope: `mixed_context`
- Fecha de referencia congelada: `2026-05-18`
- Modo defendible: batch/offline
- Configuración canónica: `config/config.yaml`

## Reproducción oficial

Desde PowerShell y con el entorno virtual preparado:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_mixed_context.py --config config/config.yaml --scope mixed_context --smoke --skip-download --use-cached-raw --run-notebooks --output-dir dist
```

La reproducción válida es end-to-end. Las etapas parciales no están
autorizadas a publicar artefactos `latest` oficiales.

## Fuentes de verdad

- Métricas consolidadas: `models/metrics/summary/`
- Filas oficiales para recálculo: `models/metrics/official/`
- Informe de métricas:
  `reports/official/cu28_metrics_official__mixed_context.md`
- Fórmula canónica:
  `reports/official/synthetic_procurement_need_formula__mixed_context.json`
- Auditoría de métricas:
  `reports/audit/cu28_metrics_consistency_report.md`
- Auditoría de hardening:
  `reports/audit/cu28_repository_hardening_report.md`
- Auditoría documental mínima:
  `reports/audit/cu28_docs_minimal_contract_report.md`
- Manifest de reproducibilidad:
  `reproducibility_manifest__mixed_context.json`
- Manifest del data blob: `data_blob_manifest.json`

Cualquier métrica fuera de `reports/official/`,
`models/metrics/official/` y `models/metrics/summary/` no es válida para
documentación, aceptación ni comunicación del CU28.

## Interpretación obligatoria

- La regresión lineal es el predictor upstream oficial.
- La neuroevolución es una comparativa experimental offline; no es el modelo
  promovido.
- Las fuentes públicas aportan contexto/proxy.
- Las variables internas de planta son sintéticas salvo sustitución explícita
  por datos observados del cliente.
- `platform_run` es una utilidad secundaria de inferencia CSV y no sustituye
  la reproducción oficial.

## Rutas excluidas

No deben usarse como fuente de resultados:

- `outputs/`
- caches, temporales y checkpoints
- artefactos de experimentos locales
- ficheros de backup
- bundles o manifests con fecha distinta de `2026-05-18`
- cualquier ruta `legacy/`, `old/`, `tmp/`, `backup/` o `archive/`

El repositorio no conserva `internal_archive/not_for_delivery/`. Si esa ruta se
crea en el futuro por una necesidad interna, queda expresamente excluida de la
entrega, del data blob y de toda validación o documentación del CU28.

## Validación mínima

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\audit_cu28_metrics_consistency.py --scope mixed_context --fail-on-mismatch
.\.venv\Scripts\python.exe scripts\audit_cu28_repository_hardening.py --scope mixed_context --fail-on-stale
.\.venv\Scripts\python.exe scripts\audit_cu28_docs_minimal_contract.py --scope mixed_context --fail-on-extra-docs
.\.venv\Scripts\python.exe scripts\package_data_blob.py --output-dir dist
.\.venv\Scripts\python.exe scripts\verify_data_blob.py --zip dist\cu28_data_blob_20260518.zip
```

## Contrato documental mínimo

`docs/` contiene únicamente:

- `docs/README.md`
- `docs/reproducibility.md`
- `docs/repository_outputs.md`
- `docs/data_lineage.md`
- `docs/data_sources_registry.md`
- `docs/data_blob_inventory.md`
- `docs/etl_pipeline.md`
- `docs/feature_engineering.md`
- `docs/input_contract.md`
- `docs/output_contract.md`
- `docs/leakage_policy.md`
- `docs/model_card_cu28.md`
- `docs/platform_usage.md`
- `docs/simulation_assumptions.md`
- `docs/simulation_data_basis.md`

Esta zona define alcance, uso, reproducibilidad, datos y contratos. Las
evidencias, auditorías y resultados generados se conservan en `reports/`.
