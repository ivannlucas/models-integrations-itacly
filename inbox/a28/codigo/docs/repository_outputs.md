# Repository Outputs

## Data acquisition

Comando:

```powershell
python -m src.main data_acquisition --mixed-context
```

Outputs:

- `data/raw/external/INE_CPI/`
- `data/raw/external/MAPA_SLAUGHTER_MAPA/`
- `data/raw/external/MAPA_PRICES_OM/`
- `data/raw/external/raw_manifest__mixed_context.json`
- `data/raw/external/*/source_manifest.json`

## ETL

Comando:

```powershell
python -m src.main etl --mixed-context
```

Outputs:

- `data/processed/external/context/external_long.csv`
- `data/processed/external/context/context_weekly_for_simulation.csv`
- `data/processed/external/context/context_proxy_limitations.json`
- `data/processed/external/context/download_registry.json`
- `data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv`
- `data/processed/synthetic/plant/synthetic_plant_metadata__mixed_context.json`
- `data/processed/baseline/feature_engineering_modeling__mixed_context.csv`
- `data/processed/baseline/modeling_metadata__mixed_context.json`
- `data/processed/baseline/modeling_weekly__mixed_context.csv`

## Feature engineering

Comando:

```powershell
python -m src.main feature_engineering --mixed-context
```

Outputs:

- `data/processed/baseline/feature_engineering_modeling__mixed_context.csv`
- `data/processed/baseline/modeling_metadata__mixed_context.json`
- `data/processed/baseline/feature_catalog__mixed_context.json`
- `data/processed/baseline/feature_contract__mixed_context.json`
- `data/processed/baseline/feature_roles_metadata__mixed_context.json`
- `data/processed/baseline/feature_selection__mixed_context.json`

## Splits

Comando:

```powershell
python -m src.main make_splits --mixed-context
```

Outputs:

- `data/splits/baseline/default__mixed_context/train.csv`
- `data/splits/baseline/default__mixed_context/validation.csv`
- `data/splits/baseline/default__mixed_context/test.csv`
- `data/splits/baseline/default__mixed_context/split_metadata.json`

## Training

Comando:

```powershell
python -m src.main train --mixed-context
```

Outputs:

- `models/artifacts/upstream_predictor_latest__mixed_context.pkl`
- `models/artifacts/purchase_trigger_latest__mixed_context.pkl`
- `models/artifacts/quantity_optimizer_latest__mixed_context.pkl`
- `models/artifacts/model_manifest__mixed_context.json`
- `models/metrics/summary/baseline_comparison_latest__mixed_context.csv`
- `models/metrics/summary/baseline_comparison_latest__mixed_context.json`
- `models/metrics/summary/neuroevolution_comparison_latest__mixed_context.csv`
- `models/metrics/summary/neuroevolution_comparison_latest__mixed_context.json`
- `models/metrics/summary/trigger_metrics_latest__mixed_context.json`
- `models/metrics/summary/quantity_optimizer_latest__mixed_context.json`
- `models/metrics/official/purchase_trigger_predictions_*__mixed_context.csv`
- `models/metrics/official/quantity_optimizer_predictions_*__mixed_context.csv`

## Prediction

Comando:

```powershell
python -m src.main predict --mixed-context
```

Output:

- `data/predictions/predictions_latest__mixed_context.csv`

## Policy simulation

Comando:

```powershell
python -m src.main policy_simulation --mixed-context
```

Outputs:

- `models/metrics/summary/policy_simulation_latest__mixed_context.json`
- `models/metrics/summary/policy_simulation_latest__mixed_context.csv`
- `models/metrics/official/policy_simulation_period_latest__mixed_context.csv`
- `models/metrics/official/policy_simulation_scenario_latest__mixed_context.csv`

## Metrics

Comando:

```powershell
python -m src.main get_stats --mixed-context
```

Outputs:

- `models/metrics/summary/metrics_summary__mixed_context.json`
- `models/metrics/summary/metrics_summary__mixed_context.csv`
- `reports/official/cu28_metrics_official__mixed_context.md`
- `reports/official/synthetic_procurement_need_formula__mixed_context.json`
- `reports/audit/cu28_metrics_consistency_report.md`

No se conservan snapshots históricos en la entrega. Para documentación solo
son válidos `models/metrics/summary/`, `models/metrics/official/` y
`reports/official/`.

Estas tres rutas son la unica fuente de metricas oficiales. `reports/audit/`
contiene verificaciones y evidencia de auditoria; no redefine resultados.

## Notebooks

Comando:

```powershell
python scripts/run_notebooks.py --scope mixed_context
```

Outputs:

- `reports/notebooks/*.html`
- `reports/figures/eda/*.png`
- `reports/tables/eda/*.csv`
- `reports/eda/eda_summary__mixed_context.md`
- `reports/eda/eda_summary__mixed_context.json`
- `reports/eda/notebook_execution_summary__mixed_context.json`

## Data blob

Comandos:

```powershell
python scripts/package_data_blob.py --output-dir dist
python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip
```

Outputs:

- `data_blob_manifest.json`
- `dist/cu28_data_blob_<YYYYMMDD>.zip`
- `dist/cu28_data_blob_<YYYYMMDD>.manifest.json`
- `dist/cu28_data_blob_<YYYYMMDD>.sha256`

## Platform CSV run

Comando:

```powershell
python -m src.main platform_run --input data/demo/customer_upload_example.csv --output outputs/demo_run/
```

Outputs:

- `outputs/demo_run/validation_report.json`
- `outputs/demo_run/recommendations.csv` con accion comprar/no comprar, cantidad final, motivo, riesgo y comparacion frente a baseline
- `outputs/demo_run/policy_simulation_results.csv`
- `outputs/demo_run/summary_metrics.json`

Estos outputs son regenerables por `platform_run`. El comando no reentrena modelos; ejecuta una recomendacion sobre el CSV de entrada.

`outputs/` es una zona local y regenerable. No forma parte de la evidencia
oficial, no es fuente de metricas oficiales y no sustituye los artefactos
`mixed_context`.
