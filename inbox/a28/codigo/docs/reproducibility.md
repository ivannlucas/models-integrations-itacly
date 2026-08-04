# Reproducibility

## Scope

La ruta oficial reproducible de CU28 es `mixed_context`. Reconstruye por CLI el pipeline defendible desde raw externos/proxy hasta ETL, capa sintetica de planta, feature engineering, splits, entrenamiento, prediccion, simulacion, metricas, manifests y data blob.

La ventana oficial queda fijada por `project.reference_date` en `config/config.yaml`. Para v2.0.1 la fecha de corte es `2026-05-18`; cambiarla es una ejecucion nueva y no reproduce la corrida documental.

`platform_run` queda fuera de esta ruta:

- `mixed_context` es la reproduccion oficial auditable.
- `platform_run` es solo inferencia batch/offline sobre un CSV de cliente o demo.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick path

```powershell
python -m src.main data_acquisition --mixed-context
python -m src.main etl --mixed-context
python -m src.main feature_engineering --mixed-context
python -m src.main make_splits --mixed-context
python -m src.main train --mixed-context
python -m src.main predict --mixed-context
python -m src.main policy_simulation --mixed-context
python -m src.main get_stats --mixed-context
python scripts/run_notebooks.py --scope mixed_context
python -m src.reproducibility.verify_reproducibility --manifest reproducibility_manifest__mixed_context.json
python scripts/package_data_blob.py --output-dir dist
python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip
```

Orquestador oficial:

```powershell
python scripts/reproduce_mixed_context.py --config config/config.yaml --scope mixed_context --smoke --run-notebooks
python scripts/reproduce_mixed_context.py --config config/config.yaml --scope mixed_context --full
```

El orquestador automatiza la secuencia oficial, pero no sustituye la ejecucion y auditoria por fases.

Las metricas no se documentan manualmente en `docs/`. Los resultados vigentes
se leen de `reports/official/cu28_metrics_official__mixed_context.md`,
`models/metrics/summary/` y `models/metrics/official/`.

## Reproducción por fases

| Paso | Comando | Genera | Se puede reejecutar | Requiere raw |
| --- | --- | --- | --- | --- |
| Data acquisition | `python -m src.main data_acquisition --mixed-context` | `data/raw/external/*`; `data/raw/external/raw_manifest__mixed_context.json` | Si | Si |
| ETL | `python -m src.main etl --mixed-context` | `data/processed/external/context/*`; `data/processed/synthetic/plant/*`; `data/processed/baseline/feature_engineering_modeling__mixed_context.csv` | Si | Si |
| Feature engineering | `python -m src.main feature_engineering --mixed-context` | `data/processed/baseline/feature_engineering_modeling__mixed_context.csv`; `data/processed/baseline/modeling_metadata__mixed_context.json` | Si | No, si el ETL ya produjo el dataset base |
| Splits | `python -m src.main make_splits --mixed-context` | `data/splits/baseline/default__mixed_context/*` | Si | No, si existe el dataset modelable |
| Train | `python -m src.main train --mixed-context` | `models/artifacts/*__mixed_context.*`; `models/metrics/summary/baseline_comparison_latest__mixed_context.json`; `models/metrics/summary/neuroevolution_comparison_latest__mixed_context.json`; `models/metrics/summary/trigger_metrics_latest__mixed_context.json`; `models/metrics/summary/quantity_optimizer_latest__mixed_context.json` | Si | No, si existen splits persistidos |
| Predict | `python -m src.main predict --mixed-context` | `data/predictions/predictions_latest__mixed_context.csv` | Si | No, si existen splits y artefactos |
| Policy simulation | `python -m src.main policy_simulation --mixed-context` | `models/metrics/summary/policy_simulation_latest__mixed_context.json`; `models/metrics/summary/policy_simulation_latest__mixed_context.csv` | Si | No, si existen predicciones y artefactos |
| Metrics | `python -m src.main get_stats --mixed-context` | `models/metrics/summary/metrics_summary__mixed_context.json`; `models/metrics/summary/metrics_summary__mixed_context.csv` | Si | No |
| Notebooks | `python scripts/run_notebooks.py --scope mixed_context` | `reports/notebooks/*.html`; `reports/figures/eda/*.png`; `reports/tables/eda/*.csv`; `reports/eda/eda_summary__mixed_context.*` | Si | No, si ya existen artefactos de pipeline |
| Verification | `python -m src.reproducibility.verify_reproducibility --manifest reproducibility_manifest__mixed_context.json` | verificacion hash/paths | Si | No |
| Data blob | `python scripts/package_data_blob.py --output-dir dist` y `python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip` | `dist/cu28_data_blob_<YYYYMMDD>.zip`; `.manifest.json`; `.sha256` | Si | No, si ya existen artefactos oficiales |

## Reentrenamiento

`train --mixed-context` reentrena desde `data/splits/baseline/default__mixed_context/`.

- usa `train` y `validation`;
- reserva `test` para evaluacion final;
- no debe seleccionar configuracion con `test`;
- persiste artefactos en `models/artifacts/`;
- persiste metricas en `models/metrics/summary/`.

Comandos:

```powershell
python -m src.main make_splits --mixed-context
python -m src.main train --mixed-context
python -m src.main predict --mixed-context
python -m src.main policy_simulation --mixed-context
python -m src.main get_stats --mixed-context
```

Modos soportados:

```powershell
python -m src.main train --mixed-context --smoke
python -m src.main train --mixed-context --full
```

`--smoke` y `--full` aplican tambien a `data_acquisition`, `etl`, `feature_engineering`, `make_splits`, `predict`, `policy_simulation`, `get_stats` y `scripts/reproduce_mixed_context.py`. `scripts/run_notebooks.py` solo expone `--smoke`.

## Reconstrucción desde raw

```powershell
python -m src.main data_acquisition --mixed-context
python -m src.main etl --mixed-context
python -m src.main data_acquisition --mixed-context --use-cached-raw
python -m src.main data_acquisition --mixed-context --fail-on-missing-raw
```

Limites declarados:

- los raw oficiales/proxy viven en `data/raw/external/`;
- las fuentes externas no son datos reales de fabrica;
- las variables internas de planta son sinteticas salvo carga posterior de cliente;
- `MAPA_PRICES_OM` se traza como fallback y no como feed semanal activo.
- `MAPA_SLAUGHTER_MAPA` cubre desde 2021 en la snapshot local; `supply_index` queda como `NaN` antes del primer dato observado y se imputa solo dentro de entrenamiento/inferencia.
- La medida de adaptacion a datos reales es un criterio operativo/documental en esta version; no existe detector automatico de drift.

## Outputs clave

- `data/raw/external/raw_manifest__mixed_context.json`
- `data/processed/external/context/external_long.csv`
- `data/processed/external/context/context_weekly_for_simulation.csv`
- `data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv`
- `data/processed/baseline/feature_engineering_modeling__mixed_context.csv`
- `data/splits/baseline/default__mixed_context/{train.csv,validation.csv,test.csv,split_metadata.json}`
- `models/artifacts/{upstream_predictor_latest__mixed_context.pkl,purchase_trigger_latest__mixed_context.pkl,quantity_optimizer_latest__mixed_context.pkl,model_manifest__mixed_context.json}`
- `data/predictions/predictions_latest__mixed_context.csv`
- `models/metrics/summary/*.json`
- `reports/eda/eda_summary__mixed_context.json`
- `reproducibility_manifest__mixed_context.json`
- `dist/cu28_data_blob_<YYYYMMDD>.zip`

## Verificación y manifests

Archivos de trazabilidad:

- `data/raw/external/raw_manifest__mixed_context.json`
- `data_blob_manifest.json`
- `reproducibility_manifest__mixed_context.json`

Comandos:

```powershell
python -m src.reproducibility.verify_reproducibility --manifest reproducibility_manifest__mixed_context.json
python scripts/package_data_blob.py --output-dir dist
python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip
python scripts/audit_cu28_docs_minimal_contract.py --scope mixed_context --fail-on-extra-docs
```

Los manifests recogen, segun el caso, rutas relativas, tamano, `SHA256`, fecha de generacion, `scope`, `commit`, configuracion referenciada y outputs generados.

El manifest oficial `reproducibility_manifest__mixed_context.json` representa una ejecucion end-to-end del orquestador. Una ejecucion parcial de `get_stats` escribe `reproducibility_manifest_partial__mixed_context.json` con `manifest_scope = partial_get_stats`.
