# CU28 mixed_context notebooks

Estos notebooks son evidencia analitica y documental del pipeline `mixed_context`.
Los `.ipynb` se versionan sin outputs embebidos. La evidencia ejecutada y
regenerable se publica en `reports/notebooks/`, `reports/figures/eda/` y
`reports/tables/eda/`; los notebooks no son fuente de verdad de metricas.

Orden de ejecucion:
- `00_data_sources_audit.ipynb`
- `01_raw_data_profile.ipynb`
- `02_external_context_eda.ipynb`
- `03_synthetic_plant_layer_eda.ipynb`
- `04_feature_engineering_audit.ipynb`
- `05_modeling_dataset_eda.ipynb`
- `06_split_validation_and_leakage_audit.ipynb`
- `07_training_and_policy_results_eda.ipynb`

Proposito por notebook:
- `00`: audita fuentes, estado y trazabilidad local.
- `01`: perfila los raw oficiales antes del ETL.
- `02`: analiza las senales externas/proxy procesadas.
- `03`: documenta la capa operativa sintetica de planta.
- `04`: audita feature engineering y leakage.
- `05`: analiza el dataset final modelable.
- `06`: valida splits cronologicos y separacion de test.
- `07`: visualiza entrenamiento, prediccion y resultados de politica.

Inputs esperados:
- `data/raw/external/*`
- `data/processed/external/context/*`
- `data/processed/synthetic/plant/*`
- `data/processed/baseline/*`
- `data/splits/baseline/default__mixed_context/*`
- `models/metrics/summary/*`
- `data/predictions/predictions_latest__mixed_context.csv`

Outputs generados:
- `reports/figures/eda/*.png`
- `reports/tables/eda/*.csv`
- `reports/notebooks/*.html`
- `reports/eda/eda_summary__mixed_context.md`
- `reports/eda/eda_summary__mixed_context.json`
- `reports/eda/notebook_execution_summary__mixed_context.json`

Ejecucion total:

```bash
python scripts/run_notebooks.py --scope mixed_context
```

Alternativa notebook por notebook:

```bash
jupyter nbconvert --execute --to html notebooks/00_data_sources_audit.ipynb --output-dir reports/notebooks/
```

Advertencia metodologica:
- Los notebooks son evidencia analitica; la ruta oficial sigue siendo CLI/script.
- La ruta oficial es `data_acquisition -> etl -> feature_engineering -> make_splits -> train -> predict -> policy_simulation -> get_stats`.
- Las variables internas de planta son sinteticas salvo carga posterior de cliente.
- Las fuentes reales son senales externas/proxy, no historicos internos completos de fabrica.
