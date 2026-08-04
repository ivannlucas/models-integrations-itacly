# CU28 - NEUROCARN-OPT

Repositorio reproducible de modelo para CU28, orientado a reconstruccion end-to-end por CLI de una ruta `mixed_context` batch/offline para aprovisionamiento carnico. La linea oficial combina ETL de contexto externo, capa sintetica de planta, entrenamiento de modelos, prediccion, simulacion de politica, metricas, evidencia analitica y verificacion por manifests.

El repositorio no debe leerse como una demo de plataforma. El flujo promovido es la reproduccion auditable por fases y no la inferencia CSV como punto de entrada principal.

## Estado actual

- Linea oficial principal: `mixed_context`
- Ruta oficial: reproduccion end-to-end por CLI
- Modo de despliegue defendible: batch/offline
- Modelo/pipeline oficial:
  - `external context ETL`
  - `synthetic plant layer`
  - `upstream predictor`
  - `purchase trigger`
  - `quantity optimizer`
  - `policy simulation`
- Fuentes reales/proxy activas:
  - `INE_CPI`
  - `MAPA_SLAUGHTER_MAPA`
- Fuente trazada no activa: `MAPA_PRICES_OM` como `fallback_constant`
- KPI funcional:
  - `aggregate_excess_reduction_pct`
  - `stockout_guardrail_pass`
- Notebooks EDA: generados como evidencia analitica
- Data blob: verificable con hash

## Fuente única de verdad

- Pesos de `synthetic_procurement_need`: `config/config.yaml`.
- Métricas vigentes: `models/metrics/summary/*__mixed_context.json`.
- Filas usadas para recalcular: `models/metrics/official/*__mixed_context.csv`.
- Informe generado: `reports/official/cu28_metrics_official__mixed_context.md`.
- Auditoría: `reports/audit/cu28_metrics_consistency_report.md`.

No se conservan snapshots históricos en la entrega. Cualquier métrica fuera de
`reports/official/`, `models/metrics/official/` y
`models/metrics/summary/` carece de validez documental. Las métricas se copian
desde los artefactos oficiales regenerados, nunca desde ejecuciones parciales.

## Linea oficial principal

La linea oficial es `mixed_context`. Su reconstruccion defendible es por CLI y por fases:

- `data_acquisition` restaura o verifica snapshots raw oficiales/proxy.
- `etl` transforma raw en contexto semanal, capa sintetica y dataset modelable.
- `feature_engineering` materializa el dataset oficial de modelado.
- `make_splits` persiste `train`, `validation` y `test`.
- `train` reentrena los artefactos oficiales.
- `predict` genera predicciones sobre el split oficial de test o un input explicito.
- `policy_simulation` evalua la politica de compra.
- `get_stats` resume metricas agregadas.
- `run_notebooks` genera evidencia EDA.
- `verify_reproducibility` y el data blob fijan trazabilidad verificable.

## Rutas relevantes del repositorio

```text
cu28-neurocarn-opt/
|-- config/
|   |-- config.yaml
|   |-- platform_config.yaml
|   `-- manufacturing_profiles.yaml
|-- data/
|   |-- raw/external/
|   |-- interim/
|   |-- processed/external/context/
|   |-- processed/synthetic/plant/
|   |-- processed/baseline/
|   |-- splits/baseline/default__mixed_context/
|   |-- predictions/
|   `-- demo/
|-- docs/
|   |-- README.md
|   |-- data_blob_inventory.md
|   |-- data_lineage.md
|   |-- data_sources_registry.md
|   |-- etl_pipeline.md
|   |-- feature_engineering.md
|   |-- input_contract.md
|   |-- leakage_policy.md
|   |-- model_card_cu28.md
|   |-- output_contract.md
|   |-- platform_usage.md
|   |-- repository_outputs.md
|   |-- reproducibility.md
|   |-- simulation_assumptions.md
|   `-- simulation_data_basis.md
|-- models/
|   |-- artifacts/
|   `-- metrics/
|       |-- official/
|       `-- summary/
|-- notebooks/
|-- reports/
|   |-- audit/
|   |-- official/
|   |-- figures/eda/
|   |-- tables/eda/
|   |-- notebooks/
|   `-- eda/
|-- scripts/
|-- src/
|   |-- data_acquisition/
|   |-- etl/
|   |-- feature_engineering/
|   |-- training/
|   |-- prediction/
|   |-- simulation/
|   |-- evaluation/
|   |-- reproducibility/
|   `-- cli/
|-- tests/
|-- dist/
`-- README.md
```

## Instalacion

Usar el mismo interprete para instalar y ejecutar la CLI.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Entorno minimo

- Python soportado: `>=3.12,<3.13`
- Entorno verificado: Windows + PowerShell
- Ejecucion: local, batch/offline
- CPU: suficiente para pipeline y notebooks oficiales
- GPU: no requerida
- RAM recomendada: `>=4 GB`
- Limites operativos:
  - no tiempo real
  - no compra autonoma
  - no validacion industrial final
  - fuentes externas reales/proxy
  - variables internas sinteticas o cargadas por cliente

## CU28 reproducibility quick path

Comandos oficiales de la linea reproducible `mixed_context`, ejecutables por separado:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .

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
python scripts/audit_cu28_docs_minimal_contract.py --scope mixed_context --fail-on-extra-docs
python scripts/package_data_blob.py --output-dir dist
python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip
```

Orquestador oficial de conveniencia:

```powershell
python scripts/reproduce_mixed_context.py --config config/config.yaml --scope mixed_context --full --run-notebooks
```

Aclaraciones:

- el orquestador no sustituye los comandos por fase;
- solo automatiza la secuencia oficial;
- cada fase puede ejecutarse, auditarse y repetirse por separado.

## Comandos oficiales por fase

| Fase | Comando | Input principal | Output principal | Evidencia |
| --- | --- | --- | --- | --- |
| Data acquisition | `python -m src.main data_acquisition --mixed-context` | `data/raw/external/*` cacheado o descargable | `data/raw/external/INE_CPI/`; `data/raw/external/MAPA_SLAUGHTER_MAPA/`; `data/raw/external/MAPA_PRICES_OM/`; `data/raw/external/raw_manifest__mixed_context.json` | `data/raw/external/*/source_manifest.json` |
| ETL | `python -m src.main etl --mixed-context` | `data/raw/external/` | `data/processed/external/context/external_long.csv`; `data/processed/external/context/context_weekly_for_simulation.csv`; `data/processed/external/context/context_proxy_limitations.json`; `data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv`; `data/processed/synthetic/plant/synthetic_plant_metadata__mixed_context.json`; `data/processed/baseline/feature_engineering_modeling__mixed_context.csv`; `data/processed/baseline/modeling_metadata__mixed_context.json` | `data/interim/**` y `docs/etl_pipeline.md` |
| Feature engineering | `python -m src.main feature_engineering --mixed-context` | `data/processed/baseline/modeling_weekly__mixed_context.csv` y contexto procesado | `data/processed/baseline/feature_engineering_modeling__mixed_context.csv`; `data/processed/baseline/modeling_metadata__mixed_context.json` | `docs/feature_engineering.md` |
| Splits | `python -m src.main make_splits --mixed-context` | `data/processed/baseline/feature_engineering_modeling__mixed_context.csv` | `data/splits/baseline/default__mixed_context/train.csv`; `data/splits/baseline/default__mixed_context/validation.csv`; `data/splits/baseline/default__mixed_context/test.csv`; `data/splits/baseline/default__mixed_context/split_metadata.json` | `split_metadata.json` |
| Training | `python -m src.main train --mixed-context` | `data/splits/baseline/default__mixed_context/` | `models/artifacts/upstream_predictor_latest__mixed_context.pkl`; `models/artifacts/purchase_trigger_latest__mixed_context.pkl`; `models/artifacts/quantity_optimizer_latest__mixed_context.pkl`; `models/artifacts/model_manifest__mixed_context.json`; `models/metrics/summary/baseline_comparison_latest__mixed_context.json`; `models/metrics/summary/neuroevolution_comparison_latest__mixed_context.json`; `models/metrics/summary/trigger_metrics_latest__mixed_context.json`; `models/metrics/summary/quantity_optimizer_latest__mixed_context.json` | `model_manifest__mixed_context.json` |
| Prediction | `python -m src.main predict --mixed-context` | `data/splits/baseline/default__mixed_context/test.csv` por defecto | `data/predictions/predictions_latest__mixed_context.csv` | `predictions_latest__mixed_context.csv` |
| Policy simulation | `python -m src.main policy_simulation --mixed-context` | `data/predictions/predictions_latest__mixed_context.csv` y artefactos oficiales | `models/metrics/summary/policy_simulation_latest__mixed_context.json`; `models/metrics/summary/policy_simulation_latest__mixed_context.csv` | `policy_simulation_latest__mixed_context.json` |
| Metrics summary | `python -m src.main get_stats --mixed-context` | `models/metrics/summary/` | `models/metrics/summary/metrics_summary__mixed_context.json`; `models/metrics/summary/metrics_summary__mixed_context.csv` | `metrics_summary__mixed_context.json` |
| EDA notebooks | `python scripts/run_notebooks.py --scope mixed_context` | artefactos del pipeline oficial | `reports/notebooks/*.html`; `reports/figures/eda/*.png`; `reports/tables/eda/*.csv`; `reports/eda/eda_summary__mixed_context.md`; `reports/eda/eda_summary__mixed_context.json` | `reports/eda/notebook_execution_summary__mixed_context.json` |
| Reproducibility verification | `python -m src.reproducibility.verify_reproducibility --manifest reproducibility_manifest__mixed_context.json` | `reproducibility_manifest__mixed_context.json` | verificacion `valid=<bool>` y conteo de hashes | `reproducibility_manifest__mixed_context.json` |
| Data blob | `python scripts/package_data_blob.py --output-dir dist` y `python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip` | artefactos oficiales ya generados | `dist/cu28_data_blob_<YYYYMMDD>.zip`; `dist/cu28_data_blob_<YYYYMMDD>.manifest.json`; `dist/cu28_data_blob_<YYYYMMDD>.sha256` | `data_blob_manifest.json` y sidecar `.sha256` |

## Artefactos generados por fase

- Adquisicion y trazabilidad raw: `data/raw/external/`, `data/raw/external/*/source_manifest.json`, `data/raw/external/raw_manifest__mixed_context.json`
- ETL y contexto: `data/processed/external/context/`
- Capa sintetica de planta: `data/processed/synthetic/plant/`
- Dataset modelable y metadata: `data/processed/baseline/`
- Splits persistidos: `data/splits/baseline/default__mixed_context/`
- Modelos oficiales: `models/artifacts/`
- Metricas oficiales: `models/metrics/summary/`
- Predicciones: `data/predictions/`
- Evidencia analitica: `reports/notebooks/`, `reports/figures/eda/`, `reports/tables/eda/`, `reports/eda/`
- Auditoria reproducible: `reproducibility_manifest__mixed_context.json`, `data_blob_manifest.json`, `dist/`

## ETL y reconstrucción de datos

La ruta `mixed_context` reconstruye un caso defendible a partir de fuentes externas/proxy y una capa sintetica de planta declarada. No reconstruye historicos reales de fabrica.

- `data_acquisition` restaura o verifica snapshots raw oficiales/proxy.
- `etl` transforma esos raw en contexto semanal y dataset modelable.
- `feature_engineering` persiste el dataset oficial de entrenamiento/prediccion.
- las fuentes externas no son datos reales de fabrica;
- las variables internas de planta son sinteticas salvo carga posterior de cliente.

## Reconstrucción desde fuentes raw

Los raw oficiales/proxy viven en `data/raw/external/`.

- `data_acquisition` crea o verifica snapshots.
- `etl` transforma raw en contexto semanal y dataset modelable.
- `INE_CPI` y `MAPA_SLAUGHTER_MAPA` son las fuentes contextuales activas.
- `MAPA_PRICES_OM` queda trazada como fallback y no como feed semanal activo.

Comandos:

```powershell
python -m src.main data_acquisition --mixed-context
python -m src.main etl --mixed-context
```

Fallback soportado por CLI:

```powershell
python -m src.main data_acquisition --mixed-context --use-cached-raw
python -m src.main data_acquisition --mixed-context --fail-on-missing-raw
```

## Entrenamiento y reentrenamiento

`train --mixed-context` reentrena los artefactos oficiales desde los splits persistidos. El entrenamiento usa `train` y `validation`. `test` queda reservado para evaluacion final. La seleccion y configuracion no debe hacerse con `test`. Los artefactos quedan en `models/artifacts/` y las metricas en `models/metrics/summary/`.

Comandos base:

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

Notas sobre modos:

- `--smoke` y `--full` aplican a `data_acquisition`, `etl`, `feature_engineering`, `make_splits`, `train`, `predict`, `policy_simulation`, `get_stats` y `scripts/reproduce_mixed_context.py`.
- `scripts/run_notebooks.py` expone `--smoke`, pero no `--full`.

## Predicción

Comando oficial:

```powershell
python -m src.main predict --mixed-context
```

Output oficial:

- `data/predictions/predictions_latest__mixed_context.csv`

En la ruta reproducible, `predict --mixed-context` usa por defecto `data/splits/baseline/default__mixed_context/test.csv` si no se indica `--input`.

## Politica de Simulación

Comando oficial:

```powershell
python -m src.main policy_simulation --mixed-context
```

Outputs principales:

- `models/metrics/summary/policy_simulation_latest__mixed_context.json`
- `models/metrics/summary/policy_simulation_latest__mixed_context.csv`

La simulacion compara la politica propuesta frente a referencias batch/offline y expone KPI funcionales como `aggregate_excess_reduction_pct` y `stockout_guardrail_pass`.

## Métricas

Comando oficial:

```powershell
python -m src.main get_stats --mixed-context
```

Outputs principales:

- `models/metrics/summary/metrics_summary__mixed_context.json`
- `models/metrics/summary/metrics_summary__mixed_context.csv`

Las metricas oficiales viven bajo `models/metrics/summary/` y cubren baseline, comparativa neuroevolutiva, trigger, optimizer, simulacion y resumen agregado.

## Notebooks EDA

Los notebooks de `notebooks/` son evidencia analitica de la linea oficial y se ejecutan despues del pipeline CLI.

```powershell
python scripts/run_notebooks.py --scope mixed_context
```

Outputs:

- `reports/notebooks/*.html`
- `reports/figures/eda/*.png`
- `reports/tables/eda/*.csv`
- `reports/eda/eda_summary__mixed_context.md`
- `reports/eda/eda_summary__mixed_context.json`

No sustituyen la ruta oficial por CLI.

## Data blob

El data blob empaqueta la evidencia defendible de `mixed_context` para auditoria offline.

```powershell
python scripts/package_data_blob.py --output-dir dist
python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip
```

Outputs:

- `dist/cu28_data_blob_<YYYYMMDD>.zip`
- `dist/cu28_data_blob_<YYYYMMDD>.manifest.json`
- `dist/cu28_data_blob_<YYYYMMDD>.sha256`

## Fingerprints, manifests y verificación

Manifests principales:

- `data/raw/external/raw_manifest__mixed_context.json`
- `data_blob_manifest.json`
- `reproducibility_manifest__mixed_context.json`

Comandos:

```powershell
python -m src.reproducibility.verify_reproducibility --manifest reproducibility_manifest__mixed_context.json
python scripts/package_data_blob.py --output-dir dist
python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip
```

Estos manifests contienen, segun su tipo:

- rutas relativas;
- tamano;
- `SHA256`;
- fecha de generacion;
- `scope`;
- `commit`;
- configuracion referenciada;
- outputs generados y rutas requeridas.

## Ejecución batch sobre CSV de cliente

```powershell
python -m src.main platform_run --input data/demo/customer_upload_example.csv --output outputs/demo_run/
```

El CSV demo representa una carga operativa minima de materia prima carnica: fecha, materia prima, inventario, requerimiento previsto, lead time, cobertura de seguridad, yield, waste, coste, vida util y destino productivo. `destination_profile` es el destino productivo previsto de la materia prima; no significa que el sistema recomiende comprar producto terminado.

Columnas minimas:

- `date`
- `raw_material_id`
- `destination_profile`
- `current_inventory_tons`
- `expected_requirement_tons`
- `lead_time_days`
- `safety_coverage_days`
- `expected_yield_rate`
- `expected_waste_rate`
- `unit_purchase_cost`
- `shelf_life_days`

`platform_run` imprime en consola:

- resumen de estado (`platform_status`, `row_count`, `triggered_orders`, metricas agregadas);
- rutas de outputs;
- tabla `RECOMMENDED PURCHASES` con materia prima, destino productivo, accion `BUY`/`DO_NOT_BUY`, cantidad final, motivo, excedente y stockout estimados;
- secciones `INTERPRETATION` y `DATA USE` para aclarar que el comando ejecuta recomendaciones y no reentrena.

Outputs esperados de esta utilidad:

- `outputs/demo_run/validation_report.json`
- `outputs/demo_run/recommendations.csv`
- `outputs/demo_run/policy_simulation_results.csv`
- `outputs/demo_run/summary_metrics.json`

El valor de la demo no es únicamente devolver una cantidad en toneladas. El sistema devuelve una recomendación trazable: acción comprar/no comprar, cantidad final recomendada, motivo, riesgo estimado y comparación funcional frente a baseline.

Si solo se observa `order_quantity_tons`, el sistema puede parecer una calculadora de toneladas. La lectura correcta es la recomendación completa: trigger + cantidad + riesgo + simulación frente a baseline.

`platform_run` es una utilidad de inferencia batch sobre un CSV de cliente o demo. No sustituye la ruta oficial de reproduccion `mixed_context`, no reentrena modelos y no reconstruye el ETL completo.

## Documentacion principal

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

`docs/` contiene solo contrato tecnico estable. Las metricas y resultados
vigentes se publican en `models/metrics/summary/`,
`models/metrics/official/` y `reports/official/`; las auditorias y la evidencia
generada viven en `reports/`.

## Exclusiones de entrega

Las salidas locales bajo `outputs/`, caches, temporales y artefactos de
experimentos no forman parte de la entrega ni son fuente de métricas. El
repositorio no conserva una carpeta `legacy/` entregable.
