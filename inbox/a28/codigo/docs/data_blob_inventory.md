# Data Blob Inventory

## Purpose

El blob oficial mixed_context existe para auditoria y trazabilidad. Se entrega fuera del flujo principal de plataforma batch/offline para no inflar el repositorio Git con paquetes pesados innecesarios.

## What the blob contains

- raw snapshots oficiales de `INE_CPI` y `MAPA_SLAUGHTER_MAPA`;
- raw snapshot trazado de `MAPA_PRICES_OM` como evidencia `fallback_constant`;
- procesados oficiales mixed_context;
- splits oficiales mixed_context;
- artefactos de modelo oficiales mixed_context;
- predicciones oficiales mixed_context;
- metricas oficiales mixed_context;
- manifest de reproducibilidad;
- CSV demo;
- snapshot de `README.md`, `DELIVERY_README_CU28.md` y de todos los contratos
  permitidos en `docs/`;
- `data_blob_manifest.json` y `SHA256SUMS.txt`.

## What the blob does not contain

- historicos internos completos de planta;
- compras reales observadas por cliente;
- notebooks exploratorios;
- logs;
- fuentes candidatas no activas;
- afirmaciones de validacion industrial final.

## Source status

- Active: `INE_CPI`, `MAPA_SLAUGHTER_MAPA`
- Traced but not active: `MAPA_PRICES_OM`
- Candidate only: `MAPA_CONSUMO_PANEL`, `DATACOMEX_TRADE_PRESSURE`, `EUROSTAT_SLAUGHTER`

## Synthetic variables

Las variables internas de planta siguen siendo sinteticas en la ruta demo y pasan a ser customer-provided solo cuando el cliente carga su CSV operativo. Los proxies externos del blob no convierten el dataset final en un historico real de fabrica carnica.

## Blob location

- zip: `dist/cu28_data_blob_<YYYYMMDD>.zip`
- manifest sidecar: `dist/cu28_data_blob_<YYYYMMDD>.manifest.json`
- zip sha256 sidecar: `dist/cu28_data_blob_<YYYYMMDD>.sha256`

## Verification

```powershell
python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip
```

El conjunto documental incluido debe coincidir exactamente con la whitelist
de `docs/README.md`; el manifest registra ruta, tamano y SHA-256 de cada
documento.

## Rebuild and reproduce

```powershell
python scripts/reproduce_mixed_context.py --config config/config.yaml --scope mixed_context --smoke
python scripts/build_data_manifest.py --output data_blob_manifest.json
python scripts/package_data_blob.py --output-dir dist
python scripts/verify_data_blob.py --zip dist/cu28_data_blob_<YYYYMMDD>.zip
```

## Source table

| source_id | status | official_url | download_url_or_endpoint | license_or_terms_url | raw_path | processed_artifacts | role | limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `INE_CPI` | `active` | `https://www.ine.es/dyngs/DAB/index.htm?cid=1722` | `https://www.ine.es/jaxiT3/files/t/csv_bdsc/76128.csv` | `https://www.ine.es/dyngs/AYU/en/index.htm?cid=125` | `data/raw/external/INE_CPI/76128.csv` | `external_long.csv`, `context_weekly_for_simulation.csv`, `feature_engineering_modeling__mixed_context.csv` | Contextual inflation proxy. | No representa compras reales ni inventario real de planta. |
| `MAPA_SLAUGHTER_MAPA` | `active` | `https://www.mapa.gob.es/es/estadistica/temas/estadisticas-agrarias/ganaderia/encuestas-sacrificio-ganado/` | `https://www.mapa.gob.es/es/estadistica/temas/estadisticas-agrarias/ganaderia/encuestas-sacrificio-ganado/` | `https://www.mapa.gob.es/es/atencion-al-ciudadano/aviso-legal` | `data/raw/external/MAPA_SLAUGHTER_MAPA/` | `external_long.csv`, `context_weekly_for_simulation.csv`, `feature_engineering_modeling__mixed_context.csv` | Contextual slaughter/supply proxy. | No es una serie observada de compras de materia prima. |
| `MAPA_PRICES_OM` | `traced` | `https://servicio.mapa.gob.es/es/alimentacion/temas/observatorio-cadena/cadenas-valor/sistema-de-precios-om` | `https://servicio.mapa.gob.es/dam/mapa/contenido/alimentacion/servicios/observatorio-de-precios-de-los-alimentos/sistema-de-informacion-de-precios-origen---destino/s132026rv0.xlsx` | `https://www.mapa.gob.es/es/atencion-al-ciudadano/aviso-legal` | `data/raw/external/MAPA_PRICES_OM/s502025rv0.xlsx` | `context_weekly_for_simulation.csv`, `context_proxy_limitations.json` | Fallback traced evidence only. | `purchase_price_index` queda como constante y no debe defenderse como feed semanal activo. |
