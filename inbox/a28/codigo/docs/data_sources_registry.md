# Data Sources Registry

## Official mixed_context scope

Las fuentes externas del blob oficial se limitan a proxies contextuales para la ruta mixed_context. No se presentan como datos reales de fabrica carnica ni como sustituto de variables internas de planta.

## Active and traced sources

| source_id | status | official_url | download_url_or_endpoint | access_date | license_or_terms_url | local_raw_path | processed_artifact_path | redistribution_allowed | role | limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `INE_CPI` | `active` | `https://www.ine.es/dyngs/DAB/index.htm?cid=1722` | `https://www.ine.es/jaxiT3/files/t/csv_bdsc/76128.csv` | `2026-05-18` | `https://www.ine.es/dyngs/AYU/en/index.htm?cid=125` | `data/raw/external/INE_CPI/76128.csv` | `data/processed/external/context/external_long.csv`, `data/processed/external/context/context_weekly_for_simulation.csv`, `data/processed/baseline/feature_engineering_modeling__mixed_context.csv` | `yes_with_attribution_cc_by_4_0` | Contextual inflation and price-pressure proxy. | No representa compras reales de materia prima ni inventario de planta. |
| `MAPA_SLAUGHTER_MAPA` | `active` | `https://www.mapa.gob.es/es/estadistica/temas/estadisticas-agrarias/ganaderia/encuestas-sacrificio-ganado/` | `https://www.mapa.gob.es/es/estadistica/temas/estadisticas-agrarias/ganaderia/encuestas-sacrificio-ganado/` | `2026-05-18` | `https://www.mapa.gob.es/es/atencion-al-ciudadano/aviso-legal` | `data/raw/external/MAPA_SLAUGHTER_MAPA/` | `data/processed/external/context/external_long.csv`, `data/processed/external/context/context_weekly_for_simulation.csv`, `data/processed/baseline/feature_engineering_modeling__mixed_context.csv` | `yes_with_source_citation_unless_third_party_rights_apply` | Contextual slaughter/supply proxy. | Proxy macro sectorial, no ledger semanal de compras de una planta carnica. |
| `MAPA_PRICES_OM` | `traced` | `https://servicio.mapa.gob.es/es/alimentacion/temas/observatorio-cadena/cadenas-valor/sistema-de-precios-om` | `https://servicio.mapa.gob.es/dam/mapa/contenido/alimentacion/servicios/observatorio-de-precios-de-los-alimentos/sistema-de-informacion-de-precios-origen---destino/s132026rv0.xlsx` | `2026-05-18` | `https://www.mapa.gob.es/es/atencion-al-ciudadano/aviso-legal` | `data/raw/external/MAPA_PRICES_OM/s502025rv0.xlsx` | `data/processed/external/context/context_weekly_for_simulation.csv`, `data/processed/external/context/context_proxy_limitations.json` | `yes_with_source_citation_unless_third_party_rights_apply` | Fallback traced evidence only. | La ruta oficial no defiende una serie semanal activa; `purchase_price_index` queda como `fallback_constant=100`. |

## Candidate sources not active in the official blob

- `MAPA_CONSUMO_PANEL`
  Candidata historica no empaquetada. No alimenta la ruta mixed_context defendible actual.
- `DATACOMEX_TRADE_PRESSURE`
  Candidata/referenciada en limitaciones, pero no integrada en `external_long.csv` ni en el blob oficial.
- `EUROSTAT_SLAUGHTER`
  Candidata/referenciada en limitaciones, pero no integrada en el contexto procesado oficial.

## Internal variables

- variables como inventario, requirement, lead time, waste, yield, stockout y ordenes observadas siguen siendo sinteticas salvo carga de cliente;
- el dataset demo no representa un cliente real;
- `order_quantity_tons` es una salida calculada de politica, no un registro observado de compra.
