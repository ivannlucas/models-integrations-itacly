# Simulation Data Basis

La base actual de simulacion se apoya en una mezcla trazable de contexto externo real/proxy, variables internas sinteticas o cargadas por cliente y reglas fallback documentadas. No se presenta como validacion industrial final ni como sustituto de datos reales completos de cliente.

## External context actually defended

- `INE_CPI` entra como proxy contextual activo de presion inflacionaria.
- `MAPA_SLAUGHTER_MAPA` entra como proxy contextual activo de oferta/sacrificio.
- `MAPA_PRICES_OM` queda solo como `traced/fallback_constant`, no como feed semanal activo.

## Internal plant variables

- `current_inventory_tons`, `expected_requirement_tons`, `lead_time_days`, `safety_coverage_days`, `expected_yield_rate` y `expected_waste_rate` son sinteticos en la capa reproducible o cargados por cliente para una inferencia CSV.
- No se dispone en este repositorio de historicos completos observados de ordenes de compra, recepciones, mermas, stockout, vida util operativa cerrada, BOM real ni rendimiento real por orden.

## Final modelable dataset

- `data/processed/external/context/external_long.csv` concentra el contexto externo trazado.
- `data/processed/external/context/context_weekly_for_simulation.csv` agrega el contexto semanal de simulacion.
- `data/processed/baseline/feature_engineering_modeling__mixed_context.csv` es un dataset derivado y mixto, no un historico puro de fabrica.
- `order_quantity_tons` es una salida calculada de decision, no un registro observado de compra.

## Leakage exclusions and non-claims

- operacion online en planta;
- exactitud industrial validada sobre historicos completos;
- ordenes observadas de compra usadas como target directo oficial;
- una politica autonoma de compras lista para produccion.

La capa sintetica permite reconstruccion tecnica, pruebas y comparacion
controlada. No constituye validacion industrial final; esa validacion exige
historicos observados del cliente y criterios de aceptacion acordados para su
operacion.
