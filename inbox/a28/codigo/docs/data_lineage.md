# Data Lineage

## Official mixed_context trace

`INE_CPI` raw  
-> `data/processed/external/context/external_long.csv`  
-> `data/processed/external/context/context_weekly_for_simulation.csv`  
-> `data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv`  
-> `data/processed/baseline/feature_engineering_modeling__mixed_context.csv`  
-> trigger / optimizer / policy metrics

`MAPA_SLAUGHTER_MAPA` raw  
-> `data/processed/external/context/external_long.csv`  
-> `data/processed/external/context/context_weekly_for_simulation.csv`  
-> `data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv`  
-> `data/processed/baseline/feature_engineering_modeling__mixed_context.csv`  
-> trigger / optimizer / policy metrics

`MAPA_PRICES_OM` raw  
-> `fallback_constant`  
-> `data/processed/external/context/context_proxy_limitations.json`  
-> no defender como feed semanal activo

## Interpretation rules

- external source = real/proxy contextual;
- external proxies are not observed plant histories, purchase ledgers,
  receipts, inventory movements or realized waste;
- internal plant variables = synthetic or customer-provided;
- final modelable dataset = mixed and derived;
- `order_quantity_tons` = calculated decision output, not observed purchase record.

## Scope boundaries

- el blob oficial mixed_context no afirma que el dataset modelable sea un historico real completo de fabrica carnica;
- `feature_engineering_modeling__mixed_context.csv` mezcla proxies externos y variables sinteticas/controladas;
- los proxies externos no deben reinterpretarse como historicos reales de
  planta ni como sustitutos de datos longitudinales del cliente;
- la arquitectura batch/offline sigue separando decidir si comprar (`purchase_trigger_flag`) de cuanto comprar (`order_quantity_tons`).
