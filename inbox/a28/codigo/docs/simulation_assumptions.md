# Simulation Assumptions

## Datos reales de cliente

- inventario disponible;
- requirement previsto;
- lead time;
- cobertura objetivo;
- yield esperado;
- waste esperado;
- coste unitario;
- vida util.

## Senales proxy externas

- pueden incorporarse como contexto adicional;
- no son obligatorias para el flujo minimo;
- no sustituyen datos internos de cliente.

## Variables sinteticas

- capa sintetica generada por la ruta `mixed_context`;
- CSV demo usado unicamente como ejemplo de inferencia;
- probabilidades heuristicas del trigger fallback;
- baseline policy simple usada para comparacion;
- metricas agregadas derivadas de simulacion.

## Outputs calculados

- `purchase_trigger_proba`
- `purchase_trigger_flag`
- `recommended_action`
- `quantity_optimizer_recommendation_tons`
- `order_quantity_tons`
- `decision_reason`
- `risk_level`
- `excess_tons`
- `stockout_tons`
- metricas agregadas frente a baseline

## Variables excluidas por leakage o indisponibilidad

- ordenes reales ya emitidas para el periodo objetivo;
- recepciones reales cerradas posteriores al horizonte;
- stockout final conocido del mismo horizonte;
- BOM final no disponible;
- rendimiento real observado despues del periodo evaluado.

Los valores de resultado de la simulacion no se fijan en este contrato. Deben
consultarse en `../reports/official/cu28_metrics_official__mixed_context.md` y
en los artefactos oficiales de `../models/metrics/`.
