# Model Card CU28

## Nombre

`CU28 - NEUROCARN-OPT`

## Tipo de solucion

Plataforma batch/offline de soporte a la decision para aprovisionamiento de materia prima carnica.

## Objetivo operativo

Ayudar a un cliente a evaluar si conviene comprar materia prima y cuanto comprar bajo una politica reproducible, comparando la recomendacion contra una politica base.

## Entradas minimas

- `date`
- `raw_material_id`
- `current_inventory_tons`
- `expected_requirement_tons`
- `lead_time_days`
- `safety_coverage_days`
- `expected_yield_rate`
- `expected_waste_rate`
- `unit_purchase_cost`
- `shelf_life_days`
- `destination_profile`

## Salidas principales

- `purchase_trigger_proba`
- `purchase_trigger_flag`
- `recommended_action`
- `quantity_optimizer_recommendation_tons`
- `order_quantity_tons`
- `decision_reason`
- `risk_level`
- `excess_tons`
- `stockout_tons`
- `aggregate_excess_reduction_pct`
- `aggregate_stockout_change_pct`
- `stockout_guardrail_pass`

## Arquitectura

- Predictor upstream: estima `synthetic_procurement_need`, una senal de presion
  de aprovisionamiento que no es la cantidad final.
- Etapa 1: `Purchase Trigger`
- Etapa 2: `Quantity Optimizer`
- Simulacion downstream: `Policy Simulation`

La decision separa dos preguntas: si conviene comprar y, solo si el trigger es
positivo, cuanto comprar. `order_quantity_tons` es la salida final y queda en
`0.0` cuando el trigger es negativo.

## Entrenamiento reproducible

- Ruta: `mixed_context`.
- Splits cronologicos: `train` para ajuste, `validation` para seleccion y
  calibracion, `test` solo para evaluacion final.
- Targets supervisados: `synthetic_procurement_need`,
  `purchase_trigger_label` y `quantity_optimizer_target_tons`.
- Semillas, hiperparametros y umbrales: definidos en `config/config.yaml`.
- Resultados vigentes: `reports/official/cu28_metrics_official__mixed_context.md`,
  `models/metrics/summary/` y `models/metrics/official/`.

## Alcance

- soporte a la decision;
- ejecucion batch/offline;
- cliente configurable por contrato de datos;
- validacion funcional parcial en entorno controlado.

## No alcance

- tiempo real en planta;
- compra autonoma;
- validacion industrial final;
- predictor directo unico de cantidades optimas reales.

## Estado de validacion

La version actual es una implementacion minima y reproducible con reglas fallback documentadas. Sirve para demo, trazabilidad y preparacion para carga futura de datos reales de cliente.

`platform_run` ejecuta recomendaciones sobre un CSV operativo, pero no reentrena. El entrenamiento reproducible pertenece a `mixed_context`, que combina proxies externos, capa sintetica documentada y variables derivadas. No debe presentarse como validacion industrial final ni como historico real completo de planta.
