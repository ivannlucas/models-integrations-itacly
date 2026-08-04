# Output Contract

## Artefactos escritos por corrida

| Archivo | Formato | Contenido |
| --- | --- | --- |
| `validation_report.json` | JSON | Resultado completo de validacion del input. |
| `recommendations.csv` | CSV | Recomendacion oficial fila a fila que se entrega al usuario de negocio. |
| `policy_simulation_results.csv` | CSV | Salida fila a fila de la simulacion comparativa frente a baseline. |
| `summary_metrics.json` | JSON | Metricas agregadas para defensa ejecutiva y tecnica. |

## Columnas de `recommendations.csv`

| Campo | Tipo | Obligatoria | Puede ser null | Reglas de calculo |
| --- | --- | --- | --- | --- |
| `date` | date/string | Si | No | Copia normalizada de la fecha de input. |
| `raw_material_id` | string | Si | No | Copia del identificador de materia prima. |
| `destination_profile` | string | Si | No | Copia del destino productivo previsto. No es producto comprado. |
| `current_inventory_tons` | float | Si | No | Copia del input validado. |
| `expected_requirement_tons` | float | Si | No | Copia del input validado. |
| `lead_time_days` | float | Si | No | Copia del input validado. |
| `safety_coverage_days` | float | Si | No | Copia del input validado. |
| `expected_yield_rate` | float | Si | No | Copia del input validado. |
| `expected_waste_rate` | float | Si | No | Copia del input validado. |
| `unit_purchase_cost` | float | Si | No | Copia del input validado. |
| `shelf_life_days` | float | Si | No | Copia del input validado. |
| `purchase_trigger_proba` | float | Si | No | Score/probabilidad de necesidad de compra calculada por el trigger. |
| `purchase_trigger_flag` | int | Si | No | `1` si se activa compra; `0` si se bloquea. |
| `recommended_action` | string | Si | No | `BUY` si `purchase_trigger_flag = 1`; `DO_NOT_BUY` si `purchase_trigger_flag = 0`. |
| `quantity_optimizer_recommendation_tons` | float | Si | No | Recomendacion bruta del optimizador antes del gating final. |
| `order_quantity_tons` | float | Si | No | Cantidad final recomendada. Si `purchase_trigger_flag = 0`, debe valer `0.0`. |
| `decision_reason` | string | Si | No | Texto legible para negocio segun trigger, gap de cobertura y politica aplicada. |
| `projected_stock_after_lead_time_tons` | float | Si | No | `current_inventory_tons - expected_requirement_tons * lead_time_days / 7`. |
| `safety_stock_tons` | float | Si | No | `expected_requirement_tons * safety_coverage_days / 7`. |
| `coverage_gap_tons` | float | Si | No | `max(safety_stock_tons - projected_stock_after_lead_time_tons, 0)`. |
| `risk_level` | string | Si | No | `LOW`, `MEDIUM` o `HIGH` segun regla documentada abajo. |
| `baseline_order_quantity_tons` | float | Si, si baseline disponible | Si | Cantidad de referencia calculada por la politica baseline. |
| `delta_order_vs_baseline_tons` | float | Si, si baseline disponible | Si | `order_quantity_tons - baseline_order_quantity_tons`. |
| `excess_tons` | float | Si, si simulacion disponible | Si | Excedente estimado bajo la politica recomendada. |
| `stockout_tons` | float | Si, si simulacion disponible | Si | Stockout estimado bajo la politica recomendada. |

## Regla de riesgo

`risk_level` se calcula de forma simple y trazable:

- `HIGH`: `stockout_tons > 0` o `coverage_gap_tons > 0` con `purchase_trigger_proba >= 0.75`;
- `MEDIUM`: se activa compra, existe gap de cobertura o `purchase_trigger_proba >= 0.50`;
- `LOW`: resto de casos.

Esta regla es diagnostica para demo y reunion. No sustituye validacion industrial ni una matriz de riesgo aprobada por el cliente.

## Metricas agregadas de `summary_metrics.json`

| Campo | Tipo | Descripcion |
| --- | --- | --- |
| `row_count` | int | Numero de filas procesadas. |
| `triggered_orders` | int | Numero de filas con `purchase_trigger_flag = 1`. |
| `total_order_quantity_tons` | float | Suma de `order_quantity_tons`. |
| `aggregate_excess_reduction_pct` | float | Reduccion porcentual de excedente frente a baseline. |
| `aggregate_stockout_change_pct` | float | Cambio porcentual de stockout frente a baseline. |
| `stockout_guardrail_pass` | bool | Cumplimiento del guardrail de stockout. |

## Reglas criticas

- `order_quantity_tons` es la salida final.
- `quantity_optimizer_recommendation_tons` es previa al gating.
- `purchase_trigger_flag = 0` fuerza `order_quantity_tons = 0.0`.
- `quantity_optimizer_recommendation_tons` no debe presentarse como decision final cuando el trigger es negativo.
- `excess_tons` y `stockout_tons` son metricas estimadas por simulacion, no inputs upstream.
- los valores agregados de una corrida local no son metricas oficiales del
  modelo; las metricas oficiales se publican en
  `../reports/official/cu28_metrics_official__mixed_context.md`,
  `../models/metrics/summary/` y `../models/metrics/official/`.
