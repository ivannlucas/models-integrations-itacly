# Feature Engineering

## Route

La fase oficial parte de `synthetic_plant_layer__mixed_context.csv` y genera `feature_engineering_modeling__mixed_context.csv`.

## Included transformations

- lags temporales: `1`, `2`, `4`, `8`
- rolling means: `4`, `12`
- variables calendario: `date_year`, `date_month`, `date_quarter`, `date_week_of_year`
- interacciones: `demand_supply_gap`, `demand_supply_ratio`
- encoding de contexto productivo: `manufacturing_context_profile__*`, `product_family__*`, `recipe_profile__*`, `shelf_life_class__*`
- aliases oficiales:
  - `current_inventory_tons`
  - `expected_requirement_tons`
  - `lead_time_days`
  - `safety_coverage_days`
  - `expected_yield_rate`
  - `expected_waste_rate`
  - `raw_material_id`
  - `destination_profile`
- targets derivados:
  - `purchase_trigger_label`
  - `quantity_optimizer_target_tons`
  - `baseline_order_quantity_tons`

Convenciones:

- las cantidades se expresan en toneladas;
- yield y waste se expresan como ratios entre `0` y `1`;
- `product_family` y `product_family_alias`, cuando aparecen, son aliases
  heredados de `destination_profile`, no materias primas ni productos
  terminados comprados.

## Model input boundaries

Upstream predictor:

- usa contexto externo, variables operativas sinteticas y features temporales;
- no usa `order_quantity_tons`;
- no usa `purchase_trigger_flag` o `purchase_trigger_proba`;
- no usa `quantity_optimizer_target_tons`.

Purchase trigger:

- usa inventario, requirement, lead time, cobertura, yield/waste y contexto;
- no usa salidas del optimizador.

Quantity optimizer:

- usa outputs predichos del trigger y variables operativas/contextuales;
- no usa `purchase_trigger_label` durante inferencia;
- la salida final queda gateada fuera del modelo.

Policy simulation:

- usa `order_quantity_tons`, `baseline_order_quantity_tons`, exceso y stockout calculados;
- no realimenta esas variables a etapas upstream.
