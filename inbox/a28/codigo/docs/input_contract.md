# Input Contract

El cliente debe cargar un CSV tabular con una fila por combinacion `date` + `raw_material_id` + `destination_profile`. El CSV operativo permite ejecutar recomendaciones. El entrenamiento industrial requiere historico longitudinal.

## Columnas minimas

| Campo | Tipo | Unidad | Obligatorio | Descripción | Ejemplo |
| --- | --- | --- | --- | --- | --- |
| `date` | fecha ISO o parseable | dia | Si | Fecha de la necesidad operativa o semana de planificacion. | `2025-01-05` |
| `raw_material_id` | string | n/a | Si | Identificador de materia prima carnica. | `RM_BEEF_TRIM_A` |
| `destination_profile` | string | n/a | Si | Destino productivo previsto de la materia prima. No es el producto comprado. | `cooked_standard_line` |
| `current_inventory_tons` | float | toneladas | Si | Inventario disponible al inicio del horizonte. | `58.0` |
| `expected_requirement_tons` | float | toneladas | Si | Requerimiento previsto de materia prima. | `22.0` |
| `lead_time_days` | float | dias | Si | Dias esperados hasta recepcion. | `5.0` |
| `safety_coverage_days` | float | dias | Si | Cobertura objetivo de seguridad. | `10.0` |
| `expected_yield_rate` | float | ratio 0-1 | Si | Rendimiento esperado de la materia prima. | `0.89` |
| `expected_waste_rate` | float | ratio 0-1 | Si | Merma esperada. | `0.02` |
| `unit_purchase_cost` | float | coste por tonelada | Si | Coste unitario esperado. | `3.80` |
| `shelf_life_days` | float | dias | Si | Vida util esperada de la materia prima/lote. | `28` |

## Reglas de validacion

- no se permiten `NaN` en columnas obligatorias;
- `date` debe poder convertirse a fecha;
- cantidades, coberturas, lead time, coste y vida util no pueden ser negativas;
- `expected_yield_rate` y `expected_waste_rate` deben quedar entre `0` y `1`;
- `raw_material_id` y `destination_profile` no pueden venir vacios.

## Que datos necesita el cliente para ejecutar la recomendacion

Para usar `platform_run`, el cliente necesita preparar un CSV operativo con:

- inventario actual por materia prima;
- requerimiento previsto de materia prima;
- lead time esperado;
- cobertura de seguridad objetivo;
- yield y waste esperados;
- coste unitario;
- vida util;
- destino productivo previsto de esa materia prima.

Estos campos permiten calcular el trigger de compra, la cantidad bruta, el gating final, la recomendacion final y la simulacion frente a baseline. No se requiere historico completo para ejecutar una recomendacion puntual.

## Que datos harian falta para entrenamiento industrial real

Para entrenar, calibrar o validar industrialmente sobre operacion real harian falta historicos reales longitudinales:

- inventario historico;
- consumo real;
- ordenes de compra;
- recepciones;
- lead times reales;
- mermas reales;
- stockout real;
- costes reales;
- vida util por lote;
- decisiones pasadas;
- resultado posterior de esas decisiones.

El CSV operativo permite ejecutar recomendaciones. El entrenamiento industrial requiere histórico longitudinal.

## Observaciones de negocio

- `destination_profile` describe el destino productivo previsto, no el producto comprado.
- Si aparece `product_family`, debe documentarse como alias heredado de `destination_profile`.
- El sistema decide aprovisionamiento de materia prima carnica; no automatiza compras ni valida industrialmente la politica.
- `purchase_trigger_label`, targets de entrenamiento, predicciones y metricas
  downstream no forman parte del contrato de inferencia.

## Limites de disponibilidad

El repositorio no presupone que existan historicos internos completos de
compras, recepciones, mermas, stockout, BOM o rendimiento cerrado por orden.
Cuando esos datos existan, deben incorporarse mediante una nueva version del
contrato y una validacion industrial especifica; no se infieren de los proxies
externos ni del CSV demo.
