# CU28 customer input examples

Estos CSV son ejemplos sintéticos de entrada para probar el contrato de datos de CU28.

No representan una fábrica real ni históricos reales de compra. Están pensados para validar comportamientos del pipeline batch/offline por escenarios.

## Archivos

| Archivo | Escenario | Uso esperado |
| --- | --- | --- |
| `00_uploaded_example_normalized.csv` | Copia normalizada del CSV subido | Referencia mínima basada en el contrato actual |
| `01_balanced_reference_input.csv` | Operación relativamente equilibrada | Validación estándar del flujo |
| `02_shortage_pressure_input.csv` | Inventario bajo y presión de requirement alta | Debe activar más compras |
| `03_no_purchase_expected_input.csv` | Inventario alto y menor presión operativa | Debe reducir o bloquear compras |
| `04_fresh_high_waste_input.csv` | Vida útil corta y merma elevada | Prueba de riesgo en producto fresco |
| `05_long_lead_time_supplier_risk_input.csv` | Lead time alto y mayor cobertura de seguridad | Prueba de riesgo logístico/proveedor |
| `06_multi_material_mixed_profiles_input.csv` | Varios materiales y perfiles productivos | Prueba multi-material y multi-contexto |

## Columnas

Todos los archivos respetan el contrato de entrada:

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

## Ejecución recomendada

```powershell
python -m src.main platform_run --input data/demo/01_balanced_reference_input.csv --output outputs/example_balanced/
python -m src.main platform_run --input data/demo/02_shortage_pressure_input.csv --output outputs/example_shortage/
python -m src.main platform_run --input data/demo/03_no_purchase_expected_input.csv --output outputs/example_no_purchase/
```

Para incluirlos en el repositorio, copiarlos a `data/demo/` o a una carpeta `data/examples/customer_inputs/`.
