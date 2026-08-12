# Platform Usage

`platform_run` es exclusivamente una utilidad batch/offline de inferencia
sobre un CSV operativo del cliente o de demo. No es la ruta oficial de
entrenamiento, evaluacion ni reproduccion. La ruta oficial para esas funciones
es `mixed_context`.

## Ejecucion

```powershell
python -m src.main platform_run --input data/demo/customer_upload_example.csv --output outputs/demo_run/
```

Opciones de presentacion:

```powershell
python -m src.main platform_run --input data/demo/customer_upload_example.csv --output outputs/demo_run/ --max-cli-rows 20
python -m src.main platform_run --input data/demo/customer_upload_example.csv --output outputs/demo_run/ --show-all-recommendations
```

El CSV demo es solo un ejemplo del contrato de inferencia. No es el dataset
oficial de entrenamiento.

## Flujo

1. valida el CSV contra `input_contract.md`;
2. deriva stock proyectado, stock de seguridad y gap de cobertura;
3. ejecuta el trigger de compra;
4. ejecuta el optimizador de cantidad;
5. aplica el gating: `purchase_trigger_flag = 0` fuerza
   `order_quantity_tons = 0.0`;
6. simula la politica frente a baseline;
7. escribe los archivos descritos en `output_contract.md`.

## Salidas locales

- `outputs/demo_run/validation_report.json`
- `outputs/demo_run/recommendations.csv`
- `outputs/demo_run/policy_simulation_results.csv`
- `outputs/demo_run/summary_metrics.json`

La consola muestra estado, rutas, conteos y una tabla de recomendaciones. Los
valores pertenecen a la corrida local.

`outputs/` es regenerable y nunca es fuente oficial de metricas. Las metricas
oficiales viven en `../models/metrics/summary/`,
`../models/metrics/official/` y `../reports/official/`.

## Limites

`platform_run`:

- no reentrena modelos;
- no reconstruye el ETL `mixed_context`;
- no valida industrialmente el sistema;
- no sustituye historicos reales de cliente;
- no automatiza compras.

`destination_profile` representa el destino productivo previsto de la materia
prima. Si aparece `product_family`, es un alias heredado y no el producto
comprado.
