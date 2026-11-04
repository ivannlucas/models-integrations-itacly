# Datos de prueba — ml28-meat-neuroevolutionary-raw-materials-prediction

Extraídos de la entrega del equipo de IA (`data/demo/` y `golden_cases` de `inbox/a28/manifest.yaml`),
sin valores inventados. Regenerar: `.venv/bin/python outputs/a28/datos_prueba/generar_datos_prueba.py "$PWD"`.

**No hay datos de entrenamiento:** ml28 es un motor de reglas determinista sin pesos
(`training.supported: false`); `POST /train` responde con `TrainingNotSupportedError`.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_case_NNN_*.json` | `POST /models/ml28-meat-neuroevolutionary-raw-materials-prediction/predict` | Body inline (una fila) de cada uno de los 5 golden cases |
| `expected_predictions.csv` | — | Entrada + salida esperada (`esperado_*`) de cada golden case |
| `predict_batch_*.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Los 8 CSV demo del equipo de IA (escenarios; ver tabla abajo) |

| CSV batch | Escenario | Filas | Compras (BUY) | Reducción exceso % |
|---|---|---|---|---|
| `customer_upload_example` / `00_uploaded_example_normalized` | CSV demo oficial (golden) | 20 | 12 | 20.959 |
| `01_balanced_reference_input` | Operación equilibrada | 20 | 1 | 40.609 |
| `02_shortage_pressure_input` | Inventario bajo, demanda alta | 20 | 20 | 19.326 |
| `03_no_purchase_expected_input` | Inventario alto | 20 | 0 | 0.0 |
| `04_fresh_high_waste_input` | Fresco, merma alta | 20 | 8 | 56.224 |
| `05_long_lead_time_supplier_risk_input` | Lead time largo | 20 | 20 | 21.173 |
| `06_multi_material_mixed_profiles_input` | Multi-material y multi-perfil | 32 | 15 | 23.508 |

Comprobado en local (2026-11-04): los 5 casos inline coinciden exactamente con `expected_predictions.csv`
(incluido `case_005`, que verifica el gating: recomendación bruta 1.638 t pero `order_quantity_tons = 0`);
el CSV demo reproduce el resumen del manifest (12 compras, 20.959 %). Guardrail de rotura de stock en
verde en los 8 CSV.
