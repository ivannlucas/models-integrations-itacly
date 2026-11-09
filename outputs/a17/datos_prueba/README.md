# Datos de prueba — ml17-meat-market-price-analysis

Extraídos de la entrega del equipo de IA (`data/splits/official_v1_4/`, `data/examples/`,
`data/predictions/` y `golden_cases` del manifest), sin valores inventados. Regenerar:
`.venv/bin/python outputs/a17/datos_prueba/generar_datos_prueba.py "$PWD"`.

Ridge que predice el precio del porcino clase E España (€/100 kg) en el mes t+1 a partir de las
variables del mes t.

> **Ojo con `/train`:** cada llamada publica una **nueva versión del modelo base** (`BASE_ARTIFACT_FOLDER`)
> y el servicio pasa a servirla. No lanzar `train.csv` contra un entorno compartido salvo que se quiera
> eso; para volver atrás, `POST /models/ml17-meat-market-price-analysis/rollback`.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_caso_001…018.json` | `POST /models/ml17-meat-market-price-analysis/predict` | 18 golden cases (mes t del tramo de test) |
| `predict_inline_ejemplo_2025_12.json` | ídem | Ejemplo de inferencia del equipo de IA (base 2025-12-01) |
| `expected_ejemplo_2025_12.json` | — | Su predicción original para 2026-01-01: 142.3145 |
| `predict_batch_test.csv` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Los 31 meses origen del benchmark oficial (t = 2023-05 … 2025-11) |
| `expected_test.csv` | — | Precio real observado en t+1 para cada fila del batch |
| `train.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": ""}` | Split de train oficial: 121 meses (2013-05 … 2023-05) |

Los `expected` de los golden cases del manifest son el **precio real** en t+1, no la salida del modelo:
se comparan con tolerancia 2×MAE = 8.921.

## Resultados comprobados en local (2026-11-09)

- Golden inline: 16/18 dentro de tolerancia. `caso_006` (9.55) y `caso_018` (26.6) quedan fuera por error
  del propio modelo, ya documentado en `outputs/a17/verification_report.md`.
- Ejemplo 2025-12: 142.31448869331666, idéntico al original.
- Batch test (31 meses): MAE 4.4605 y RMSE 6.4091, idénticos al benchmark oficial (memoria Tabla 8).
- Train (publicación del modelo base simulada, artefactos sin tocar): MAE 5.2979, RMSE 7.1946, MASE 0.7607,
  R² 0.9404 y acierto direccional 0.7833, idénticos a la memoria Tabla 7.
