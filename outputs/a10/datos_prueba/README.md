# Datos de prueba — ml10-dairy-disease-vector-detection

Pipeline en dos etapas: un detector YOLO localiza los insectos y un clasificador decide la especie
(`fly`, `mos`, `tick`). Datos reales de la entrega (`inbox/a10/codigo/.../data/`), sin valores
inventados. Regenerar: `.venv/bin/python -I outputs/a10/datos_prueba/generar_datos_prueba.py "$PWD"`

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_<id>.json` | `POST .../predict` (inline) | Los 25 golden cases del manifest, imagen en base64 con `det_conf_thresh` 0,2 y `cls_conf_thresh` 0,5 |
| `predict_batch_golden.zip` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Las mismas 25 imágenes, renombradas `<id>.<ext>` (hay crops homónimos) |
| `expected_golden.csv` | — | Especie, confianza, nº de vectores, resumen y detecciones (caja incluida) esperados |
| `train_30_por_especie.zip` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | `{fly,mos,tick}/*.jpg`: 30 crops por especie del split de train (auto-split 70/15/15) |

Golden cases:
- 18 crops del split de test de clasificación;
- 6 imágenes completas sin recortar;
- `data/test.jpg`.

Tolerancia del manifest: 0,01 absoluto en las confianzas.

## Resultados comprobados en local (2026-11-10)

- Inline y batch: 25/25 dentro de tolerancia (especie, confianza, nº de vectores, resumen y cada
  detección, con la caja a ±2 px).
- Train (solo el clasificador; MLflow simulado, artefacto base sin tocar): ~21 s en GPU; 63/12/15;
  16 épocas; acc. test 0,867 sobre 15 crops. Prueba de humo: el clasificador servido usa ~1.900 crops.
