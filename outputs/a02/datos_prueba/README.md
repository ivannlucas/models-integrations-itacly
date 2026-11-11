# Datos de prueba — ml2-fungal-cnn-disease-detection

CNN que clasifica una hoja de vid en 5 clases (`black_rot`, `downy_mildew`, `healthy`, `powdery_mildew`,
`trunk_disease`). Imágenes reales de la entrega (`inbox/a02/codigo/data/`), sin valores inventados.
Regenerar: `.venv/bin/python -I outputs/a02/datos_prueba/generar_datos_prueba.py "$PWD"`

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_caso_NNN_<clase>.json` | `POST .../predict` (inline) | Los 18 golden cases del manifest (split de test), imagen en base64 |
| `predict_batch_golden.zip` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Las mismas 18 imágenes |
| `expected_golden.csv` | — | Clase, confianza y probabilidades esperadas (checkpoint servido) y etiqueta real |
| `train_30_por_clase.zip` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | `<clase>/*.jpg`: 30 imágenes por clase del split de train (`data/splits/train.csv`) |

## Resultados comprobados en local (2026-11-10)

- Inline y batch: 18/18 idénticos a `expected_golden.csv` (diferencia 0 en las probabilidades). Los 18
  aciertan además la etiqueta real.
- Train (MLflow simulado, artefacto base sin tocar): ~25 s en GPU; split 80/20 → 120/30; 44 épocas
  (early stopping); accuracy val 0,767, F1 0,754. Es una prueba de humo con 150 imágenes; el modelo
  servido se entrenó con 3.801.
