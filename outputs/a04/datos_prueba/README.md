# Datos de prueba — ml4-lactic-cnn-thermal-early-disease-detection

EfficientNet-B0 que clasifica una termografía de ubre de oveja como `Healthy` o `SCM` (mastitis
subclínica). Regenerar (descarga ~3 MB de Zenodo por rangos):
`.venv/bin/python -I outputs/a04/datos_prueba/generar_datos_prueba.py "$PWD"`

**Origen.** `inbox/a04/codigo/` llega vacío. Las imágenes salen del dataset público TIDS en Zenodo,
**versión 2** (record 18854578), cuyo ZIP es idéntico byte a byte al de la versión 1 que usó el equipo de
IA. No se usa la versión 3 (2026-10-31): sustituye la termografía del animal 11, que es un golden case.
Las de `s3://…/datasets/modelo-4/` no sirven: son las imágenes completas (`TIDS_full_images`), y el
modelo y los golden usan los recortes (`TIDS_cropped`).

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_tids_test_<ID>_<clase>.json` | `POST .../predict` (inline) | Los 18 golden cases del manifest (split de test oficial), recorte en base64 |
| `predict_batch_golden.zip` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Las mismas 18 imágenes (`<clase>/<ID>.jpg`) |
| `expected_golden.csv` | — | Salida esperada del checkpoint servido (`real_served_output`) y etiqueta clínica real |
| `train_tids_50.zip` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | `ID_Labels.csv` + `TIDS_cropped/{healthy,SCM}/<ID>.jpg`: los 25 primeros IDs de cada clase que no son golden |

El modelo acierta 11 de los 18 frente a la etiqueta real (la memoria da accuracy 0,595 en test). Los
fallos son del modelo, no de los datos: `expected_golden.csv` recoge la salida real servida.

## Resultados comprobados en local (2026-11-10)

- Inline y batch: 18/18 con la misma clase; confianza a ≤5e-5 (los esperados están redondeados a 4
  decimales).
- Train (MLflow simulado, artefacto base sin tocar): ~6 s en GPU; 40/10; métricas val 1,0. Con 10
  imágenes de validación no dicen nada. Además, los IDs bajos de cada clase son de sesiones de
  captura distintas (SCM de 2023-12-06, sanas de 2024-01-17), lo que facilita separarlas. Es solo una
  prueba de humo.

**Aviso sobre el dataset.** La versión 3 de TIDS corrige errores de las versiones 1/2, con las que se
entrenó el modelo servido:
- 245 era un duplicado de 128;
- la termografía de 11 era de otra fecha;
- el recorte de 612 venía de la imagen de 15;
- 153 era en realidad 135.

Ninguna corrección cambia la etiqueta sano/SCM, pero el reentrenamiento con TIDS debería hacerse con
la versión 3.
