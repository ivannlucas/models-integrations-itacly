# Datos de prueba — ml7-cereals-grain-pest-detection

YOLO que detecta insectos en grano almacenado (`cf`, `sz`, `rd`, `tc`, `os`) y devuelve la especie
dominante. Regenerar (necesita las credenciales S3 del `.env` y descarga 10 MB de Google Drive):
`.venv/bin/python -I outputs/a07/datos_prueba/generar_datos_prueba.py "$PWD"`

**Origen.** `inbox/a07/codigo/` llega vacío. Fuentes:
- Imágenes: copia del Pestdataset en `s3://…/datasets/modelo-7/raw/images/<ID>.jpg`.
- Anotaciones VOC: `annos.zip` de la carpeta pública de Google Drive que enlaza la memoria.

La conversión VOC → YOLO usa el orden de clases de `config.yaml` (`cf, sz, rd, tc, os`). Es el que aprendió
el detector y no el alfabético (ver el bug corregido en `constants.py`). Como el ETL original
(`relabel_invalid_classes` + `remove_images_with_class`), descarta la imagen entera si algún objeto no es de
esas clases (`unknown`, `ct`: 72 de 18.069 imágenes). El generador comprueba que las especies de los 23
golden coinciden con su `ground_truth_species`.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_caso_NNN.json` | `POST .../predict` (inline) | Los 23 golden cases del manifest (split de test), imagen en base64 |
| `predict_batch_golden.zip` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Las mismas 23 imágenes |
| `expected_golden.csv` | — | Especie dominante, confianza, nº de detecciones y conteo por especie esperados |
| `train_yolo.zip` | `POST .../train` con `{"data_path": "<ruta o s3://…>", "mlflow_run_id": "<run>"}` | Dataset YOLO en `train_yolo/`: `images/` + `labels/` {train 40, val 10, test 10} + `dataset.yaml` |

`train_yolo.zip` contiene 8/2/2 imágenes por especie dominante, las primeras por ID que no son golden.
`/train` acepta el ZIP en local o en `s3://`, como el resto de plugins de imagen. También acepta una
carpeta o un `dataset.yaml` locales. Del `dataset.yaml` de un ZIP solo usa `names`/`nc` (5 clases): las
rutas de los splits se fijan a `images/{train,val[,test]}` junto al yaml.

## Resultados comprobados en local

- 2026-11-10. Inline y batch: 23/23 idénticos a `expected_golden.csv` (diferencia 0).
- 2026-11-11. Train con `train_yolo.zip` (MLflow simulado, artefacto base sin tocar): 71 s en GPU; val
  mAP50 0,542, mAP50-95 0,359, precisión 0,517, recall 0,683. Las mismas métricas que con la carpeta del
  10-11 (early stopping en la época 45, mejor en la 25, de 300 máx.). Es una prueba de humo con 40
  imágenes; el modelo servido usa 12.597.
- Sin restos de Ultralytics: `yolo26n.pt` (comprobación de AMP) va a `<tmp>/ultralytics/weights`, la
  validación final escribe en el directorio temporal del entrenamiento (antes creaba
  `runs/detect/val-N`) y su callback de MLflow y su telemetría están desactivados
  (`app/domain/services/ultralytics_settings.py`). El `settings.json` del usuario no se toca.
