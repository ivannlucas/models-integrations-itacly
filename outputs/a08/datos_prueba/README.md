# Datos de prueba — ml8_cereals_img_anomaly_detector

MobileNetV3 de dos cabezas: categoría (`hongos`, `insectos`, `otros`, `sano`) y cereal (`arroz`, `maiz`,
`sorgo`, `trigo`). Regenerar (necesita las credenciales S3 del `.env`):
`.venv/bin/python -I outputs/a08/datos_prueba/generar_datos_prueba.py "$PWD"`

> **No hay golden cases ejecutables.** `inbox/a08/codigo/` llega vacío y las 19 imágenes de los
> `golden_cases` (split de test de 7 datasets de Kaggle) no están en ningún sitio accesible. La única
> copia del dataset real es `s3://…/datasets/modelo-8/{cereal}/train/{categoria}/` (50 por combinación),
> que es del split de **train**. Las imágenes de predicción de aquí tienen etiqueta real, pero no hay una
> salida esperada auditada, y el modelo ya las vio al entrenar. Sirven para probar los endpoints, no para
> medir el modelo.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_<cereal>_<categoria>.json` | `POST .../predict` (inline) | 1 imagen por cada una de las 12 combinaciones que existen, en base64 |
| `predict_batch_12.zip` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Las mismas 12 imágenes |
| `etiquetas_reales.csv` | — | Cereal y categoría reales (carpeta de origen) de cada imagen |
| `train_8_por_combinacion.zip` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | `{cereal}/{train,validation}/{categoria}/`: 6 + 2 imágenes por combinación |

Se toman por orden de clave, saltando las de más de 400 KB (unas pocas llegan a 10 MB). Maíz no tiene
`otros` y sorgo solo tiene `hongos`: así es el dataset real.

## Resultados comprobados en local (2026-11-10)

- Inline: 12/12 aciertan categoría y cereal, como cabe esperar al ser imágenes de train. Batch da lo
  mismo que inline.
- Train (MLflow simulado, artefacto base sin tocar): ~29 s en GPU; 64 de train (el procedimiento
  original recorta `hongos`) y 24 de validación; 10 + 5 épocas; acc. val categoría 54,2 % y cereal
  66,7 %. La primera vez descarga de internet los pesos ImageNet de MobileNetV3 (21 MB).
