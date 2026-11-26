# Datos de prueba — ml6-meat-poultry-disease-classification

Dos ensembles independientes de 3 CNN con voto por mayoría. El cuerpo elige cuál con `model_key` en
`/predict` (obligatorio; sin él, 422) y con `task` en `/train`:

- `thermal`: imagen térmica del ave → `healthy` / `sick` (DenseNet-121, EfficientNetV2-S, ResNet-50).
- `rgb`: imagen RGB de heces → `healthy` / `cocci` / `ncd` / `salmo` (VGG19-BN, EfficientNet-B7,
  MobileNetV3-L).

Regenerar (descarga ~16 MB de Mendeley y ~130 MB de Zenodo por rangos; ~1 min):
`.venv/bin/python -I outputs/a06/datos_prueba/generar_datos_prueba.py "$PWD"`

**Origen.** `inbox/a06/codigo/` está vacío. Las imágenes salen de las mismas fuentes públicas que los
`golden_cases` (ver `known_issues` del manifest):

- thermal: Mendeley `vwbb4dc9b7` v1 (`Healthy.rar` y `Sick.rar`). El script comprueba el sha256
  publicado. Para descomprimir hace falta 7-Zip o unrar; en WSL sirve el `7z.exe` de Windows.
  `tar.exe` de Windows no sirve: se salta los 8 nombres en árabe. Si existe
  `inbox/a06/codigo/data/thermal/raw/`, se usa esa copia.
- rgb: Zenodo `5801834` v3. Solo se bajan los ficheros necesarios, con peticiones HTTP por rangos.

El script recalcula el split con el algoritmo de `ensure_data_splits` (seed 42, 70/15/15) y aborta si
el test no coincide con los golden. En thermal coinciden los 40 ficheros. En rgb el test tiene
53/57/29/53 imágenes por clase y contiene los 23 golden. La copia de S3 (`datasets/`) no se ha podido
consultar: el certificado TLS del endpoint estaba caducado el 2026-11-26.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_thermal_test_NNN_<clase>.json` | `POST .../predict` (inline) | 10 golden thermal (los 5 primeros de cada clase), imagen en base64 y `"model_key": "thermal"` |
| `predict_inline_rgb_{test,tie}_NNN_<clase>.json` | `POST .../predict` (inline) | 5 golden rgb (el fichero más pequeño de cada clase y el empate a tres más pequeño), `"model_key": "rgb"` |
| `predict_batch_thermal.zip` | `POST .../predict` con `{"mode": "batch", "model_key": "thermal", "data_path": "<ruta>"}` | Los 40 golden thermal (= split de test completo) en `healthy/` y `sick/` |
| `predict_batch_rgb.zip` | `POST .../predict` con `{"mode": "batch", "model_key": "rgb", "data_path": "<ruta>"}` | Las 5 imágenes rgb de inline, en `pcr<clase>/` |
| `expected_golden_thermal.csv`, `expected_golden_rgb.csv` | — | Clase del ensemble y voto de cada modelo esperados (salida del pipeline original), etiqueta real, ruta en el ZIP y JSON inline |
| `train_thermal_10_por_clase.zip` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>", "task": "thermal"}` | `healthy/`, `sick/`: las 10 primeras imágenes (orden de nombre) del split de train de cada clase, ficheros originales |
| `train_rgb_10_por_clase.zip` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>", "task": "rgb"}` | `pcr{cocci,healthy,ncd,salmo}/`: las 10 primeras imágenes del split de train de cada clase, en PNG 256×256 (ver abajo) |

**Golden usados.** Las imágenes rgb pesan entre 0,7 y 7,4 MB: los 23 golden rgb suman 80 MB (~107 MB
en base64). Por eso no van todos. thermal va completo en batch, y así el batch reproduce además las
métricas de la memoria. La selección mira solo el nombre o el tamaño del fichero, nunca si el modelo
acierta.

- `rgb_tie_001` prueba el desempate a tres: cada modelo vota una clase y gana el voto de `vgg19_bn`
  (`healthy`; la etiqueta real es `cocci`).
- `thermal_test_003` (`A1_1_124_3_0_مصاب.bmp`, «infectado» en árabe, guardada en `Healthy`) es la
  posible etiqueta errónea que documenta el manifest.

**Salida.** `confidence` es la fracción de votos (1/3, 2/3 o 1), no una probabilidad. Se deduce de los
votos, así que el CSV no la repite. `model_probabilities` es informativo y no tiene valor esperado: el
original no lo calcula. Cuando las imágenes del batch van en carpetas de clase, la respuesta trae
además `evaluated_metrics`.

**Train.** El plugin exige al menos 7 imágenes por clase (el split 70/15/15 tiene que dejar una en
cada parte). Con 10 por clase el split queda en 7/1/2 por clase.

Los JPEG rgb originales pesan entre 0,5 y 7,6 MB, y 10 por clase serían ~120 MB. Lo primero que hace
el plugin con una imagen rgb es `Resize((256, 256))`. El generador aplica ese mismo `Resize` y guarda
el resultado en PNG, que no pierde información. Volver a aplicar el `Resize` a una imagen de 256×256
no la cambia. La red recibe, por tanto, el mismo tensor que con el JPEG original, y el generador lo
comprueba imagen a imagen. Lo único que este ZIP no ejercita es decodificar y reducir JPEG grandes, y
eso ya lo cubre el predict rgb.

## Resultados comprobados en local (2026-11-26)

Las pruebas usan el plugin real a través de FastAPI `TestClient` (`make_model_router` +
`ModelContainer`), en **CPU** (`CUDA_VISIBLE_DEVICES` vacío, 6 hilos de torch) y con los artefactos
locales.

- **Inline.** thermal 10/10 y rgb 5/5: la clase del ensemble y los votos de los 3 modelos coinciden
  con los esperados (el manifest fija tolerancia 0). Todas las térmicas miden 240×320, así que no sale
  ningún `warning`.
- **Batch.** thermal 40/40 (10 s) y rgb 5/5 (2 s), con la clase y los 3 votos idénticos. El softmax
  de cada modelo es idéntico al de inline (diferencia máxima 0).
  - `evaluated_metrics` thermal reproduce la memoria (Tabla 4). Ensemble: accuracy 0,8000,
    precision_w 0,7964, recall_w 0,8000 y f1_w 0,7960. Recall `sick` 0,8846 y `healthy` 0,6429.
  - Por modelo: densenet121 0,675, efficientnet_v2s 0,675 y resnet50 0,775, con f1_w 0,6774,
    0,6808 y 0,7604.
  - Las métricas del batch rgb (5 imágenes) no tienen valor estadístico.
- **Train.** MLflow va simulado: el tracker y la subida se sustituyen por una copia local. Los 6
  artefactos base tienen el mismo md5 y mtime antes y después de las pruebas, y su md5 es el del
  manifest.
  - thermal: **89 s** en CPU. Split 14/2/4, épocas 27/16/13 (densenet121 / efficientnet_v2s /
    resnet50). Accuracy de test del ensemble 0,50 con 4 imágenes. Se suben `densenet121.pt`,
    `efficientnet_v2s.pt`, `resnet50.pt` y `metadata.json`. `/predict` con ese run da 200; el mismo
    run con `model_key: "rgb"` da 422.
  - rgb: **196 s** en CPU. Split 28/4/8, épocas 15/11/13 (vgg19_bn / efficientnet_b7 /
    mobilenet_v3l), las mismas que la prueba de humo del `verification_report.md`. Accuracy de test
    del ensemble 1,0 con 8 imágenes. Se suben `vgg19_bn.pt`, `efficientnet_b7.pt`,
    `mobilenet_v3l.pt` y `metadata.json`. `/predict` con ese run da 200; con
    `model_key: "thermal"`, 422.
  - Antes de entrenar se midió una época por modelo en CPU: ~2,3 s en thermal y ~21 s en total
    para los 3 modelos rgb. Con el mínimo de 11 épocas del early stopping salían ~1,5 y ~4-5 min.

  En `/train`, `per_class[*].support` es entero, como en el batch (antes salía como float `2.0`;
  corregido en `TrainResponse` en la misma rama que estos datos).

  Son pruebas de humo: prueban el circuito, no el modelo. Con 2 o 4 imágenes de validación por
  ensemble, las métricas no dicen nada. El reentreno completo rgb (1 255 imágenes) dura ~2 h 45 min
  en GPU según la memoria.
