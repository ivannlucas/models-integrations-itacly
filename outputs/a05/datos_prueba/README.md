# Datos de prueba — ml5-meat-cow-behaviour

Pipeline Faster R-CNN (detecta vacas) + ByteTrack (las sigue) + SlowFast (clasifica el comportamiento en
12 clases a partir de un clip de 32 frames). Datos reales de la entrega (CSIRO Cow Behavior Dataset en
`inbox/a05/codigo/data/` y `inbox/a05/golden_cases_real.json`), sin valores inventados. Regenerar:
`.venv/bin/python -I outputs/a05/datos_prueba/generar_datos_prueba.py "$PWD"`

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_caso_NNN_<comportamiento>_trackN.json` | `POST .../predict` (inline) | Los 25 golden cases: `frames_base64` con 32 ROI 224×224 de una vaca (2-3 por clase) |
| `expected_golden.csv` | — | Comportamiento real de cada clip (pureza 100 % en sus 32 frames) |
| `predict_batch_video.mp4` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | 64 frames reales (2,1 s, 1920×1080, 30 fps) del clip `1333_…_beh4_ani2_ins1_cut_00058` |
| `batch_video_anotaciones.csv` | — | Anotación real de esos 64 frames (track, comportamiento y caja) |
| `comprobar_batch_video.py` | — | Ejecuta el batch y lo compara frame a frame con la anotación |
| `train_cvat_9clips.zip` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | `raw_frames/<clip>/img_NNNNN.jpg` + `annotations/<clip>/annotations/instances_default.json` |

**Inline.** El ROI se recorta como `utils/data_utils.py` (caja COCO entera, recortada a la imagen,
redimensionada a 224×224) y se codifica en JPEG. Los dos casos de `running` tienen solo 30 y 26 frames
reales: se completan duplicando el último, como el original (`padded_to_32`). Se predice con
`threshold` 0,5.

**Batch.** El vídeo se ha montado con los frames reales del clip, en orden. Es un clip del split de
test, que no está ni en golden ni en el ZIP de train. Sus 8 vacas aparecen en los 64 frames con un
comportamiento constante: 5 comportamientos distintos y solo 1 `grazing`. Batch no tiene golden: se
compara con la anotación (`comprobar_batch_video.py`, que empareja cada fila con la vaca anotada de
mayor IoU ≥ 0,5). Hasta que un track acumula 32 frames, el plugin devuelve como relleno la primera clase
(`grazing`, confianza 1,0), igual que `main.py:65-67` del original. Se mantiene por fidelidad y solo se
puntúa a partir del frame 31. Con el clip anterior (6 de 7 vacas en `grazing`), el relleno coincidía con
la respuesta correcta y la prueba no distinguía entre relleno y predicción.

**Train.** Reentrena solo el clasificador SlowFast; el detector queda fuera. El ZIP lleva 9 clips de
train, el primero de cada código `behN`, sin los clips de los golden. Cada clip va con sus 32 primeros
frames, y el JSON de CVAT se ha filtrado a esos frames: el clip completo tiene 450 frames, unos 25 MB.
El split 70/15/15 es por clip.

## Resultados comprobados en local (2026-11-10)

- Inline: 24/25. Falla el caso 017 (`resting-standing`, clip `1349_…`), que sale `hidden` con 0,881.
  Es el mismo fallo del modelo documentado en `verification_report.md`.
- Batch (2026-11-11, ~30 s en GPU): 190/256 filas puntuadas (74,2 %) y 6/8 vacas bien en el último
  frame. 5 vacas aciertan en todos sus frames: 3 `resting-standing`, 1 `ruminating-lying` y 1 `grazing`
  (0,99, predicción real). Fallan:
  - track 5: anotada `resting-lying`, sale `ruminating-lying` en 28/33 (0,60 en el último frame);
  - track 6: anotada `hidden`, sale `resting-lying` en 33/33;
  - track 7: `ruminating-lying`, acierta 28/33.

  Hay 248 filas de relleno y 8 sin vaca anotada (IoU < 0,5). Diagnóstico: clasificando inline esas vacas
  con las **cajas anotadas**, la ventana de los frames 32-63 acierta las 8 (`resting-lying` 0,75,
  `hidden` 0,73). La de los frames 0-31 falla precisamente esas dos, con confianza ≤ 0,53. El clasificador
  duda entre comportamientos tumbados parecidos, y en batch recibe los recortes del detector en vez de
  las cajas anotadas. Es el comportamiento del modelo, no un fallo de la lógica del plugin.
- Train (MLflow simulado, artefactos base sin tocar): 28 clips de train / 7 de val, 30 épocas fijas,
  best_val_acc 28,6 %. **Tarda ~33 min** en la GPU de 8 GB de este equipo (RTX 2070 SUPER, con la
  memoria al límite): ~1 min por época, de los que solo ~12 s son lectura de frames. Prueba de humo: el
  modelo servido se entrenó con los 502 clips.
