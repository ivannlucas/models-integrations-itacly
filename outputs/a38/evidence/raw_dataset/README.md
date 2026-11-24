# Evidencias con el dataset raw NLN-EMP (revisión del 2026-11-24)

Análisis del plugin ml38 contra los ficheros raw del dataset público NLN-EMP (4TU.ResearchData,
CC0). El dataset descomprimido ocupa 84 GB y no está en el repo. Para relanzar los scripts hace
falta una copia local: `<dataset_dir>` es la carpeta `Dataset/Dataset` que contiene
`Vibration/` y `Electric/`.

Todos los scripts se lanzan desde la raíz del repo, con `PYTHONPATH=.` y `KERAS_BACKEND=torch`
(el backend del servidor). En una máquina con GPU y torch con CUDA, añadir `CUDA_VISIBLE_DEVICES=`
para forzar CPU, como en los pods. Keras con backend torch usa la GPU si la ve, y con estas MLP
pequeñas eso es mucho más lento: `physics_term.py` tardó 78 s en CPU frente a más de 40 min con
la GPU ocupada por otro proceso. `<codigo_dir>` es `inbox/a38/codigo` (código del equipo de IA, no se
commitea) y `<out_dir>` es cualquier carpeta temporal.

| Script | Qué comprueba | Salida | Duración |
|---|---|---|---|
| `rebuild_from_raw.py <dataset_dir> <codigo_dir> <out_dir>` | Recalcula las 877 ventanas (vib ch1 + corriente ch1) con el código original y con el plugin; versiones recortadas; pico 1X medido | ficheros en `<out_dir>` | ~6 min, lee 16 GB |
| `analyze_rebuild.py <codigo_dir> <out_dir>` | ETL raw → parquet, port vs original, modo raw vs `predictions.csv`, efecto de la longitud (KI-05), 1X (KI-07) | `results/analyze_rebuild.txt` | <1 min |
| `channel_variants.py <dataset_dir> <codigo_dir> <out_dir>` | Mismas ventanas con los otros canales: vibración ch2-5, eléctrico ch2-6 (KI-06) | `channels.csv` en `<out_dir>` | ~16 min, lee ~68 GB |
| `analyze_channels.py <codigo_dir> <out_dir>` | Acierto de familia por canal | `results/analyze_channels.txt` | <1 min |
| `grouping_cv.py <codigo_dir>` | OOF de familia con los grupos del equipo frente a grupos por fallo físico (KI-17) | `results/grouping_cv.txt` | ~4 min |
| `physics_term.py <codigo_dir>` | Si la regla física se cumple en los datos y ablación λ = 0 / 0,3 (KI-18) | `results/physics_term.txt` | ~5 min |

`results/http_checks.txt` recoge las peticiones `/predict` contra el servidor local tras los
cambios del plugin (rechazos por canal y longitud, mensajes de error, inline con
`model_key: "raw"`).

Las cifras de "acierto" con las ventanas de canal 1 son in-sample: el modelo servido se entrenó
con esas mismas ventanas. Sirven para comparar variantes (longitud, canal), no como medida de
generalización. La única medida de generalización es la validación cruzada agrupada.
