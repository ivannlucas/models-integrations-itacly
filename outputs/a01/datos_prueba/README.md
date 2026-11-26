# Datos de prueba — ml1-wine-pest-disease-detection

YOLOv8 recorta la hoja o el racimo y un Deep Ensemble de 3 CNN clasifica una de 15 plagas/patologías de
la vid. Una cabeza tabular estima un índice de riesgo 0-1 a partir de 9 sensores. Sin imagen, solo se
calcula el riesgo, con el RandomForest regresor (`IMAGEN_NO_DISPONIBLE`). Todo sale de la entrega del
equipo de IA (`inbox/a01/codigo/data/`) y del manifest. No hay valores inventados.
Regenerar: `.venv/bin/python -I outputs/a01/datos_prueba/generar_datos_prueba.py "$PWD"`

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_caso_NN_<clase>.json` | `POST .../predict` (inline) | Los 20 golden cases del manifest (split de test): imagen 224 en base64 (`IMAGENES_RESIZED224`, la misma con la que se fijaron los valores esperados) + 9 sensores → `HIBRIDO_ENSAMBLE` |
| `predict_inline_sin_imagen_input_fila_2.json` | ídem | Solo sensores: fila 2 (`N/A`) del `input.csv` de ejemplo del equipo → `RIESGO_TABULAR_EXPERT` |
| `predict_inline_rechazo_sensores_incompletos.json` | ídem | Caso de contrato del manifest (`{"Mes": 1}`) → **422** «Faltan sensores requeridos» |
| `predict_batch_golden.zip` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | `entrada.csv` (`Path_Imagen` + 9 sensores + `CO2_ppm`/`VOC_ppb`, que se ignoran) + `images/`. 23 filas: los 20 golden y las 3 del `input.csv` del equipo |
| `expected_golden.csv` | — | Clase, confianza, riesgo y método esperados de las 23 filas (en el orden de `entrada.csv`) y, en los golden, la etiqueta y el riesgo reales |
| `train_1000_por_clase.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | 15.000 filas de `data/splits/train_rf.csv`: las 1000 primeras de cada una de las 15 clases, en el orden del fichero y con sus columnas originales |

**Valores esperados.** En los golden son los `predicho_*` del manifest: salida del plugin servido, a 4
decimales. caso_13 y caso_17 son **errores del modelo** ya documentados (OIDIO → HEALTHY_HOJA / OIDIO_RACIMO),
no de la integración; por eso `acierto_clase_modelo` es `False`. En las 3 filas del equipo son los de su
`data/predictions/predictions.csv`. La fila 1 (`image (698).JPG`) no trae la foto en la entrega, así que
el equipo y el plugin la resuelven sin imagen. En el ZIP tampoco va, a propósito. La fila 2 es `N/A`.

Tolerancias del manifest: clase exacta; riesgo ±2×MAE (0,378) sobre el riesgo **real**. Aquí se compara
además con la salida esperada del modelo a 4 decimales, que es más estricto.

**Train.** Solo reentrena los dos RandomForest tabulares (el de clase, solo para devolver accuracy/F1; el
de riesgo, que es el que se sube). YOLO y el ensemble CNN no se reentrenan
(`manifest.training.scope_note`). El modelo nuevo vive **solo en el run de MLflow**. Para usarlo, se pasa
`mlflow_run_id` en `/predict`: cambia el riesgo de las peticiones **sin imagen**, y con imagen la salida
es la misma que la del modelo base. El plugin valida las columnas, que no haya etiquetas desconocidas, que
estén las 15 clases (`constraints_entrenamiento` del manifest; si falta alguna responde 400 nombrándola)
y un mínimo de 10 filas. El CSV las trae todas.

## Resultados comprobados en local (2026-11-26)

App real (`main.app` + `TestClient`, `MODEL=ml1-…`), artefactos de `artifacts/ml1_wine_pest_disease_detection/`,
CPU (`CUDA_VISIBLE_DEVICES=`), MLflow simulado con un sqlite local en una carpeta temporal.

- **Inline**: 21/21 con la clase y el método esperados (incluidos los dos errores del modelo, caso_13 y
  caso_17, que salen igual que en el manifest). Riesgo idéntico a 4 decimales en los 21 (diferencia
  máx. 5,0e-5). La confianza es idéntica a 4 decimales en 19/21. Los dos restantes son caso_12 y caso_16,
  los dos MILDIU (0,673433 frente a 0,6735 y 0,670240 frente a 0,6703, diferencia máx. 6,7e-5). Tiempo
  medio: 0,7-1 s por petición. Rechazo de contrato: 422 con «Faltan sensores requeridos: […8 sensores]».
- **Batch** (23 filas, 20-35 s): `n_predictions` 23, sin errores de fila. Clase y método esperados en
  23/23 y riesgo a 4 decimales en 23/23. Confianza idéntica a 4 decimales en 20/23: además de caso_12 y
  caso_16, la fila 0 del equipo (Downy Mildew_116) da 0,648777 frente al 64,87 % de su `predictions.csv`
  (diferencia máx. 7,7e-5). Las dos filas sin imagen dan 0,2985 y 0,1770, como el equipo.
- **Origen de esas tres diferencias** (las tres son MILDIU, con la clase y el riesgo exactos). Es ruido
  numérico float32 de la CPU. En el mismo proceso, la misma petición da siempre el mismo valor. Al cambiar
  `torch.set_num_threads` (1-12), la confianza se mueve ~2e-5 (caso_12: 0,673430-0,673444). Ninguna
  configuración probada reproduce exactamente la 4.ª cifra del manifest ni la del equipo, que se midieron
  en otro entorno. La regresión de riesgo (sensores → MLP) no varía. Tolerancias sin tocar: esto se
  documenta, no se absorbe.
- **Train** (MLflow sqlite simulado): ~1,6 s por llamada, 200. Split 80/20 → n_train 12.000 / n_test 3.000.
  accuracy 0,1147; F1 macro 0,1210; F1 weighted 0,1213; MAE riesgo 0,1933. Dos ejecuciones dan
  resultados idénticos (`random_state=42`). El run recibe las 6 métricas, el tag `model_id` y un solo
  artefacto, `model/rf_riesgo_expert.joblib` (17 MB). La accuracy, cercana al azar (1/15), es la del RF
  original (0,116 con las 400.000 filas): con solo sensores no se distingue la plaga.
- **Predict con el run reentrenado**: sin imagen (fila 2), riesgo 0,2154 frente a 0,1770 del modelo base.
  Con imagen (caso_01), salida idéntica a la del modelo base. Batch: las dos filas sin imagen pasan a
  0,3542 y 0,2154.
- **Artefacto base intacto**: md5 y mtime de los 6 ficheros de `artifacts/ml1_wine_pest_disease_detection/`
  iguales antes y después. RSS máximo del proceso de verificación: 2,6 GB.
