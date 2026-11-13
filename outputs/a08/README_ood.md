# ml8: detección de imágenes «desconocido»

## Problema

QA observó que, con sus imágenes, el modelo predecía casi siempre `insectos` con `trigo` o `maiz`.
El modelo no tiene una clase «ninguna de estas». Su softmax reparte siempre el 100 % entre las 4
categorías y los 4 cereales, así que ante cualquier foto de otro tipo responde con alguna de sus
clases. Ocurre con hojas de vid, insectos sueltos, grano o imágenes térmicas, y además con más de
0,9 de confianza.

Por eso un umbral de confianza no sirve. Con un umbral que conserve el 95 % de las imágenes reales,
la confianza solo detecta el 1-7 % de las hojas de vid, las garrapatas o los insectos.

## Solución

El filtro está en `app/plugins/ml8_cereals_img_anomaly_detector/ood.py` y **no toca el modelo**:
las predicciones de las imágenes del tipo con el que se entrenó salen igual que antes.

- **Qué mide.** Usa las características internas de la red (960 valores, la entrada de sus dos
  cabezas). Con ellas calcula una regresión logística entre imágenes reales e imágenes de otros
  dominios, y le suma la distancia a la imagen de entrenamiento más parecida.
- **Umbral.** Se fija para conservar el 95 % de imágenes reales no usadas en el ajuste
  (validación cruzada de 5 particiones).
- **Salida.** Por encima del umbral, `categoria` y `cereal` valen `desconocido`, y
  `es_desconocido` vale `true`. Las etiquetas que habría dado la red se guardan en
  `categoria_modelo` y `cereal_modelo`. `confianza_*` y `probabilidades_*` siguen siendo la salida
  de la red. `puntuacion_ood` y `umbral_ood` permiten ver el margen.
- **Ficheros.** Son nuevos y van junto al checkpoint, en
  `s3://<STORAGE_BUCKET>/artifacts/fixed/ml8_cereals_img_anomaly_detector/`:
  - `ood_reference.npz`, con el banco de características, la regresión y el umbral (1 MB);
  - `ood_outliers.zip`, con 1.441 imágenes de otros dominios a 256 px (20 MB).
- **Sin esos ficheros**, el plugin sirve como antes, sin detectar «desconocido».
- **Reentrenamiento.** `/train` recalibra el filtro para el modelo reentrenado. Usa como banco sus
  imágenes de entrenamiento y como referencia para el umbral las de validación, con los mismos
  outliers. Lo guarda en su run de MLflow. Los runs anteriores al filtro no lo tienen y sirven como
  antes.

## Datos

| Conjunto | Origen | Imágenes |
|---|---|---|
| Reales de ml8 | `s3://…/datasets/modelo-8/{cereal}/train/{categoria}/` (única copia accesible) | 600 |
| Hojas de vid | a01 `data/raw/IMAGENES`, a02 `data/raw/datasets` | 200 + 200 |
| Heces de ave | a06 `data/fecal/raw` | 200 |
| Moscas/mosquitos, garrapatas, insectos sobre ganado | a10 `data/classification/raw`, `data/detection/raw/insects` | 200 × 3 |
| Vacas | a05 `data/images/raw_frames` | 200 |
| Térmicas de ubre, plagas del grano | `outputs/a04/datos_prueba`, `outputs/a07/datos_prueba` | 18 + 23 |
| Control: sorgo real no usado | a10 `Sorghum Disease Image Dataset`, una de las fuentes del dataset de ml8 | 200 |

## Resultados (2026-11-13, `--evaluate`)

En cada fila, ese dominio se deja **fuera** del ajuste: mide cómo se comporta el filtro con un
tipo de imagen que no ha visto nunca. El umbral conserva el 95 % de las imágenes reales.

| Dominio no visto | Solo distancia k-NN | Filtro (regresión + k-NN) |
|---|---|---|
| Hojas de vid (a01) | 62 % | 95 % |
| Hojas de vid (a02) | 61 % | 100 % |
| Heces de ave (a06) | 10 % | 67 % |
| Moscas/mosquitos (a10) | 31 % | 89 % |
| Garrapatas (a10) | 60 % | 81 % |
| Insectos sobre ganado (a10) | 24 % | 66 % |
| Vacas (a05) | 84 % | 96 % |
| Térmicas de ubre (a04) | 98 % | 69 % |
| Plagas del grano (a07) | 81 % | 77 % |

- **Sorgo real no usado** marcado como «desconocido»: **0,6 %** (200 imágenes).
- **Filtro final**, ajustado con todos los dominios: supera el umbral el 99,4 % de los outliers.

Comprobación con el plugin real:

| Conjunto | Resultado |
|---|---|
| `outputs/a08/datos_prueba/predict_batch_12.zip` | 0/12 «desconocido», con las 4 categorías |
| Golden de a02 | 18/18 «desconocido» |
| Golden de a10 | 24/25 «desconocido» |
| Golden de a07 | 23/23 «desconocido» |

Las tres últimas filas son de dominios incluidos en el ajuste, así que no miden la generalización.

**Límites.**
- Las 600 imágenes reales son del split de train, porque no hay otra copia. Con imágenes reales
  nunca vistas, la tasa de «desconocido» puede superar el 5 %. El control de sorgo (0,6 %) indica
  que no es el caso.
- Un tipo de imagen muy distinto de todos los dominios usados puede pasar sin detectarse. Cuantos
  más dominios tenga `ood_outliers.zip`, mejor generaliza.

## Regenerar

```
.venv/bin/python outputs/a08/generar_referencia_ood.py --evaluate          # baja las 600 de S3
.venv/bin/python outputs/a08/generar_referencia_ood.py --id-dir <dir> --evaluate
```

El script necesita los `inbox/aNN/codigo/` de a01, a02, a05, a06 y a10, que no se commitean.
