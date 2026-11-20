# ml10: detecciones «desconocido»

## Problema

El desarrollador observó dos cosas al probar el modelo en la plataforma:

1. **Ficheros que no son imágenes.** Con un JSON (u otro fichero de datos), inline o en lote, la
   tarea fallaba con «features debe contener 'image_path' o 'image_base64'». El mensaje no decía
   qué había pasado ni qué había que subir.
2. **Imágenes ajenas al modelo.** Con fotos que no tienen nada que ver (hojas de vid, vacas, heces
   de ave, grano…) devolvía moscas, mosquitos o garrapatas con confianza alta.

ml10 detecta, recorta y clasifica, y ninguna de las dos etapas tiene una clase «ninguna de
estas». El detector YOLO dispara sobre cualquier cosa con forma de bicho (det_conf 0,2). El
clasificador MobileNetV3 reparte siempre el 100 % entre `fly`, `mos` y `tick`: con un recorte de
hoja de vid responde con alguna de las tres y a menudo con más de 0,9.

## Solución

### Punto 1: mensajes de error

`plugin.py` comprueba la extensión y el contenido antes de predecir. Cualquier entrada que no sea
una imagen o un ZIP de imágenes devuelve un **422** con un mensaje que explica qué ha llegado y qué
hay que usar. El front lo muestra tal cual. Ejemplo con un JSON:

> Formato de fichero no soportado (fichero .json). Este modelo analiza fotografías (JPG, JPEG,
> PNG, BMP o TIFF): una imagen en la predicción individual o un ZIP de imágenes en la predicción
> por lotes.

Lo mismo para un base64 que no es una imagen, un ZIP sin imágenes (indica qué contiene, p. ej.
«contiene: 2 .json») o una petición sin imagen.

### Punto 2: «desconocido» por detección

Es el mismo filtro que ml8 (`app/domain/services/feature_ood.py`, compartido) aplicado **a cada
recorte** en `app/plugins/ml10_dairy_disease_vector_detection/ood.py`. **No toca el modelo**: el
detector y el clasificador dan lo mismo que antes.

- **Qué mide.** Usa las características internas del clasificador (960 valores, la entrada de su
  cabeza), calculadas en la misma pasada. Combina la distancia al recorte real más parecido con una
  regresión logística entre recortes reales y recortes de otros dominios.
- **Umbral.** Conserva el 95 % de los recortes reales de validación (no están en el banco).
- **Salida por detección.** Por encima del umbral, `species` vale `desconocido` y
  `es_desconocido` vale `true`. La etiqueta de la red se guarda en `species_modelo`, y
  `cls_conf`, `det_conf` y la caja no cambian. `puntuacion_ood` da el margen.
- **Salida por imagen.**
  - Las detecciones «desconocido» no cuentan en `vectors_count` ni en `species_summary`, no
    tienen Grad-CAM y se dibujan en gris. `unknown_count` las cuenta.
  - Si todas las detecciones son «desconocido», `prediction` es `desconocido`, `confidence` es 0
    y `es_desconocido` es `true`.
  - Si hay alguna reconocida, la predicción es la especie dominante entre ellas.
  - `umbral_ood` va en la respuesta y en cada fila del lote.
- **Front.** Muestra un aviso en gris que explica por qué no se cuenta como vector y una tarjeta
  de «Detecciones no reconocidas»; en lote añade la columna «No reconocidas».
- **Ficheros.** Son nuevos y van junto al clasificador, en
  `s3://<STORAGE_BUCKET>/artifacts/fixed/ml10_dairy_disease_vector_detection/`:
  - `ood_reference.npz` (11,7 MB): banco de características, regresión y umbral;
  - `ood_crops.zip` (120 MB): recortes a ≤ 256 px de lado corto. Contiene `id_train/` y `id_val/`
    (recortes reales del detector) y `outliers/<dominio>/` (8 dominios).
- **Sin esos ficheros**, el plugin sirve como antes, sin «desconocido».
- **Reentrenamiento.** El backbone está congelado, pero las estadísticas de BatchNorm cambian, así
  que las características también. `/train` recalibra el filtro con la misma función que el script
  (`build_crops_reference`):
  - banco: sus recortes de train más `id_train/`;
  - umbral: sus recortes de validación más `id_val/`;
  - regresión: frente a `outliers/`.

  Lo guarda en su run de MLflow junto al clasificador y registra la métrica `ood_threshold`. Si
  falta `ood_crops.zip`, o el run es anterior al filtro, ese modelo sirve sin «desconocido».

## Datos

| Conjunto | Origen | Recortes |
|---|---|---|
| Banco | Recortes de clasificación de train (a10 `classification/splits/train`) | 1.920 |
| | Detector sobre `detection/splits/train`, coincidentes con caja anotada (IoU ≥ 0,5) | 3.096 |
| | Detector sobre las fotos originales de clasificación cuyos recortes están en train | 1.653 |
| Validación (umbral) | Lo mismo con los splits de val | 410 + 636 + 344 |
| Outliers | Hojas de vid a01 y a02, heces de ave a06, vacas a05, sorgo (dentro del raw de a10, no usado por ml10), térmicas de ubre a04, plagas del grano a07, cereales a08 | 1.656 |
| Solo evaluación | Test de clasificación, de detección y de fotos; fotos no usadas en ningún split; gráficos y capturas de a17, a19, a21, a26, a28 y a41 | — |

Los recortes de las **fotos originales** se añadieron porque el detector no recorta igual que el
preprocesado de a10, que recorta la caja anotada con un 10 % de margen. Sin ellos, el 8,4 % de los
recortes de fotos reales no usadas salía «desconocido»; con ellos, el 4,4 %.

## Resultados (2026-11-19, `--evaluate`)

Porcentaje de recortes marcados «desconocido». En la columna «Dominio fuera del ajuste», ese
dominio se quita de los outliers: mide cómo se comporta el filtro con un tipo de imagen que no ha
visto nunca.

| Conjunto | Recortes | Solo k-NN | Dominio fuera del ajuste | Filtro final |
|---|---|---|---|---|
| Reales: clasificación test | 415 | 4,1 % | — | 2,4 % |
| Reales: detector test (caja anotada) | 705 | 6,1 % | — | 6,1 % |
| Reales: fotos de clasificación de test | 342 | 2,3 % | — | 2,0 % |
| Reales: fotos sin usar en ningún split | 2.597 | 2,3 % | — | 4,5 % |
| Detector test sin caja anotada (falsos positivos) | 176 | 30,7 % | — | 31,8 % |
| Heces de ave (a06) | 214 | 64,5 % | 90,2 % | 100 % |
| Cereales (a08) | 8 | 75,0 % | 75,0 % | 100 % |
| Plagas del grano (a07) | 16 | 62,5 % | 62,5 % | 100 % |
| Hojas de sorgo | 213 | 79,8 % | 77,5 % | 100 % |
| Térmicas de ubre (a04) | 5 | 100 % | 100 % | 100 % |
| Vacas (a05) | 400 | 95,0 % | 59,8 % | 100 % |
| Hojas de vid (a01) | 400 | 69,5 % | 99,0 % | 100 % |
| Hojas de vid (a02) | 400 | 70,0 % | 100 % | 100 % |
| Gráficos y capturas (nunca en el ajuste) | 400 | 100 % | — | 97,8 % |

Comprobación con el plugin real (`ood_reference.npz` guardado, umbral 0,0411):

| Conjunto | Resultado |
|---|---|
| Golden inline del manifest | 25/25 correctos |
| Golden batch | 0 de 25 detecciones «desconocido» |
| Test de detección de a10, por imagen | 52 de 640 imágenes con detecciones, todas «desconocido» (8,1 %) |
| Fotos de test, moscas/mosquitos | 1,2 % de imágenes «desconocido» |
| Fotos de test, garrapatas | 5,0 % |
| Fotos no usadas, garrapatas (300) | 6,5 % |
| Heces de ave, vacas, vid, grano, térmicas, cereales | Solo «desconocido» o sin vectores, salvo 1 mosca en heces de ave (de 30) |

**Límites.**
- **Falso «desconocido» en el test de detección.** Por imagen es del 8 %, por encima del 5 % de
  diseño. Por recorte anotado es del 6 %. No lo explican los insectos que el clasificador no
  conoce: sin las avispas asiáticas, las libélulas y los ciempiés de ImageNet que el dataset de
  detección incluye como «Vector», sigue en el 7,8 %. Afecta a fotos con un solo insecto pequeño
  o atípico.
- **Fotos de fondo.** 552 fotos de FLY_MOS no tienen ningún insecto anotado. En una muestra de
  300 fotos no usadas, el detector dispara en el 70 % y el filtro marca «desconocido» el 67 % de
  esas imágenes. Es lo correcto,
  pero el resto sigue contando como vector.
- **Dominios muy distintos de todos los usados** pueden pasar sin detectarse. Cuantos más dominios
  tenga `outliers/`, mejor generaliza.
- **El tamaño de `ood_crops.zip` (120 MB)** se debe a que `/train` necesita las imágenes para
  recalibrar. Solo se descarga al reentrenar, no al servir.

## Regenerar

```
.venv/bin/python outputs/a10/generar_referencia_ood.py --evaluate --work-dir ~/.cache/ml10_ood_work
```

- Escribe `ood_reference.npz` y `ood_crops.zip` en `artifacts/ml10_dairy_disease_vector_detection/`.
- Necesita los `inbox/aNN/codigo/` de a01, a02, a05, a06, a10, a17, a19, a21, a26, a28 y a41, que
  no se commitean.
- Con `--work-dir`, las etapas de recortes ya hechas se reutilizan. Desde cero tarda horas en CPU:
  pasa el detector y el clasificador por las ~10.500 fotos de a10 y las de los otros dominios.
