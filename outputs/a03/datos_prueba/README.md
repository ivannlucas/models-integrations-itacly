# Datos de prueba — ml3-wine-disease-pest-forecast

Deep Ensemble (LSTM + CNN-1D + BiGRU) que, con la última ventana de 168 h de sensores de una serie,
diagnostica 11 clases de enfermedad o plaga de la vid y estima la severidad (0-1). Para regenerarlos
(descarga unos 6 MB de Open-Meteo la primera vez, ~4 min):
`.venv/bin/python outputs/a03/datos_prueba/generar_datos_prueba.py "$PWD"`

**Origen.** No había ninguna copia del dataset etiquetado. `inbox/a03/codigo/` no se commitea, el
"contenedor externo" que cita su README no tiene enlace y `outputs/a03/datasets/train.csv` (igual que
`s3://…/datasets/modelo-3/train.csv`) son solo las 149 series de **test**, sin etiquetas. El script
restaura en `inbox/a03/codigo/` el código del equipo de IA desde git (`29dc9a0^`) y ejecuta su propio
pipeline: `descarga_clima_historico.py` (Open-Meteo), `data_cleaner.py` y `vid_simulator.py` (semilla
42). Antes de escribir nada, comprueba que las 149 series de test regeneradas son **idénticas** al
original. Lo son: 249 406 filas, los 8 sensores bit a bit en float32, `Parcela_ID` y la longitud de
cada serie. También cuadran el total de ventanas de test del manifest (224 374) y sus `train_range`.
Los `.keras`, `scaler.pkl` y `label_encoder.pkl` restaurados tienen el mismo md5 que los servidos.

Hay un ajuste: el dataset original lleva las marcas de tiempo 1 h por delante de las que devuelve hoy
Open-Meteo con `timezone=Europe/Madrid`. Por eso el script suma 1 h al clima antes de simular. Sin ese
desfase no coincide casi ninguna fila; con él coinciden todas.

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_batch_golden.parquet` / `.csv` | `POST /models/ml3-wine-disease-pest-forecast/predict` con `{"mode": "batch", "data_path": "<ruta>"}` | Las 20 series de los golden_cases del manifest (últimas 168 h de cada una, 3 360 filas) |
| `expected_golden.csv` | — | Salida esperada (`expected` del manifest) y la clase y el grado reales de la última hora |
| `predict_batch_series_cortas.parquet` / `.csv` | ídem | 3 series reales de menos de 168 h (50, 133 y 153 filas): la rama de *padding*, que los golden no cubren |
| `expected_series_cortas.csv` | — | Salida del `run_inference` original del equipo sobre el parquet (ver "NumPy" abajo) |
| `train_88_series.csv` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | 8 series por clase del split de train/val original (ninguna de test), las más cortas con ≥ 336 h. Son 83 160 filas, con `Etiqueta_Clase`, `Clase_Entrenamiento` y `Grado_Infeccion` |

El parquet (float32) es el formato original y la referencia exacta. El CSV es la misma tabla, para
subirla desde el front. No hay ficheros inline: en ml3, `"mode": "inline"` responde 422 por decisión de
producto.

## Resultados comprobados en local (2026-11-12)

- **Código original vs manifest:** el `run_inference` del equipo, con la semántica de NumPy 1.26,
  reproduce los 20 golden: clase exacta y floats a ≤ 5e-5 (el original redondea a 4 decimales).
- **Plugin, batch golden: FALLA LA TOLERANCIA en 8 de 20.** Acierta la clase en los 20, pero
  `caso_002/003/005/009/013/014/019/020` superan `rtol 0.01` (máximo 0,095 en confianza). Sale igual con
  el parquet y con el CSV. El modelo acierta 11 de 20 frente a la etiqueta real, como en el manifest.
- **Plugin, series cortas:** acierta la clase en las 3, pero 2 de 3 superan `rtol 0.01` (series 585 y
  594, máximo 0,22). Es la misma causa.
- **Train** (MLflow simulado, artefactos base sin tocar; md5 comprobado): con `EPOCHS=1` tarda 86 s
  en CPU (aquí TensorFlow no ve la GPU), con un pico de RAM de 2,9 GB. Ventanas: 45 801 de train,
  10 919 de val y 11 656 de test. Métricas: acc 0,62 y f1_macro 0,34. Es solo una prueba de humo. El
  procedimiento completo (hasta 50 épocas × 3 redes con early stopping de paciencia 2) no se ha
  ejecutado: en CPU se estiman entre 10 y 40 min.

## Fallo de fidelidad del plugin con NumPy 2 (sin corregir)

`feature_engineering._calcular_horas_mojado_serie` hace `lluvia[t] > 0.1` sobre un elemento float32.
Con NumPy 1.26, el entorno del manifest y del entrenamiento, se compara en float64: 0,1 en float32 vale
0,10000000149 > 0,1, así que la hora cuenta como **mojada**. Con NumPy 2 (NEP 50; el venv tiene 2.5.3)
se compara en float32 y la hora cuenta como **seca**. Una lluvia de exactamente 0,1 mm es frecuente,
porque Open-Meteo da una resolución de 0,1 mm. Eso cambia `Horas_Humedad_Foliar` y, con ello, las
predicciones. Al leer un CSV, el valor ya llega como float64 0,1 y queda seco con cualquier NumPy.

Para comprobarlo, se convirtieron los valores a float32 y luego a float64 antes de comparar
(monkeypatch, sin tocar el plugin). Con eso, los 20 golden vuelven a ≤ 1e-6, tanto desde el parquet
como desde el CSV. Esa es la corrección candidata. No se ha aplicado ni se ha tocado ninguna
tolerancia. El `verification_report.md` de a03 (2026-08-12, 20/20 a 1e-6) se hizo con NumPy 1.x.

## Otros avisos

- `outputs/a03/datasets/train.csv` y `s3://…/datasets/modelo-3/train.csv` **no sirven para `/train`**:
  no tienen `Clase_Entrenamiento` ni `Grado_Infeccion`. Son el split de test sin etiquetas y valen como
  entrada de batch (149 series).
- `/train` necesita también `Etiqueta_Clase`, la clase de la serie que se usa para el split
  estratificado. No está en `TRAIN_HARD_REQUIRED_COLUMNS`: si falta, el error es un `KeyError` y no el
  mensaje de columnas requeridas.
