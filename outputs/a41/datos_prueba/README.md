# Datos de prueba — ml41-meat-curing-machinery-acoustic-anomaly

Audio-MAE ViT-Tiny + distancia de Mahalanobis: detecta anomalías en grabaciones de 10 s de una máquina.
`machine`, `machine_id` y `snr` no son medidas: eligen cuál de los 48 checkpoints puntúa el audio.

**Origen del audio.** La entrega del equipo de IA llega sin audio (`inbox/a41/codigo/data/raw` vacío). Los WAV
son los del dataset público MIMII (Zenodo 3384388), carpeta `-6_dB_fan/id_00/`. Son los mismos ficheros
de los 22 `golden_cases` del manifest (ver `golden_cases.audio_source_real`). No hay valores inventados.
Los WAV originales tienen 8 canales y aquí van en **mono float32**: es la misma señal que obtiene el plugin con
`librosa.load(mono=True)` (espectrograma idéntico bit a bit) y ocupa la cuarta parte.

Regenerar (descarga ~120 MB de Zenodo por rangos, ~1 min):
`.venv/bin/python outputs/a41/datos_prueba/generar_datos_prueba.py "$PWD"`

| Fichero | Endpoint | Contenido |
|---|---|---|
| `predict_inline_caso_001_anomalo.json` | `POST .../predict` (inline) | WAV anómalo en base64 → `predicted_label` 1 |
| `predict_inline_caso_012_normal.json` | ídem | WAV normal → `predicted_label` 0 |
| `predict_inline_caso_013_falso_positivo.json` | ídem | WAV normal que el modelo marca como anómalo (falso positivo documentado) |
| `predict_batch_golden.zip` | `POST .../predict` con `{"mode": "batch", "data_path": "<ruta>"}` | `manifest.csv` (`filename,machine,machine_id,snr`) + los 22 WAV de los golden cases |
| `expected_golden.csv` | — | `maha_score` y `predicted_label` esperados de los 22 casos (umbral 6.6067) |
| `train_fan_id00_-6dB.zip` | `POST .../train` con `{"data_path": "<ruta>", "mlflow_run_id": "<run>"}` | `-6_dB_fan/id_00/normal/` (40 WAV) + `abnormal/` (los 11 anómalos de los golden) |

Tolerancias (manifest): `predicted_label` exacto; `maha_score` ±0,5 % relativo; `mse_score` **no se compara**
(el original enmascara parches al azar sin semilla, varía ~5 % entre llamadas).

Los golden cases se puntúan con el modelo **`fan/id_00/0_dB`**, aunque el audio sea de -6 dB: así se fijaron
en el manifest (la combinación se identificó por estadísticas del espectrograma, que acertaron máquina y
unidad pero no el SNR). Los valores esperados son los de esa combinación.

**Train.** Entrena desde cero, con el procedimiento original, la combinación `fan/id_00/-6_dB`. El
resultado se guarda **solo en el run de MLflow** y no toca el checkpoint base. Para predecir con él, hay que
pasar `mlflow_run_id` y `"snr": "-6_dB"`. Con otra combinación responde `UserModelUnavailableError` y no
recurre al modelo base. 40 normales es el mínimo para `fan`: el split 80/20 deja 32 de train, que es el
`batch_size`, y el original descarta el lote incompleto. Con menos clips, error explícito. Es una
prueba de humo: con 32 clips el modelo no aprende casi nada (AUC 0,52). El reentrenamiento real usa los
~1000 normales de la combinación.

## Resultados comprobados en local (2026-11-09)

- Inline: los 3 JSON dan el `maha_score` y la etiqueta esperados.
- Batch: 22/22 dentro de tolerancia (diferencia relativa máx. de `maha_score` 2,8e-4, en GPU), etiquetas
  idénticas, recall 1,0 y FPR 0,818 como en el manifest (~2 s).
- Train (MLflow simulado, artefactos base sin tocar): `best_val_loss` 0,888994; n_train/n_val 32/8;
  11 anómalos; auc_maha 0,5227; auc_mse 0,3636; fnr 0,0909; fpr 0,875; recall 0,9091; umbral 41,8206.
  Idéntico en dos ejecuciones en GPU. Tiempo: ~66 s en GPU.
- Train en CPU: ~465 s (~8 min). Las métricas difieren algo porque el original solo usa autocast bf16 en
  CUDA (`best_val_loss` 0,890167; auc_maha 0,5341; fnr 0; fpr 0,75; recall 1,0; umbral 44,2053). No son un
  valor esperado fijo: dependen del dispositivo.
- Predict con el modelo reentrenado (`mlflow_run_id` + `snr` -6_dB) sobre el caso 001: maha_score 61,40,
  `predicted_label` 1 con el umbral nuevo 41,82.
