# Verificación — ml41-meat-curing-machinery-acoustic-anomaly (a41)

**Fecha:** 2026-10-02
**Plugin:** `app/plugins/ml41_meat_curing_machinery_acoustic_anomaly/`
**Manifest:** `inbox/a41/manifest.yaml` (22 golden_cases, combo fan/id_00/0_dB)
**Motivo de esta verificación:** el plugin ya estaba integrado (manifest + golden_cases
completos) pero nunca se había generado este informe — gap detectado en una auditoría
repo-wide del estado de `inbox/`/`outputs/` de los 25 plugins registrados.

> **[CERRADO 2026-10-10] Verificación con el dataset MIMII completo.** Ya están en `inbox/a41/codigo/data/`
> los 54.057 WAV crudos y los `.npy` procesados de train/val/test.
>
> **Golden cases (Parte B): 22/22 ✅.** Ejecutados contra el endpoint HTTP real. Origen corregido: el audio de
> `sample_data/` son los ficheros homónimos de `-6_dB_fan/id_00`, no de `0_dB_fan/id_00`. Con los de 0 dB
> fallaban los 22, con diferencias del 10-120 %; un barrido de las 48 carpetas localizó la de −6 dB. Los
> valores esperados corresponden a ese audio puntuado con el modelo de `fan/id_00/0_dB`.
>
> | Caso | Fichero | Etiqueta real | maha esperado | maha obtenido | Δ rel. | Etiqueta | ¿OK? |
> |---|---|---|---|---|---|---|---|
> | caso_001 | abnormal/00000000.wav | 1 | 15.2161 | 15.2191 | 2.0e-04 | 1 | ✅ |
> | caso_002 | abnormal/00000001.wav | 1 | 12.2056 | 12.2023 | 2.7e-04 | 1 | ✅ |
> | caso_003 | abnormal/00000002.wav | 1 | 9.3580 | 9.3582 | 1.7e-05 | 1 | ✅ |
> | caso_004 | abnormal/00000003.wav | 1 | 8.2902 | 8.2919 | 2.0e-04 | 1 | ✅ |
> | caso_005 | abnormal/00000004.wav | 1 | 16.2408 | 16.2388 | 1.2e-04 | 1 | ✅ |
> | caso_006 | abnormal/00000005.wav | 1 | 10.3027 | 10.3040 | 1.2e-04 | 1 | ✅ |
> | caso_007 | abnormal/00000006.wav | 1 | 6.7848 | 6.7856 | 1.3e-04 | 1 | ✅ |
> | caso_008 | abnormal/00000007.wav | 1 | 9.7266 | 9.7263 | 3.1e-05 | 1 | ✅ |
> | caso_009 | abnormal/00000008.wav | 1 | 6.8919 | 6.8922 | 3.8e-05 | 1 | ✅ |
> | caso_010 | abnormal/00000009.wav | 1 | 13.6871 | 13.6875 | 3.0e-05 | 1 | ✅ |
> | caso_011 | abnormal/00000010.wav | 1 | 13.2500 | 13.2477 | 1.8e-04 | 1 | ✅ |
> | caso_012 | normal/00000000.wav | 0 | 6.1289 | 6.1290 | 3.0e-05 | 0 | ✅ |
> | caso_013 | normal/00000001.wav | 0 | 12.8497 | 12.8506 | 7.6e-05 | 1 | ✅ |
> | caso_014 | normal/00000002.wav | 0 | 5.9723 | 5.9706 | 2.8e-04 | 0 | ✅ |
> | caso_015 | normal/00000003.wav | 0 | 12.3090 | 12.3100 | 7.9e-05 | 1 | ✅ |
> | caso_016 | normal/00000004.wav | 0 | 10.7194 | 10.7193 | 1.4e-05 | 1 | ✅ |
> | caso_017 | normal/00000005.wav | 0 | 11.6504 | 11.6496 | 7.3e-05 | 1 | ✅ |
> | caso_018 | normal/00000006.wav | 0 | 8.3878 | 8.3890 | 1.4e-04 | 1 | ✅ |
> | caso_019 | normal/00000007.wav | 0 | 8.5015 | 8.5019 | 4.0e-05 | 1 | ✅ |
> | caso_020 | normal/00000008.wav | 0 | 17.2392 | 17.2396 | 2.6e-05 | 1 | ✅ |
> | caso_021 | normal/00000009.wav | 0 | 11.0048 | 11.0024 | 2.1e-04 | 1 | ✅ |
> | caso_022 | normal/00000010.wav | 0 | 10.0492 | 10.0473 | 1.9e-04 | 1 | ✅ |
>
> Recall 1,0 y FPR 0,818 (9 falsas alarmas en 11 normales), como declaraba el manifest. Tolerancia: etiqueta
> exacta y `maha_score` ≤ 0,5 %; la diferencia máxima es 2,8e-4, en GPU.
>
> | Comprobación con datos reales | Resultado |
> |---|---|
> | Métricas de las 48 combinaciones con el plugin (normales de val frente a anómalos de test, `.npy` originales, unos 29.000 espectrogramas, GPU) frente a `table_auc_por_caso.csv` | AUC máx. Δ 2,9e-4, media idéntica (0,7720); FNR 0,0912 frente a 0,0909; FPR 0,4684 idéntico; 48/48 con FNR ≤ 10 %; FNR igual en 44/48 (4 difieren en una muestra); umbrales a ≤ 1,5 % |
> | Reentrenamiento real de `fan/id_00/0_dB` en GPU: plugin (desde los WAV crudos) frente a `run_training` original (desde los `.npy`) | **Iguales a 4 decimales**: best_val_loss 0,7400; AUC Maha 0,7896; umbral 7,2069; recall 0,9017; auc_mse 0,4221; 808/203 |
> | Ese reentrenamiento frente al modelo entregado | val_loss 0,7400 frente a 0,7401; AUC 0,7896 frente a 0,7585. La diferencia es variación del propio original entre ejecuciones en GPU |
>
> **Corregido además.** En CUDA el original entrena con autocast bf16 y GradScaler, y el plugin entrenaba en
> fp32. Ahora lo hace igual (solo en CUDA). Sin este cambio, el reentrenamiento en GPU del plugin no habría
> coincidido con el del original.
>
> **Estado: LISTO PARA PR.** Solo queda la acción operativa de subir los 48 directorios `vit_tiny_*` a S3.
> Tests: 909 passed; flake8 limpio; pylint 9,33/10.

> **[ACTUALIZADO 2026-10-10] Reanálisis con el código original (`inbox/a41/codigo/`, memoria v1.4).**
> Llegaron los 48 checkpoints Audio-MAE y los 48 del baseline, pero ningún audio: `data/` está vacío y
> `sample_data/` no está en el repo ni en este equipo.
>
> | Comprobación | Resultado |
> |---|---|
> | Umbrales del plugin (`thresholds.py`) frente a `reports/table_auc_por_caso.csv` | 48/48 idénticos; las 48 combinaciones cumplen FNR ≤ 10 % |
> | Inferencia plugin frente al original (`wav_to_logmel` + `load_model` + `mahalanobis_scores`), 48 combinaciones × 3 WAV sintéticos (10 s a 16 kHz, 5 s con relleno, 44,1 kHz estéreo), CPU | Espectrograma idéntico; `maha_score` con diferencia relativa ≤ 4,8e-7; `predicted_label` idéntico en 144/144 |
> | Arquitectura y pesos | `state_dict` y módulos idénticos; mismo CLS con la misma entrada y la misma semilla |
> | `/train` frente a `run_training` original (`force=True`, 2 épocas, 40 WAV sintéticos de fan, CPU) | **Pesos idénticos (diferencia 0)**, misma normalización (−18,055 / 4,788), mismo split (32/8) y mismo `best_val_loss` |
> | Servidor real con los checkpoints en `artifacts/` local | `/health` loaded; `/predict` inline 200 (umbral de la tabla); `/train` sin run 422 |
>
> **Corregido en `/train`.** Hacía un ajuste fino de los pesos del checkpoint base, con su normalización.
> El original nunca lo hace: `run_training` se salta el entrenamiento si ya existe `best.pth` y, con
> `force=True`, entrena un modelo nuevo. Además se corrigieron tres detalles:
> - el split usaba `default_rng` en vez de `RandomState(42)`;
> - la semilla se fijaba después de crear el modelo;
> - `drop_last` era condicional.
>
> El nuevo `/train` reproduce el original al bit. En CPU tarda unas 1-1,5 h por combinación con datos
> reales; el original tardaba 155 s en GPU.
>
> **Precisión sobre `maha_score`.** No es determinista al bit: con `mask_ratio=0` el encoder sigue
> barajando parches. La variación entre ejecuciones es ≤ 6e-7 relativo. Entre CPU y GPU llega a 1,7e-4,
> lo que solo importa con un score justo en el umbral.
>
> **Sigue pendiente:**
> - los 22 golden cases: hace falta `sample_data/` (fan/id_00/0_dB);
> - subir los 48 directorios `vit_tiny_*` a S3.
>
> Tests: 909 passed; flake8 limpio; pylint 9,32/10 (igual que antes del cambio). El texto de abajo es
> la revisión del 2026-10-02.

## Hallazgo crítico — los checkpoints NO existen en el S3 de producción

Arrancando el servicio real contra el bucket S3 configurado en `.env`
(`MODEL=ml41-meat-curing-machinery-acoustic-anomaly python main.py`):

```
WARNING - No files found in S3 for model: ml41_meat_curing_machinery_acoustic_anomaly
WARNING - Ml41 plugin loaded but no combination checkpoints were found under ml41_meat_curing_machinery_acoustic_anomaly
INFO    - Ml41MeatCuringMachineryAcousticAnomalyPlugin loaded: ready=False device=cpu
```

`/health` responde `200 {"loaded": false}` y **`/predict` responde 500 en toda petición**:
```json
{"detail":"El modelo no está cargado (no hay checkpoints disponibles)."}
```

`s3://<STORAGE_BUCKET>/artifacts/fixed/ml41_meat_curing_machinery_acoustic_anomaly/` está
**vacío** — los ~2 GB de las 48 combinaciones (`best.pth` + `maha_stats.npz` por
machine×machine_id×snr, ver `inbox/a41/manifest.yaml` → `artifacts`) nunca se subieron a S3.
Esto **no es un bug de código**: el plugin detecta la ausencia correctamente (`ready=False`,
error explícito en vez de un 500 críptico o un crash), pero el modelo **no sirve ni una sola
predicción real tal y como está desplegado hoy**.

**Acción requerida (no es algo que yo pueda resolver desde aquí):** subir los 48 directorios
de checkpoint a `s3://<STORAGE_BUCKET>/artifacts/fixed/ml41_meat_curing_machinery_acoustic_anomaly/`
siguiendo `path_pattern` del manifest. Hasta entonces el modelo está caído en producción.

## Checklist técnico (Parte A)

- [x] **flake8**: 0 hallazgos.
- [x] **pytest** (`tests/unit/test_ml41_meat_curing_machinery_acoustic_anomaly.py` +
      `test_ml41_threshold_persistence.py`): 9/9 passed. Usan `FakePlugin` — validan wiring
      HTTP/esquema, no el modelo real (mismo alcance que el resto de la suite).
- [x] **pylint** (`app/plugins/ml41_.../`): 8.92/10. Solo avisos de estilo en `trainer.py`
      (complejidad, imports locales de sklearn) — código de entrenamiento, no de inferencia.
- [x] **pip-audit**: 2 CVEs en `setuptools==80.9.0` (PYSEC-2026-3447). Preexistente y
      transversal a todo el repo, no específico de a41.
- [x] **Arranque local contra S3 real** (`MODEL=ml41-... python main.py`): arranca limpio,
      `/health` responde 200. **Pero `ready=False`** — ver hallazgo crítico arriba.
- [~] **`/predict`**: responde 500 explícito por falta de artefactos (ver arriba) — no se
      puede verificar el pipeline de inferencia real end-to-end en este entorno hasta que los
      checkpoints estén en S3.
- [x] **`/stats`**: 200 OK, responde con la descripción/inputs/outputs del modelo
      correctamente aunque `ready=False`.
- [~] **`/train`**: registrado correctamente en `app/registry.py` (train_request/response
      cableados, manifest `training.supported=true`), pero no ejercitado end-to-end — requiere
      los 48 checkpoints base para fine-tune y no están disponibles (mismo bloqueo).

## Correctitud contra golden dataset (Parte B) — BLOQUEADA, dos motivos independientes

1. **Checkpoints no disponibles en S3** (ver hallazgo crítico) — `/predict` no puede ejecutar
   ninguna inferencia real ahora mismo, con o sin los WAV de prueba.
2. **Audio de los golden_cases no disponible localmente**: `inbox/a41/manifest.yaml` referencia
   `sample_data/{abnormal,normal}/*.wav` (22 ficheros, combo fan/id_00/0_dB) que vivían en
   `inbox/a41/codigo/` — carpeta ya eliminada tras el manifest-extraction original (es
   gitignored por convención del repo, `inbox/*/codigo/`). No se ha podido re-ejecutar la
   Parte B en esta revisión.

**Para completar la Parte B hace falta, como mínimo, una de estas dos cosas**: (a) los 48
checkpoints en S3, y (b) los 22 WAV de `sample_data/` (o el `inbox/a41/codigo/` completo de
nuevo). Con ambas, los 22 golden_cases ya definidos en el manifest son directamente ejecutables
sin más trabajo de extracción.

## Puntos ya documentados en el manifest (confirmados, no resueltos en esta revisión — no son
## bugs de integración, son limitaciones reales del modelo entregado)

- **AUC pobre en `valve`** (0.45–0.61 según combinación) — el propio manifest ya lo marca como
  "no recomendado para despliegue autónomo sin supervisión humana" para esa máquina.
- **`mse_score` no determinista** (~5-6% de variación entre ejecuciones) — documentado como
  informativo únicamente; el golden dataset usa tolerancia `informational_only` para ese campo
  y `exact`/`0.5%` solo para `predicted_label`/`maha_score` (deterministas).
- **Solo 1 de 48 combinaciones tiene golden_cases** (fan/id_00/0_dB) — las otras 47 no tienen
  casos de referencia extraídos. Dentro de esa combinación, los golden_cases ya documentan
  honestamente 9 falsos positivos esperados de 11 casos normales (FPR=0.6798 reportado en la
  memoria para esta combinación concreta) — no son fallos del plugin.
- Alcance de datasets exploratorios descartados (`bearing_metro_preliminar/*`) y elección de
  arquitectura de producción (Audio-MAE sobre Baseline) — decisiones ya documentadas y
  confirmadas, nada pendiente aquí.

## Estado final

**REQUIERE ACCIÓN OPERATIVA ANTES DE PR** — el wiring de la API (routing, DTOs, excepciones,
`/train`) está correcto y el fallo actual se reporta de forma limpia y explícita, pero el
modelo **no es funcional en producción** por ausencia total de artefactos en S3. La Parte B no
pudo completarse por falta de datos de prueba locales. Antes de dar este plugin por
verificado de extremo a extremo hacen falta, en este orden:

1. Subir los 48 checkpoints a S3 (bloqueante — sin esto el modelo no sirve nada).
2. Recuperar `sample_data/` (o `inbox/a41/codigo/`) para poder ejecutar los 22 golden_cases
   contra el servicio real y confirmar que coinciden con los valores ya registrados en el
   manifest.
3. Revisión humana de si merece la pena extraer golden_cases para más combinaciones además de
   fan/id_00/0_dB, dado que `valve` tiene el rendimiento más débil del sistema.

_Este skill no abre PR ni hace merge — el gate humano es el último paso._
