# Verificación — ml41-meat-curing-machinery-acoustic-anomaly (a41)

**Fecha:** 2026-10-02
**Plugin:** `app/plugins/ml41_meat_curing_machinery_acoustic_anomaly/`
**Manifest:** `inbox/a41/manifest.yaml` (22 golden_cases, combo fan/id_00/0_dB)
**Motivo de esta verificación:** el plugin ya estaba integrado (manifest + golden_cases
completos) pero nunca se había generado este informe — gap detectado en una auditoría
repo-wide del estado de `inbox/`/`outputs/` de los 25 plugins registrados.

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
