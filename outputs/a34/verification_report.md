# Verificación — a34 (ml34-dairy-pasteurization-energy-ga)

**Fecha:** 2026-07-10
**Plugin:** `app/plugins/ml34_dairy_pasteurization_energy_ga/`
**Manifest:** `inbox/a34/manifest.yaml` (20 golden cases: 15 predict + 5 optimize)
**Rama:** `feature/model-34-integration`

## Checklist técnico

- [x] **flake8**: 0 errores (repo completo, excluyendo `inbox/` y `outputs/`).
  Nota: se corrigió un F401 preexistente en `tests/conftest.py` (import
  `Ml35DairyOptimizeResp` sin uso, ya presente en HEAD).
- [x] **pytest**: 298/298 passed (`tests/unit/`, incluye 10 tests nuevos de ml34 con
  FakePlugin). Cobertura de los DTOs de ml34: 100%; `plugin.py` se cubre con la
  verificación de correctitud (Parte B), igual que el resto de plugins del repo
  (los unit tests validan wiring, no correctitud).
  Nota de entorno: los tests requieren `-p no:flask` en esta máquina porque el
  plugin global `pytest-flask` secuestra el fixture `client`; no es un problema
  del repo.
- [x] **pylint** (`app/plugins/ml34_dairy_pasteurization_energy_ga/`): **9.93/10**
  (baseline del plugin piloto ml35: 8.89/10). Los disables aplicados son
  exclusivamente por nombres de dominio (`T_in_leche`, `scaler_X`… que replican
  las columnas del dataset y el código original) y falsos positivos de miembros
  dinámicos de DEAP.
- [x] **pip-audit**: **0 CVEs introducidas por esta integración** (`deap==1.4.3`
  sin vulnerabilidades conocidas). Preexistentes en el repo, pendientes de decisión
  humana (no las introduce ml34):
  - `torch 2.11.0` — CVE-2025-3000 (sin versión de fix publicada)
  - `keras 3.12.3` — PYSEC-2026-73 (fix: 3.13.1)
- [x] **Arranque local + endpoints**: OK.
  - `GET /health` → `{"status":"ok","loaded":true}`
  - `POST /predict` (inline) → 200, reproduce el golden case exactamente
  - `POST /predict` (optimize, seed=1) → 200, reproduce el backtesting del equipo de IA
  - `GET /stats` → 200 con métricas del manifest
  - Nota de entorno: `main.py` completo no arranca en esta máquina porque otros
    plugins del registry requieren dependencias no instaladas localmente
    (`timm`, `detectron2` — build desde fuente —, `tensorflow`, `pytorchvideo`…).
    Se verificó con un launcher que replica el wiring exacto de `main.py`
    (mismo `ModelContainer` + `make_model_router` + `extra_predict_exceptions`)
    con solo la entrada ml34. En el pipeline/producción, donde esas dependencias
    existen, `MODEL=ml34-dairy-pasteurization-energy-ga python main.py` es equivalente.
- [x] **/train**: 200 con `TrainResponse` completo (manifest declara
  `training.supported: true`). Fine-tuning real sobre 1 000 filas de
  `train.csv`: early stopping en época 24 (~1.3 s), métricas devueltas
  coherentes con el entrenamiento original:
  `rmse_E_consumo=5.3284`, `mae_E_consumo=4.2131`, `r2_E_consumo=0.9756`,
  `rmse_T_out_leche=0.0617`, `mae_T_out_leche=0.0474`, `r2_T_out_leche=0.4248`.
  El artefacto fijo se restauró tras la prueba (md5 idéntico al entregable) y se
  reverificó un golden case después de la restauración.

## Correctitud (golden dataset)

Los 20 casos del manifest se lanzaron contra el endpoint HTTP real (servidor uvicorn
local, artefactos entregados). `expected` = salida del pipeline original del equipo
de IA (`predictions.csv` / `evaluation_rt_hist_vs_ia.csv`, alineados fila a fila con
`data/splits/test.csv`).

### Modo predict (15 casos) — tolerancia: E_consumo ±2% (2× MAE relativo ≈1%), T_out ±0.1 °C (≈1.5× RMSE)

| Caso | E_consumo esperado (kW) | Obtenido | Dif | T_out esperado (°C) | Obtenido | Dif | ¿OK? |
|---|---|---|---|---|---|---|---|
| predict_001 | 313.5495 | 313.5495 | 0.0000 | 72.9252 | 72.9252 | 0.0 | ✅ |
| predict_002 | 384.9313 | 384.9313 | 0.0000 | 72.6017 | 72.6017 | 0.0 | ✅ |
| predict_003 | 397.8086 | 397.8085 | 0.0001 | 72.6107 | 72.6107 | 0.0 | ✅ |
| predict_004 | 406.9790 | 406.9790 | 0.0000 | 72.5966 | 72.5966 | 0.0 | ✅ |
| predict_005 | 414.4876 | 414.4876 | 0.0000 | 72.5922 | 72.5922 | 0.0 | ✅ |
| predict_006 | 421.1544 | 421.1544 | 0.0000 | 72.5967 | 72.5967 | 0.0 | ✅ |
| predict_007 | 427.5305 | 427.5305 | 0.0000 | 72.5962 | 72.5962 | 0.0 | ✅ |
| predict_008 | 433.4866 | 433.4866 | 0.0000 | 72.5977 | 72.5977 | 0.0 | ✅ |
| predict_009 | 439.9113 | 439.9113 | 0.0000 | 72.5944 | 72.5944 | 0.0 | ✅ |
| predict_010 | 446.9276 | 446.9276 | 0.0000 | 72.5958 | 72.5958 | 0.0 | ✅ |
| predict_011 | 454.0950 | 454.0950 | 0.0000 | 72.5931 | 72.5931 | 0.0 | ✅ |
| predict_012 | 464.0237 | 464.0237 | 0.0000 | 72.5941 | 72.5941 | 0.0 | ✅ |
| predict_013 | 475.4329 | 475.4329 | 0.0000 | 72.5932 | 72.5932 | 0.0 | ✅ |
| predict_014 | 490.3388 | 490.3387 | 0.0001 | 72.5939 | 72.5939 | 0.0 | ✅ |
| predict_015 | 536.8580 | 536.8580 | 0.0000 | 72.6035 | 72.6035 | 0.0 | ✅ |

### Modo optimize (5 casos, GA determinista con seed registrado) — tolerancias del manifest

Los 5 casos reproducen **bit a bit** todos los campos del backtesting del equipo de
IA (setpoints, consumo, temperatura, consumo específico y fitness), pese a ejecutarse
con torch 2.1.2 CPU local frente a torch 2.5.1+cu121 del entregable:

| Caso | seed | IA_F_flow | IA_T_servicio | IA_E_consumo | IA_T_out | IA_consumo_especifico | ¿OK? |
|---|---|---|---|---|---|---|---|
| optimize_001 | 1 | 5422.10 = exp | 80.40 = exp | 412.9016 = exp | 72.30 = exp | 0.076152 = exp | ✅ |
| optimize_002 | 1928 | 5476.12 = exp | 82.30 = exp | 428.0721 = exp | 72.30 = exp | 0.078171 = exp | ✅ |
| optimize_003 | 3855 | 5362.54 = exp | 82.23 = exp | 439.8397 = exp | 72.30 = exp | 0.082021 = exp | ✅ |
| optimize_004 | 5782 | 5351.83 = exp | 81.64 = exp | 430.3456 = exp | 72.30 = exp | 0.080411 = exp | ✅ |
| optimize_005 | 7708 | 5395.19 = exp | 80.97 = exp | 426.5447 = exp | 72.30 = exp | 0.079060 = exp | ✅ |

Latencia del GA: ~0.4 s/escenario en esta máquina (el entregable reporta 0.42–0.72 s
según hardware).

**Tolerancia usada:** la declarada por caso en el manifest, derivada de
`metrics_reported` (E_consumo: 2% ≈ 2× MAE relativo; T_out: 0.1 °C ≈ 1.5× RMSE;
optimize: 2% en objetivo y consumo, 0.1 °C en T_out). En la práctica la
reproducción fue exacta y ninguna tolerancia se ajustó.

**Resultado: 20/20 casos dentro de tolerancia.**

## Hallazgos documentados (no bloqueantes, requieren conocimiento del revisor)

1. **Discrepancia memoria ↔ artefacto en KPIs del GA** — la memoria v2.0 (Tabla 10)
   y el README citan mejora del consumo específico **10.73%** (run antiguo), pero el
   `evaluation_rt_backtesting_report.json` entregado reporta **11.73%**. Decisión del
   propietario (2026-07-10): el artefacto entregado es la fuente de verdad; ambas
   cifras aparecen en la ficha técnica con nota. Las métricas del MLP coinciden
   exactamente en todas las fuentes.
2. **Datos 100% sintéticos** — todas las cifras de negocio (mejora 11.73%,
   cumplimiento 100%) provienen del simulador físico V3.4; validación Sim2Real con
   datos reales de planta pendiente (declarado también por el equipo de IA).
3. **`pygad==3.5.0` en requirements.txt del entregable no se usa** (el código
   operativo usa `deap`); parece residuo de otro entregable.
4. **`pygad` tampoco está en el requirements.txt del repo** pese a que ml35 lo
   importa en runtime — gap preexistente, fuera del alcance de esta integración,
   señalado para revisión.
5. **Determinismo del GA** — la reproducción bit a bit está verificada con
   `deap 1.4.3` + Python 3.10; cambios de versión de deap podrían alterar los
   setpoints exactos (el objetivo y la factibilidad seguirían siendo verificables
   por tolerancia).

## Estado final

**LISTO PARA PR** (gate humano pendiente: revisar los hallazgos 1–4 y decidir sobre
las CVEs preexistentes de torch/keras).
