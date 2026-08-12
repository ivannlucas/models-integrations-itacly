# Verificación — a03

Modelo: **ml3_wine_disease_pest_forecast** (Wine Disease & Pest Forecast, Deep Ensemble LSTM + CNN-1D + BiGRU)
Plugin: `app/plugins/ml3_wine_disease_pest_forecast/`
Manifest: `inbox/a03/manifest.yaml` (20 golden_cases)

Fecha: 2026-08-04 · Entorno: Python 3.10.12, numpy 1.26.4, TensorFlow 2.17.0 / Keras 3.12.4

## Checklist técnico

- [x] **flake8**: 0 errores (`flake8 app/ tests/`, superficie commiteada)
- [x] **pytest**: 339/339 passed, cobertura global 30% (`pytest tests/unit/ --cov=app`)
- [x] **pylint**: 10.00/10 sin issues nuevos (`pylint app/plugins/ml3_wine_disease_pest_forecast/ --disable=import-error`)
- [x] **pip-audit**: sin CVEs nuevas introducidas por el plugin (ver nota abajo)
- [x] **Arranque local + health + predict + stats**: OK
- [x] **/train**: 200 con `TrainResponse` completo (`training.supported: true`)

### Notas pip-audit

- `pip-audit -r requirements.txt` no se puede ejecutar en este entorno: el intérprete del
  proyecto es el Python del sistema (`/usr/bin/python3` 3.10.12) sin `ensurepip`/`venv`
  (`python3.10-venv` no instalado), y pip-audit necesita un venv temporal para resolver.
- Alternativa ejecutada: `pip-audit` (bare) sobre el entorno vivo. Resultado:
  - Sin CVEs en las dependencias propias del plugin (no añade dependencias nuevas al repo).
  - CVEs en paquetes del entorno de sistema (apt): `setuptools 59.6.0`, `wheel 0.37.1`,
    `zipp 1.0.0`, `command-not-found`, `distro-info`, `python-apt`, `ufw`, etc. — fuera del
    alcance del plugin.
  - Dos hallazgos en runtime compartido por los 48 modelos, pendientes de decisión humana
    (NO bloqueantes para este PR): `torch 2.11.0+cu130` (PYSEC-2025-194 → 2.13.0, tirado por
    `torchvision==0.26.0`) y `starlette 1.3.0` (PYSEC-2026-249 → 1.3.1). Cambiarlos afecta a
    todo el catálogo, no solo a ml3.

## Correctitud (golden dataset)

Tolerancia usada: **clase exacta** y **rtol 0.01** en `confianza_clasificacion` y
`grado_severidad`, derivada de `metrics_reported` (MAE de regresión 0.061 → rtol 1%)
— `tolerance: {clase_exact: true, float_rtol: 0.01}` del propio manifest.

| Caso | ID_Serie | Esperado (clase / conf / sev) | Obtenido (clase / conf / sev) | Diferencia | ¿OK? |
|---|---|---|---|---|---|
| caso_001 | 552 | ALTICA / 0.960752 / 0.758976 | ALTICA / 0.960752 / 0.758976 | 0 / 0 / 0 | ✅ |
| caso_002 | 862 | BLACK_ROT / 0.757278 / 0.720613 | BLACK_ROT / 0.757278 / 0.720613 | 0 / 0 / 0 | ✅ |
| caso_003 | 450 | MILDIU / 0.33255 / 0.301255 | MILDIU / 0.332550 / 0.301255 | 0 / 0 / 0 | ✅ |
| caso_004 | 100 | OIDIO / 0.568258 / 0.780486 | OIDIO / 0.568257 / 0.780486 | 0 / 1e-6 / 0 | ✅ |
| caso_005 | 671 | HEALTHY / 0.422017 / 0.091516 | HEALTHY / 0.422017 / 0.091516 | 0 / 0 / 0 | ✅ |
| caso_006 | 250 | ESCA / 0.400751 / 0.494715 | ESCA / 0.400751 / 0.494715 | 0 / 0 / 0 | ✅ |
| caso_007 | 580 | HEALTHY / 0.997361 / 0.049236 | HEALTHY / 0.997361 / 0.049236 | 0 / 0 / 0 | ✅ |
| caso_008 | 801 | BLACK_ROT / 0.782021 / 0.807961 | BLACK_ROT / 0.782021 / 0.807961 | 0 / 0 / 0 | ✅ |
| caso_009 | 545 | MILDIU / 0.756275 / 0.629311 | MILDIU / 0.756275 / 0.629311 | 0 / 0 / 0 | ✅ |
| caso_010 | 517 | OIDIO / 0.604654 / 0.814065 | OIDIO / 0.604654 / 0.814065 | 0 / 0 / 0 | ✅ |
| caso_011 | 402 | EMPOASCA / 0.710688 / 0.573809 | EMPOASCA / 0.710688 / 0.573809 | 0 / 0 / 0 | ✅ |
| caso_012 | 207 | ALTICA / 0.936209 / 0.766861 | ALTICA / 0.936209 / 0.766861 | 0 / 0 / 0 | ✅ |
| caso_013 | 364 | BLACK_ROT / 0.477046 / 0.59883 | BLACK_ROT / 0.477046 / 0.598830 | 0 / 0 / 0 | ✅ |
| caso_014 | 490 | BOTRYTIS / 0.798503 / 0.409982 | BOTRYTIS / 0.798503 / 0.409982 | 0 / 0 / 0 | ✅ |
| caso_015 | 899 | ALTICA / 0.351673 / 0.813663 | ALTICA / 0.351673 / 0.813663 | 0 / 0 / 0 | ✅ |
| caso_016 | 106 | MILDIU / 0.302824 / 0.137262 | MILDIU / 0.302824 / 0.137262 | 0 / 0 / 0 | ✅ |
| caso_017 | 782 | MILDIU / 0.322408 / 0.525987 | MILDIU / 0.322408 / 0.525987 | 0 / 0 / 0 | ✅ |
| caso_018 | 66 | HEALTHY / 0.925176 / 0.15889 | HEALTHY / 0.925176 / 0.158890 | 0 / 0 / 0 | ✅ |
| caso_019 | 376 | OIDIO / 0.561803 / 0.825139 | OIDIO / 0.561803 / 0.825139 | 0 / 0 / 0 | ✅ |
| caso_020 | 524 | HEALTHY / 0.321775 / 0.137297 | HEALTHY / 0.321775 / 0.137297 | 0 / 0 / 0 | ✅ |

**Resultado: 20/20 casos dentro de tolerancia** — clase exacta en todos y floats a
≤ 1e-6 de diferencia (muy por debajo del rtol 0.01).

### Hallazgo documentado (no es fallo del plugin)

- Los `expected` del manifest se obtuvieron ejecutando el **código entregado original**
  leyendo el parquet nativo (`data/raw/data_vin_raw.parquet`). El plugin reproduce la salida
  **bit-a-exacta** (o a 1e-6) cuando la ventana se lee desde el parquet.
- Vía `pd.read_csv` (round-trip CSV), `confianza_clasificacion` y `grado_severidad` pueden
  desviarse en el 5º-6º decimal por la conversión float32→float64 (TF → `pd.read_csv`).
  Es el mismo comportamiento del `run_inference` entregado y queda dentro de la tolerancia.
  No se ajustó ninguna tolerancia para que pasara.

## Estado final

**LISTO PARA PR** — plugin integrado y verificado en verde:

- Parte A completa: flake8, pytest (339/339), pylint 10/10, arranque real con
  `MODEL=ml3-wine-disease-pest-forecast`, `/health`, `/predict` (inline/batch),
  `/stats` y `/train` (200 con métricas) verificados contra la app viva en localhost:8000.
- Parte B completa: 20/20 golden_cases dentro de tolerancia contra el parquet nativo.
- Pendientes para revisión humana (no bloqueantes): upgrade de `torch`/`starlette` en el
  requirements compartido y decisión sobre el entorno de sistema sin `ensurepip`.
