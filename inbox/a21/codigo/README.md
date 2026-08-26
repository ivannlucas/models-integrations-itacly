# DATAGIA-21

Sistema de inferencia espacial para soporte de decisiones de compra/venta en cereal.
El proyecto combina prediccion de direccion (clasificacion) y magnitud (regresion)
en horizontes H1/H2/H3, con enfoque de robustez, trazabilidad y control de riesgo.


## 1. Estado actual del repositorio

Validado contra el codigo y artefactos presentes en el repo:

- Pipeline operativo de entrenamiento, evaluacion e inferencia.
- 6 modelos finales disponibles en `models/artifacts/`:
	- `datagia_best_h1_reg.joblib`
	- `datagia_best_h1_clf.joblib`
	- `datagia_best_h2_reg.joblib`
	- `datagia_best_h2_clf.joblib`
	- `datagia_best_h3_reg.joblib`
	- `datagia_best_h3_clf.joblib`
- Metadata de seleccion disponible en `models/artifacts/model_metadata.json`.
- Reporte de metricas consolidado en `models/metrics/full_report.json`.

Snapshot de datos generado actualmente:

- `data/processed/dataset_espacial_final.csv`: 7590 filas, 166 columnas, rango 2005-07 a 2026-07.
- `data/processed/dataset_entrenamiento_final.csv`: 7590 filas, 75 columnas.
- `data/predictions/`: rankings de arbitraje generados por `predict_v1.py --mode batch` (un CSV + un informe táctico por mes inferido; no se versiona un snapshot fijo, ver Sección 7 Paso 10).

Metricas test registradas en `models/metrics/full_report.json`:

- H1: AUC 0.7116 | DA(clf) 0.6172 | Pearson 0.3785 | MAE 0.0508
- H2: AUC 0.7441 | DA(clf) 0.6612 | Pearson 0.2653 | MAE 0.0674
- H3: AUC 0.7523 | DA(clf) 0.6261 | Pearson 0.2093 | MAE 0.1012

> Nota: estas métricas bajan unos puntos respecto a la versión del entregable
> actualmente en circulación, no por un cambio de modelo o de código sino
> porque el test set creció con ~90 filas nuevas (meses de 2026 antes sin
> target por falta de clima); el train es idéntico byte a byte (verificado
> celda a celda). Ver `models/metrics/reproducibility/` para el detalle. Por
> decisión explícita, el entregable (`documents/`) no se actualiza con estas
> cifras concretas — el cambio queda documentado aquí y en el registro de
> reproducibilidad.

> **Nota sobre el generador de ruido del target semi-sintético:** el ruido gaussiano
> que compone `precio_provincial_TARGET` se deriva de un hash estable de
> `(fecha, provincia, cereal)`, no de la posición del registro en el panel. Esto
> garantiza que añadir filas nuevas (p. ej. al refrescar datos MAPA) nunca cambie
> el ruido -ni el target- de una fila que ya existía: el conjunto de entrenamiento
> (fechas < 2021-01-01) es estable frente a futuras actualizaciones de datos. Ver
> `build_dataset._deterministic_row_noise` y `models/metrics/reproducibility/repro_run_2026-07-20.md`.

> **Nota sobre desfase MAPA (MAPA_ADMIN_LAG=3):** Los indices MAPA (IPPA/INDPAG)
> tienen un desfase administrativo de publicacion de 3 meses. Los features derivados
> de estas fuentes usan lag minimo 3 (dato disponible en el mes de inferencia).
> Para inferir en junio 2029 se necesitan datos MAPA hasta marzo 2029 como minimo.
> Ver `config/config.py` → `MAPA_ADMIN_LAG` y seccion 9 para el flujo de produccion.

## 2. Arquitectura funcional

El sistema esta organizado en 4 capas:

1) Data processing
- Construccion de panel espacial y dataset de entrenamiento.
- Split temporal estricto con frontera en 2021-01-01.
- Filtro anti-leakage (Blacklist) para evitar dependencia directa del precio pasado.

2) Training
- Busqueda por horizonte y tarea con `GridSearchCV` y `RandomizedSearchCV`.
- Modelos candidatos: Ridge, RandomForest, ExtraTrees, XGBoost, Logistic.
- Calibracion de clasificadores con `CalibratedClassifierCV`.

3) Evaluation
- Evaluacion train vs test con metricas espejo.
- Generacion de reportes numericos y graficos de diagnostico.

4) Inference
- Modo single: ficha de decision por provincia/cereal/mes.
- Modo batch: ranking de oportunidades + informe tactico.
- Umbrales de gobernanza en inferencia (ejemplo: `PROB_BULL=0.65`, `PROB_BEAR=0.35`).

## 3. Estructura del repositorio

```
DATAGIA-21/
	config/
		config.py
	data/
		raw/
		processed/
			dataset_espacial_final.csv
			dataset_entrenamiento_final.csv
			splits/
		predictions/
	models/
		artifacts/
		metrics/
	notebooks/
	documents/
	src/
		data_processing/
		training/
		get_stats/
		predict/
	requirements.txt
	README.md
```

## 4. Resumen del flujo reproducible

Este README contiene el flujo reproducible completo: desde la preparación del
entorno y la ingesta de datos hasta el entrenamiento, la evaluación y la
inferencia. Está diseñado para ser ejecutado desde la raíz del repositorio y
proporciona comandos explícitos, rutas de entrada/salida esperadas y
comprobaciones de integridad.

- Repositorio: estructura modular con `src/` para ETL, `src/training` para
  entrenamiento y `src/predict` para inferencia.
- Artefactos finales: `models/artifacts/*.joblib` y `models/artifacts/model_metadata.json`.
- Métricas: `models/metrics/full_report.json` y `models/metrics/summary_performance.md`.

## 5. Requisitos previos

- Python 3.10+ (3.11 recomendado).
- Entorno virtual creado en la raíz del repo.
- Dependencias instaladas: `pip install -r requirements.txt`.
- Datos de origen ubicados según inventario (`data/raw/` y `data/processed/manual/`).
- Credenciales GEE (opcional, solo si deseas regenerar la fase climática desde Earth Engine — ver Paso 1 para el alta del proyecto GCP).

## 6. Instalación y activación rápida

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **ADVERTENCIA SOBRE DATOS MANUALES**
> Los Pasos 2-5 requieren datos raw que NO están en el repositorio
> (archivos Excel del MAPA en `data/processed/manual/`). Si no los tienes,
> salta los Pasos 2 y 5 y ejecuta directamente desde el Paso 6: los archivos
> ya generados en `data/processed/auto/utils/` están versionados y son
> válidos, y el pipeline reproduce el dataset y los splits exactamente igual
> partiendo solo de ellos.
> Comportamiento sin datos raw, por script: `reshape_indpag4_timeseries` y
> `unify_superficies_by_year` detectan la ausencia de datos y se omiten de
> forma segura (mensaje informativo, sin escribir nada). `build_ppagados_unified`
> **sí sobrescribirá con datos vacíos** los ficheros de
> `data/processed/auto/ppagadostotal/` si se ejecuta sin los datos raw — no lo
> ejecutes si no dispones de ellos.
> Si sí tienes un boletín MAPA nuevo, no hace falta hacer el Paso 2 a mano:
> `python -m src.data_processing.ppagados.ingest_indices_precios_pagados "boletin.xlsx"`
> hace el parseo, la fusión y el staging en un solo comando (ver Paso 2).

> **MODO DE EJECUCIÓN SOPORTADO**
> Todos los comandos deben ejecutarse desde la **raíz del repositorio**
> usando `python -m <módulo>`. La ejecución directa de scripts
> (`python src/training/train.py`) NO está soportada: Python no añade
> la raíz al `sys.path` y los imports de `config` y `src` fallan.
> Forma correcta: `python -m src.training.train`

## 7. Flujo reproducible (orden estricto)

### Paso 0 — Preparación espacial (obligatorio antes de build)

```powershell
python -m src.data_processing.spatial.build_provincias_geometria
python -m src.data_processing.spatial.geocode_puertos
python -m src.data_processing.spatial.build_provincias_master
```

### Paso 1 — (Opcional) Ingesta GEE y agregado mensual

**Alta única del proyecto GCP** (solo la primera vez, o si `ee.Initialize()` falla
con `no project found`): Earth Engine exige un proyecto de Google Cloud
registrado.

1. Ir a https://code.earthengine.google.com/register con la cuenta de Google
   que se vaya a usar, elegir uso no comercial ("Unpaid usage" / "Academic &
   Research") y crear un proyecto nuevo.
2. Copiar el **Project ID** resultante (identificador técnico, no el nombre
   visible) en `config/config.yaml` → `gee_project_id`.
3. Autenticar la CLI con ese proyecto (abre el navegador; usar la MISMA
   cuenta que creó el proyecto):
   ```powershell
   .venv\Scripts\earthengine.exe --project=<tu-project-id> authenticate --force
   ```
   `--force` es necesario si ya había credenciales cacheadas de otra cuenta/proyecto
   (error típico: `Caller does not have required permission to use project ...`).

**Ingesta y fusión** (una vez autenticado): ajustar `weather.start_date` en
`config/config.yaml` al mes siguiente al último ya presente en
`climate_monthly_provinces.csv` (evita re-descargar histórico ya procesado;
`end_date` por defecto es hoy), y ejecutar:

```powershell
python -m src.data_processing.ingestion.ingest_weather_gee
python -m src.data_processing.climate.merge_recent_climate
```

`ingest_weather_gee.py` descarga de ERA5-Land (diario) y escribe, entre otros,
`data/external/clima_provincias_GEE_daily.csv`. `merge_recent_climate.py` agrega
ese fichero a mensual por mes de calendario real y fusiona (upsert por
año/mes/provincia) en `data/processed/auto/utils/climate_monthly_provinces.csv`
sin pisar el histórico previo. No usar `aggregate_gee_provinces_monthly.py` con
la salida de la ingesta actual: ese script espera un formato diario antiguo
(`date_ms`/`province_name`/`temp_k`/`precip_m`) que la ingesta ya no produce;
se mantiene solo por compatibilidad con los CSV históricos de
`data/processed/manual/climate_provinces_GEE_by_year/`.

> ⚠️ ERA5-Land tiene un desfase de publicación propio de Google (típicamente
> ~1 semana): al pedir datos "hasta hoy" es normal que el último mes quede
> incompleto (ver aviso de features parciales en el log de `ingest_weather_gee`).

Salida: `data/processed/auto/utils/climate_monthly_provinces.csv`.

### Paso 2 — Normalización y unificación de precios MAPA (ppagados)

**Para actualizar con un boletín MAPA nuevo** ("Índices y Precios Pagados
Agrarios", Excel con hojas PrePag1/PrePag2/IndPag3/IndPag4): un solo comando
convierte el Excel, lo fusiona con el histórico y deja `data/processed/auto/utils/`
listo para el Paso 7, sin pasos manuales intermedios:

```powershell
python -m src.data_processing.ppagados.ingest_indices_precios_pagados "ruta/al/boletin.xlsx"
```

Usa `--no-apply` si solo quieres generar los CSV en `data/processed/manual/ppagados4/`
sin fusionarlos todavía (por ejemplo, para revisarlos antes de aplicarlos).

**Histórico heredado** (tiers `ppagados1`/`ppagados1_2`/`ppagados2`/`ppagados3`,
ya incluidos en el repositorio, normalmente no hace falta regenerarlos): el
comando anterior es un wrapper de este flujo de más bajo nivel, útil si se
necesita reprocesar manualmente uno de esos tiers:

```powershell
python -m src.data_processing.ppagados.reshape_indpag4_timeseries
python -m src.data_processing.ppagados.fix_ppagados_date_columns
python -m src.data_processing.ppagados.build_ppagados_unified
```

Salida: `data/processed/auto/ppagadostotal/` → copiar a `data/processed/auto/utils/`
(el comando de un solo paso ya hace esta copia automáticamente).

> ⚠️ Dentro de `ppagados1_2`, `fix_ppagados_date_columns.py` solo debe
> aplicarse a ficheros que ya estén en formato fila-por-fecha. Los 4 CSV en
> formato de bloques (IndPag3/IndPag4/PrePag1/PrePag2) deben regenerarse con
> `reshape_indpag4_timeseries.py <fichero>.csv` uno a uno; aplicarles
> `fix_ppagados_date_columns.py` los corrompe silenciosamente (no falla, pero
> produce un CSV con fechas vacías).

### Paso 3 — Ingesta EUR/USD semanal

```powershell
python -m src.data_processing.ingestion.ingest_eurusd_weekly
```

Salida: `data/processed/auto/utils/eur_usd_weekly.csv`.

### Paso 4 — Ingesta mercados internacionales (MATIF/FAO)

```powershell
python -m src.data_processing.ingestion.ingest_markets_matif_fao
Copy-Item data\\raw\\mercados_internacionales.csv data\\processed\\auto\\utils\\mercados_internacionales.csv -Force
```

Nota: el script implementa degradaciones (fallback a `ZW=F`); la descarga FAO puede quedar vacía.

### Paso 5 — Unificación de superficies provinciales

```powershell
python -m src.data_processing.superficies.unify_superficies_by_year
# Copiar los total_*.csv resultantes a:
# data/processed/auto/utils/superficies_provinciales/
```

### Paso 6 — Consolidar staging en `data/processed/auto/utils/`

```powershell
# Omitir la primera copia si se actualizó el MAPA con
# ingest_indices_precios_pagados.py (Paso 2): ya deja esto hecho.
Copy-Item data\\processed\\auto\\ppagadostotal\\*.csv data\\processed\\auto\\utils -Force
Copy-Item data\\processed\\auto\\spatial\\provincias_geometria.csv data\\processed\\auto\\utils -Force
Copy-Item data\\raw\\mercados_internacionales.csv data\\processed\\auto\\utils\\mercados_internacionales.csv -Force
```

### Paso 7 — Construcción final del dataset y splits

```powershell
python -m src.data_processing.build_dataset
python -m src.data_processing.prepare_data --save-splits
```

Salidas esperadas:

- `data/processed/dataset_espacial_final.csv`
- `data/processed/dataset_entrenamiento_final.csv`
- `data/processed/splits/train_h*_*.csv`
- `data/processed/splits/test_h*_*.csv`

Comprobación (sugerida):

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/dataset_espacial_final.csv'); assert df.duplicated(subset=['date','provincia','cereal_predominante']).sum()==0; print('OK: sin duplicados')"
```

### Paso 8 — Entrenamiento

```powershell
python -m src.training.train
```

Salida: `models/artifacts/*.joblib` y `models/artifacts/model_metadata.json`.

### Paso 9 — Generar métricas y reporte

```powershell
python -m src.get_stats.get_stats
```

Salida: `models/metrics/full_report.json`, `models/metrics/summary_performance.md`, y gráficos.

Comprobación de sincronía: verificar que `evidence_bundle_id` en `model_metadata.json` y `full_report.json` coincide.

### Paso 10 — Inferencia (smoke test)

```powershell
python -m src.predict.predict_v1 --mode batch --role comprador --month 2025-06
```

Salida: `data/predictions/ranking_oportunidades_arbitraje_2025-06.csv` y `informe_tactico_2025-06.txt`.

### Paso 11 — Captura de logs y checksums (recomendado para auditoría)

```powershell
mkdir logs -Force
python -m src.training.train > logs/training_$(Get-Date -Format 'yyyyMMdd_HHmmss').log 2>&1
Get-FileHash models\\artifacts\\datagia_best_h1_reg.joblib -Algorithm SHA256 | Format-List
```

## 8. Qué versionar en Git

- **Sí:** cambios de código (`src/**`), notebooks y metadatos auditables en JSON (verificar que no contienen rutas absolutas antes de commitear).
- **No:** artefactos binarios pesados (`models/artifacts/*.joblib`), outputs regenerables (`data/processed/auto/*`, `data/predictions/*`, `models/metrics/*.png`).

## 9. Buenas prácticas operativas

- No permitir que `predict` sobrescriba `model_metadata.json` durante inferencia.
- Evitar imputaciones bidireccionales que introduzcan leakage temporal; documentar la política de imputación.
- Mantener un archivo `models/metrics/reproducibility/repro_run_<fecha>.md` con: commit hash, comandos ejecutados, checksums y notas de degradaciones.

### Actualización periódica de datos (producción)

El panel solo puede llegar hasta la fecha del dato **más antiguo** entre todas las
fuentes que se combinan (MAPA precios/costes, clima, mercados, EUR/USD). En la
práctica, el MAPA (precios percibidos y costes de insumos) es casi siempre el
cuello de botella, porque se publica con retraso administrativo y no tiene API:
hay que descargar manualmente los boletines Excel. El resto de fuentes sí tienen
script de ingesta automática.

**Paso 1 — Actualizar datos raw, por fuente:**

| Fuente | Cómo se actualiza | Automatizable |
|---|---|---|
| Precios/costes MAPA (boletín "Índices y Precios Pagados Agrarios") | Descargar el boletín más reciente de la web del MAPA a `data/raw/` y ejecutar `python -m src.data_processing.ppagados.ingest_indices_precios_pagados "ruta/al/boletin.xlsx"` — un solo comando parsea el Excel, lo fusiona con el histórico y actualiza `data/processed/auto/utils/` | Solo la descarga es manual (no hay API pública); el resto (parseo + fusión + staging) es un único comando |
| Clima (`climate_monthly_provinces.csv`) | `python -m src.data_processing.ingestion.ingest_weather_gee` + `merge_recent_climate` (ver Paso 1) | Sí, requiere credenciales GEE (`gee_project_id` en `config/config.yaml`, alta única de proyecto GCP gratuito) |
| Mercados internacionales (MATIF/FAO) | `python -m src.data_processing.ingestion.ingest_markets_matif_fao` | Sí (usa Yahoo Finance; degrada con avisos si alguna serie no está disponible) |
| EUR/USD | `python -m src.data_processing.ingestion.ingest_eurusd_weekly` | Sí |

**Paso 2 — Reconstruir panel y ejecutar inferencia:**

```powershell
python -m src.data_processing.build_dataset

# El motor detecta automaticamente si faltan targets y activa el modo produccion.
python -m src.predict.predict_v1 --mode batch --role comprador --month 2029-06
```

El log confirmara el desfase activo:
`Inferencia batch 2029-06 — datos MAPA disponibles hasta 2029-03 (desfase 3 meses)`

Si el mes esta completamente fuera del panel, el sistema lanza un error accionable
que ahora incluye la **fecha del ultimo dato real disponible por cada fuente**
(precios/costes MAPA, clima, mercados, EUR/USD), para identificar de un vistazo
cual es la fuente que limita el panel sin tener que inspeccionar cada CSV a mano:

```
ValueError: Sin datos de panel para 2026-09. Actualiza datos crudos y ejecuta build_dataset.
Cobertura actual por fuente (ultimo dato real disponible, no placeholder):
  - Precios agricultores MAPA (prepag2_total): hasta 2026-03
  - Costes insumos MAPA (prepag1_total): hasta 2026-03
  - Indices MAPA (Indpag1_total): hasta 2026-03
  - Indices MAPA (Indpag2_total): hasta 2026-03
  - Clima GEE (climate_monthly_provinces): hasta 2026-07
  - Mercados intl. (mercados_internacionales): hasta 2026-07
  - EUR/USD (eur_usd_weekly): hasta 2026-07
La fuente con la fecha mas antigua es la que limita el panel; actualiza esa fuente en data/raw/ y reejecuta build_dataset.
```

## 10. Contacto y seguimiento

Si surge cualquier inconsistencia entre la documentación y la ejecución, abrir un issue con: pasos ejecutados, salida de `git rev-parse --short HEAD`, logs y fragmentos de error.
