---
name: odd-integration
description: Usa este skill para conectar un plugin ya integrado y verificado de inference-pan-model con el servicio de detección de drift/ODD (repo retech-lote2-xai-odd-detection). A diferencia del front y de explicabilidad, aquí NO se toca código Python — es un registro puramente declarativo en models_config.json; el baseline se genera solo al arrancar el servicio. Requiere el plugin ya integrado (plugin-integration) y verificado en verde (verification) en este repo.
---

# Integración de un plugin en el servicio de detección de drift (ODD)

## Placeholders de este skill

- `{{ODD_REPO_PATH}}` — ruta local del repo de detección de drift (`retech-lote2-xai-odd-detection`).
- `{{PLUGIN_NAME}}` — `model_id`/`model_key` del plugin ya integrado en `app/plugins/<nombre>/` de
  este repo (inference-pan-model), el que se va a monitorizar.

Si al invocar este skill no se han informado ambos placeholders, para y pregunta antes de tocar
nada — nunca asumas una ruta del servicio de ODD por defecto ni "adivines" qué plugin integrar.

## Requisito previo

El plugin debe estar integrado (skill `plugin-integration`) y verificado en verde (skill
`verification`) en **este** repo. Si `outputs/<model_id>/verification_report.md` no existe o no
dice "LISTO PARA PR", para y avisa.

## Cómo es esta integración (léelo antes de tocar nada)

A diferencia del front y de explicabilidad, este servicio está diseñado para onboarding **sin
cambios de código**. Todo el registro vive en un único fichero:
`{{ODD_REPO_PATH}}/models_config.json`. `main.py`, `detector.py`, `preprocessors.py`,
`alert_publisher.py`, `baseline_loader.py`, `baseline_bootstrap.py`,
`generate_baseline.py` y `s3_baseline.py` son completamente genéricos y dirigidos por esa
configuración — no se editan para dar de alta un modelo nuevo. Si al integrar `{{PLUGIN_NAME}}`
te encuentras necesitando tocar alguno de esos ficheros, para: probablemente el modelo tiene un
tipo de dato (`multimodal`, o un caso no cubierto por `tabular`/`image`/`video`) que requiere
diseño nuevo, no una integración mecánica — confírmalo con el usuario antes de escribir código
Python en este repo.

El baseline (distribución de referencia) **no se genera a mano**: al arrancar el servicio,
`bootstrap_baselines()` recorre cada entrada de `models_config.json`, comprueba si ya existe
`baselines/<name>/baseline.json` en S3 y, si no, descarga los datos de entrenamiento configurados,
calcula el baseline y lo sube — todo automático. El único trabajo manual es: subir los datos de
entrenamiento a S3 y añadir la entrada de configuración.

## Paso 0 — Encuentra el modelo hermano más parecido ya integrado

Abre `{{ODD_REPO_PATH}}/models_config.json` y busca 1-2 entradas ya integradas del mismo tipo que
`{{PLUGIN_NAME}}` (tabular vs imagen/vídeo, con o sin columnas categóricas) para copiar su
estructura exacta.

## Paso 1 — Sube los datos de entrenamiento a S3

Fuera de este repo, sube a S3 (bucket `S3_BASELINE_BUCKET`, mismo bucket que usa
`baseline_bootstrap.py`):

- **Tabular**: el CSV de entrenamiento de `{{PLUGIN_NAME}}` en
  `datasets/{{PLUGIN_NAME}}/train.csv` — el mismo split/fuente que ya se usó como golden dataset
  en la skill `verification` de este repo, nunca datos sintéticos nuevos.
- **Imagen/vídeo**: las imágenes/vídeos de entrenamiento sueltas bajo `datasets/{{PLUGIN_NAME}}/`.

No crees nada bajo la carpeta local `datasets/` de `{{ODD_REPO_PATH}}` — esa carpeta es efímera,
la crea y borra el propio proceso de bootstrap en cada arranque; los datos reales viven solo en
S3.

## Paso 2 — Añade la entrada en `models_config.json`

Añade un objeto nuevo al array `"models"`. Campos según el tipo:

**Tabular** (copiar de un modelo hermano tabular):
```json
{
  "name": "{{PLUGIN_NAME}}",
  "_label": "<descripción legible, cosmética>",
  "type": "tabular",
  "s3_dataset_key": "datasets/{{PLUGIN_NAME}}/train.csv",
  "csv_sep": ",",
  "feature_cols": ["<col1>", "<col2>", "..."],
  "categorical_cols": ["<subconjunto de feature_cols con dtype no numérico>"],
  "max_samples": 5000
}
```

**Imagen/vídeo** (copiar de un modelo hermano de imagen):
```json
{
  "name": "{{PLUGIN_NAME}}",
  "_label": "<descripción legible, cosmética>",
  "type": "image",
  "s3_dataset_prefix": "datasets/{{PLUGIN_NAME}}/",
  "extractor": "clip"
}
```

Reglas de este fichero, verificadas contra las 16 entradas existentes — no son sugerencias:

- `"name"` es la clave de unión de todo el sistema: debe ser **idéntica byte a byte** al
  `model_key`/`model_id` que el front (`retech-lote2-xai-plataforma`) envía como segmento de ruta
  `{model_name}` en `/detect/{model_name}`. No hay ninguna tabla de mapeo aparte — si no coincide,
  el endpoint de detección da 404 aunque el baseline exista.
- `"type"` tiene que ser exactamente `"tabular"` o `"image"` (`"video"`/`"multimodal"` existen en
  código pero ningún modelo real los usa hoy) — cualquier otro valor hace fallar el bootstrap con
  `"unknown type"`.
- `feature_cols`/`categorical_cols`: saca los nombres reales del `predict_dto.py`/`manifest.yaml`
  de `app/plugins/{{PLUGIN_NAME}}/` en este repo, nunca inventados. `categorical_cols` es siempre
  subconjunto de `feature_cols`, nunca una lista aparte.
- `"max_samples": 5000` es la convención sin excepción en las 10 entradas tabulares actuales — no
  te desvíes sin motivo real.
- `"extractor": "clip"` es la convención sin excepción en las 6 entradas de imagen actuales.
- `"_label"` es puramente decorativo (no lo lee ningún código) — no dediques tiempo a pulirlo más
  allá de una frase clara.

## Paso 3 — Bump de `VERSION`

Incrementa el fichero `VERSION` (una línea, ej. `1.0.15` → `1.0.16`) — así lo hacen los dos
commits reales de alta de modelo en este repo, y alimenta el tag de la imagen Docker en
`bitbucket-pipelines.yml`.

## Paso 4 — Despliegue y generación automática del baseline

No hace falta ejecutar ningún script a mano. Al desplegar/reiniciar el servicio,
`bootstrap_baselines()` detecta que `{{PLUGIN_NAME}}` no tiene baseline en
`s3://.../baselines/{{PLUGIN_NAME}}/baseline.json`, descarga los datos del paso 1, calcula el
baseline y lo sube. Alternativa sin reinicio: `POST /baseline/{{PLUGIN_NAME}}/from-source` con el
`source_id` correspondiente, para generar (o regenerar) el baseline bajo demanda.

**Antes de dar el paso 3-4 por bueno**, verifica que `S3_BASELINE_BUCKET` (usado por
`baseline_bootstrap.py`) y `OOD_BASELINE_BUCKET`/`DRIFT_BASELINE_BUCKET` +
`CUSTOM_S3_ENDPOINT`/`CUSTOM_REGION` (usados por `s3_baseline.py`, que sirve las peticiones de
`/detect`) apuntan al **mismo** bucket/endpoint en el entorno de destino — son dos pares de
variables de entorno distintos para lo que debería ser el mismo almacén; si divergen, el baseline
generado en el bootstrap queda invisible para `/detect/{{PLUGIN_NAME}}` aunque el registro esté
bien.

## Paso 5 — Verificación

Contra la URL donde esté corriendo el servicio de ODD (local o desplegado):

```bash
curl <url-servicio-odd>/baselines            # debe listar {{PLUGIN_NAME}} tras el primer /detect que lo cachee
curl -X POST <url-servicio-odd>/detect/{{PLUGIN_NAME}} -H "Content-Type: application/json" -d '<fila de ejemplo con feature_cols>'
```
o comprobar directamente en S3 que existe `baselines/{{PLUGIN_NAME}}/baseline.json`.

## Reglas que nunca se saltan

- Nunca se toca código Python (`main.py`, `detector.py`, `preprocessors.py`,
  `baseline_bootstrap.py`, etc.) para dar de alta un modelo — si parece necesario, para y
  pregunta; es señal de que el modelo no encaja en `tabular`/`image` tal cual están definidos.
- Nunca se genera o sube a mano un `baseline.json` — siempre lo genera `bootstrap_baselines()` (o
  el endpoint `/baseline/.../from-source`) a partir de los datos reales de entrenamiento.
- Nunca se inventan `feature_cols`/`categorical_cols` — salen del contrato real del plugin
  (`predict_dto.py`, `manifest.yaml`) de este repo.
- `"name"` en `models_config.json` siempre coincide exactamente con el `model_key` que usará el
  front — confirmarlo contra el skill `front-integration` si el modelo también se está integrando
  ahí en paralelo.
- Este skill no abre PR ni hace merge en el repo de ODD — deja los cambios listos para revisión
  humana.

## Checklist de salida

```
[ ] Modelo hermano identificado en models_config.json (mismo type)
[ ] Datos de entrenamiento subidos a S3 en datasets/{{PLUGIN_NAME}}/... (mismo dataset que el golden dataset de verification)
[ ] Entrada añadida a models_config.json con name == model_key exacto usado en el front
[ ] feature_cols/categorical_cols (si tabular) sacados de predict_dto.py/manifest.yaml, no inventados
[ ] VERSION incrementado
[ ] Buckets/endpoints de bootstrap y de detect-time verificados como el mismo almacén
[ ] Servicio desplegado/reiniciado o /baseline/{{PLUGIN_NAME}}/from-source invocado
[ ] GET /baselines o S3 confirma baselines/{{PLUGIN_NAME}}/baseline.json
[ ] POST /detect/{{PLUGIN_NAME}} probado con una fila real
[ ] Cambios listos para revisión humana — sin commit/push/PR hechos por este skill
```
