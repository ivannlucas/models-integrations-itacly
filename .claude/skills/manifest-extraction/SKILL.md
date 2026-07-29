---
name: manifest-extraction
description: Usa este skill como primer paso al integrar un modelo nuevo. Genera inbox/<model_id>/manifest.yaml a partir de las 3 entradas disponibles — código del modelo, dataset incluido en el código, y el entregable (memoria .docx). Es el prerrequisito obligatorio del skill plugin-integration; no se scaffoldea ningún plugin sin manifest.
---

# Extracción del manifest

## Por qué existe este paso

Sin un manifest, un agente puede integrar un plugin perfectamente conectado (wiring OK) y aun
así no saber si predice bien. Este skill construye el manifest ANTES de tocar código de plugin,
para que la verificación posterior tenga algo real contra lo que comparar.

## Entradas esperadas

```
inbox/<model_id>/
├── codigo/          # carpeta o .rar/.zip descomprimido, tal como lo entrega el equipo de IA
└── entregable/      # memoria .docx (plantilla de memoria del proyecto)
```

## Paso 1 — Inspeccionar el código

- Localizar `src/` (o equivalente) e identificar el tipo de modelo: regresión, clasificación,
  optimización prescriptiva (modelo + optimizador tipo GA), serie temporal, etc.
- Localizar el fichero de predicción/inferencia (`predict.py`, `predictor.py`) — ahí está la
  lógica real de pre/postprocesado que hay que replicar en el plugin.
- Localizar `requirements.txt` y **comprobar su encoding** (`file` o `chardet`). Es habitual
  encontrarlo en UTF-16LE con CRLF — normalizar a UTF-8/LF antes de usarlo en cualquier sitio,
  o romperá silenciosamente `pip install`/`pip-audit` en el pipeline.
- Localizar `config.yaml` o equivalente — suele contener valores por defecto y, a veces,
  escenarios de validación oficiales ya definidos por el equipo de IA.
- Localizar el código de entrenamiento original (`train.py`, notebook, o la sección de
  entrenamiento dentro de `predict.py`) — de ahí salen `target_column`, las columnas requeridas
  y los hiperparámetros (optimizador, learning rate, epochs) que van al bloque `training` del
  manifest. Si el modelo no trae código de entrenamiento propio (p. ej. solo se entrega el
  artefacto ya entrenado, sin forma de reentrenarlo con datos nuevos), `training.supported` es
  `false` — no se inventa un procedimiento de fine-tuning que el equipo de IA no definió.

## Paso 2 — Extraer golden cases del dataset

Buscar en el código splits de test ya generados (`data/splits/X_test.csv` + `y_test.csv` o
equivalente) — normalmente están alineados fila a fila con el target real, sin escalar. Esta
es la fuente PREFERIDA de golden cases porque:

- da precisión numérica completa, no solo los casos que aparecen documentados en la memoria
- permite elegir cuántos casos usar (recomendado: 15-20 filas variadas del test set)
- es reproducible si el dataset fija un `random_seed`

Si no existe split de test explícito, usar como fuente secundaria una tabla de
escenarios/resultados ya auditados en la memoria (ver Paso 3). Nunca inventar casos ni generar
inputs sintéticos propios para golden cases.

## Paso 3 — Extraer specs de la memoria

```bash
pandoc -t markdown inbox/<model_id>/entregable/*.docx -o memoria.md
```

Buscar y extraer (los nombres de sección son razonablemente estables entre memorias del proyecto):

- **"Entradas y Salidas"** — variables de entrada (fijas vs. de control/decisión vs. de estado),
  tipo de dato, estructura de salida.
- **"Arquitectura y Algoritmos"** — tipo de modelo, framework, hiperparámetros clave.
- **Tabla de métricas de test** ("Resultados en Conjunto de Test") — R², MAE, RMSE. Se usa como
  referencia de tolerancia en el skill `verification`, no como caso verificable 1:1.
- **Tabla de escenarios/resultados** (tipo "Tabla 6", si existe) — golden cases adicionales con
  resultado ya auditado. Útil sobre todo cuando no hay split de test explícito, o para validar
  el pipeline completo (modelo + postproceso de negocio), no solo el modelo aislado.
- **Restricciones de negocio** (p. ej. "PU >= 13") — van a `constraints` en el manifest y deben
  acabar como excepción de dominio en el skill `plugin-integration`, no como un output más.

## Paso 4 — Ensamblar manifest.yaml

Esqueleto de referencia (ver `inbox/a35/manifest.yaml` como ejemplo ya completado):

```yaml
model_id: <sector>-<nombre-corto>
sector: <vitivinicola|carnico|cerealista|lacteo>
nombre: "<título completo tal como aparece en la memoria>"
task_type: <regression|classification|regression_prescriptive|...>
framework: [<lista de librerías núcleo>]

artifacts:
  folder: models/artifacts/     # ruta dentro del código entregado
  files: [...]

inputs:
  fixed: [...]
  control: [...]                # solo si hay optimización: variables que decide un GA/optimizador
  derived_if_missing: {...}     # campos que se calculan solos si no vienen informados

outputs:
  predict_inline: [...]
  # + modos adicionales (optimize, batch...) si el modelo no es un predictor puro

training:
  supported: <true|false>       # false si no hay código/procedimiento de entrenamiento entregado
  target_column: <null si supported=false>   # columna objetivo en el CSV de data_path
  required_columns: [...]       # features + target que debe traer el CSV de entrenamiento
  hyperparams: {...}            # optimizador, lr, epochs — del código de entrenamiento original
  metrics_returned: [...]       # métricas que debe devolver TrainResponse (mae, r2, ...)
  source: codigo_entrenamiento  # o memoria_seccion_X — trazabilidad de dónde salió cada dato

constraints: {...}              # restricciones duras de negocio, no un output más

metrics_reported: {...}         # del hold-out test — referencia de tolerancia, no caso verificable

golden_cases:
  - id: caso_001
    input: {...}
    expected: {...}
    source: dataset_test_split  # o memoria_tabla_X — trazabilidad de dónde salió cada caso

known_issues: [...]             # cualquier gotcha de formato/encoding/estructura detectado
```

## Reglas

- No se avanza al skill `plugin-integration` sin manifest completo.
- Cualquier campo que no se pueda rellenar con evidencia real (código, dataset o memoria) se
  deja explícitamente como `null` con un comentario explicando por qué falta — nunca se rellena
  por inferencia del agente.
- `training.target_column`, `training.required_columns` y `training.hyperparams` nunca se
  infieren ni se copian de otro modelo — o salen del código de entrenamiento/memoria de este
  modelo concreto, o `training.supported` queda en `false` con la razón en `known_issues`.
- Si la memoria no sigue la estructura de secciones esperada, señalarlo en `known_issues` y
  pedir confirmación humana antes de continuar.
