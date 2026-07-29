---
name: front-integration
description: Usa este skill para conectar un plugin ya integrado y verificado de inference-pan-model con el front de la plataforma (repo retech-lote2-xai-plataforma). Cubre los 3 puntos de conexión reales del front — resolución de vendor_payload en predict-task-manager.ts, catálogo MODEL_DESCRIPTIONS en intelligence-controller.ts, y el formulario/selector en el bundle Vue del sector en static/js/vue/. Requiere el plugin ya integrado (plugin-integration) y verificado en verde (verification) en este repo.
---

# Integración de un plugin en el front

## Placeholders de este skill

- `{{FRONT_REPO_PATH}}` — ruta local del repo del front (`retech-lote2-xai-plataforma`).
- `{{PLUGIN_NAME}}` — nombre del plugin ya integrado en `app/plugins/<nombre>/` de este repo
  (inference-pan-model), el que se va a exponer en el front.

Si al invocar este skill no se han informado ambos placeholders, para y pregunta antes de tocar
nada — nunca asumas una ruta de front por defecto ni "adivines" qué plugin integrar.

## Requisito previo

El plugin debe estar integrado (skill `plugin-integration`) y verificado en verde (skill
`verification`) en **este** repo. Si `outputs/<model_id>/verification_report.md` no existe o no
dice "LISTO PARA PR", para y avisa — este skill nunca conecta al front un plugin a medio terminar.

## Arquitectura del front relevante

`{{FRONT_REPO_PATH}}` es una plataforma no-code de modelado que expone los plugins de
inference-pan-model como un catálogo de "modelos vendor" agrupados por sector de negocio.
Integrar un modelo nuevo en el front no es escribir un microservicio nuevo — es conectarlo en 3
puntos:

| Capa | Fichero | Qué hace |
|---|---|---|
| Orquestación / respuesta | `src/service/predict-task-manager.ts` | Resuelve qué modelo interno corresponde a un `vendor_payload.model_id` devuelto por el backend, y extrae `prediction`/`score`/`xai_feature_values` de esa respuesta para mostrarlos en la UI |
| Catálogo | `src/controllers/intelligence-controller.ts` | `MODEL_DESCRIPTIONS` — diccionario `model_key -> descripción`, usado al crear el modelo (`createTrainTask`/`createPredictTask`) |
| Formulario / UI | `static/js/vue/<sector>.js` | Componente Vue del sector: opción del selector de modelo, schema de campos del CSV/formulario, y condicionales `selectedModel === "modelo-N"` para el comportamiento específico de ese modelo |

Punto opcional:
- `src/service/prediction-history.ts` (`MONITORED_MODEL_KEYS`) — solo si el modelo necesita
  histórico de monitorización/deriva, como el ya integrado `modelo-25`.

## Paso 0 — Encuentra el modelo hermano más parecido ya integrado

Igual que en `plugin-integration`, no se construye desde cero: se copia el patrón de un modelo
del mismo tipo/sector ya conectado al front. En `{{FRONT_REPO_PATH}}`:

```bash
grep -rn "modelo-<N>" src/service/predict-task-manager.ts src/controllers/intelligence-controller.ts static/js/vue/*.js
```

para 2-3 `model_key` ya integrados del mismo sector/tipo que `{{PLUGIN_NAME}}` (tabular vs
imagen, mismo sector de negocio) y usar sus 3 puntos de conexión como plantilla.

Para saber qué bundle Vue de sector le corresponde, no hay una tabla fija — se determina
grepeando el `model_key` de un modelo hermano, como arriba. A fecha de este skill existen
`fermentation-optimization.js` (vitivinícola), `costs-price-predict.js` (predicción de
precio/coste), `predictive-maintenance.js` (mantenimiento predictivo/detección de fallos),
`anomalies-detection.js` (detección de anomalías) y `ws-models.js` (genérico —
`SECTORAL_MODEL_TYPES` + aplicar modelo a fuente de datos). Esta lista puede haber cambiado desde
que se escribió este skill; confirmar siempre por grep, no confiar en ella a ciegas.

Los modelos de imagen (clasificación/detección con CNN) normalmente no necesitan un schema de
campos tabular en el bundle del sector — usan el flujo genérico de subida de imagen que ya existe
en `bundle.js`/`ws-models.js`. Confirmar contra un modelo de imagen hermano ya integrado antes de
asumir que hace falta un formulario nuevo.

## Paso 1 — model_key / model_id

Decide el `model_key` de catálogo (formato `modelo-<N>`, el siguiente número libre) y confirma
cuál es el `model_id` real que el backend/orquestador devolverá en `vendor_payload.model_id` para
`{{PLUGIN_NAME}}` — sale de `app/registry.py` de este repo (el `model_id`/`prefix` con el que se
registró el `ModelEntry`). Si no coinciden, la resolución en `predict-task-manager.ts` no
encontrará el modelo: revisar ahí el bloque de comentarios sobre colisiones de substring entre
`ml35`/`ml46`/`ml34`/`ml4` — el mismo problema puede repetirse con `{{PLUGIN_NAME}}`.

## Paso 2 — predict-task-manager.ts

Dos añadidos, copiando el patrón del modelo hermano del paso 0:

1. La rama de resolución (`vpModelId.includes(...)`) que mapea el `model_id` real devuelto por el
   vendor al identificador interno usado en el resto del fichero.
2. El bloque de extracción de campos (`if (effectiveModelId === "...")` / entrada del switch de
   `vendorPayload`) que saca `prediction`/`score`/`xai_feature_values` de la respuesta real de
   `/predict` de `{{PLUGIN_NAME}}`. Los nombres de campo deben coincidir exactamente con
   `predict_dto.py` (`PredictInlineResponse`/`PredictBatchResponse`) del plugin — nunca
   inventados.

## Paso 3 — intelligence-controller.ts

Añadir una entrada a `MODEL_DESCRIPTIONS` con el `model_key` del paso 1 y una descripción de una
frase, mismo tono que las existentes — sacada de la ficha funcional/memoria de `{{PLUGIN_NAME}}`,
nunca redactada libremente desde cero.

## Paso 4 — bundle Vue del sector

En `static/js/vue/<sector>.js` identificado en el paso 0:

- Añadir la opción del `model_key` al selector de modelos.
- Si el modelo es tabular: añadir su entrada al objeto de schema de campos (`csvField(...)` /
  lista de `fields`), con los mismos nombres/tipos que `predict_dto.py` — nunca inventar columnas.
- Replicar los condicionales `selectedModel === "modelo-N"` que gobiernan modo de entrada (inline
  vs batch), validación y qué campos se muestran, adaptados a `{{PLUGIN_NAME}}`.

## Paso 5 — rebuild y prueba local

```bash
cd {{FRONT_REPO_PATH}}
npm run client-build   # make-bundle.js + minify.js + translate-file-maker.js — regenera los .min.js
npm run build          # solo si también tocaste TypeScript (predict-task-manager.ts / intelligence-controller.ts)
```

Arrancar el front en local y probar el flujo completo: crear modelo → predict inline → predict
batch (si aplica) → comprobar que la ficha de resultados muestra `prediction`/`score`/XAI
correctamente, contra el backend de inference-pan-model corriendo en local con `{{PLUGIN_NAME}}`
cargado.

## Reglas que nunca se saltan

- Nunca se inventan nombres de campo, columnas de CSV ni descripciones — todo sale del contrato
  real del plugin (`predict_dto.py`, `manifest.yaml`) o de la memoria/ficha funcional ya generada.
- Nunca se edita a mano un `.min.js` — siempre se edita el fuente (`static/js/vue/<sector>.js` o
  el `.ts` correspondiente) y se regenera con `npm run client-build`.
- Este skill no abre PR ni hace merge en el repo del front — deja los cambios listos para revisión
  humana, igual que el resto del flujo de este repo.
- Si no existe un modelo hermano parecido (primer modelo de un sector nuevo), para y confírmalo
  con el usuario antes de crear un bundle Vue nuevo desde cero — es una decisión de diseño de UI,
  no una integración mecánica.

## Checklist de salida

```
[ ] Modelo hermano identificado y sus 3 puntos de conexión localizados
[ ] model_key asignado y model_id real de app/registry.py confirmado
[ ] predict-task-manager.ts: resolución de vendor_payload.model_id + extracción de campos añadidas
[ ] intelligence-controller.ts: entrada en MODEL_DESCRIPTIONS añadida
[ ] static/js/vue/<sector>.js: opción de modelo + schema de campos + condicionales añadidos
[ ] npm run client-build (y npm run build si aplica) sin errores
[ ] Probado en local: crear modelo, predict inline/batch, ficha de resultados correcta
[ ] Cambios listos para revisión humana — sin commit/push/PR hechos por este skill
```
