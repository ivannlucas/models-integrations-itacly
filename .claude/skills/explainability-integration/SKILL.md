---
name: explainability-integration
description: Usa este skill para conectar un plugin ya integrado y verificado de inference-pan-model con el servicio de explicabilidad (repo retech-lote2-xai-explicabilidad, SHAP). A diferencia del front, aquí no hay una carpeta por modelo — se añade una rama por model_id dentro de TabularXAIPlugin/VisionXAIPlugin y una entrada en el registro estático de container.py. Requiere el plugin ya integrado (plugin-integration) y verificado en verde (verification) en este repo.
---

# Integración de un plugin en el servicio de explicabilidad

## Placeholders de este skill

- `{{XAI_REPO_PATH}}` — ruta local del repo de explicabilidad (`retech-lote2-xai-explicabilidad`).
- `{{PLUGIN_NAME}}` — nombre del plugin ya integrado en `app/plugins/<nombre>/` de este repo
  (inference-pan-model), el que se va a explicar.

Si al invocar este skill no se han informado ambos placeholders, para y pregunta antes de tocar
nada — nunca asumas una ruta del servicio de explicabilidad por defecto ni "adivines" qué plugin
integrar.

## Requisito previo

El plugin debe estar integrado (skill `plugin-integration`) y verificado en verde (skill
`verification`) en **este** repo. Si `outputs/<model_id>/verification_report.md` no existe o no
dice "LISTO PARA PR", para y avisa.

## Cómo NO es esta integración

`{{XAI_REPO_PATH}}` **no** replica el patrón `app/plugins/<nombre>/` de este repo. No hay carpeta
por modelo. Solo existen **dos** clases de plugin en todo el servicio:

- `app/domain/services/plugins/tabular/plugin.py` → `TabularXAIPlugin` — un `elif self._model_id
  == "<id>":` por cada modelo tabular ya integrado, dentro de `_load()`, `explain()` y
  `validate()`.
- `app/domain/services/plugins/vision/plugin.py` → `VisionXAIPlugin` — mismo patrón para modelos
  de imagen/vídeo.

Integrar `{{PLUGIN_NAME}}` es añadir una rama nueva dentro de la clase que corresponda, no crear
ficheros nuevos por modelo (salvo el submódulo opcional del paso 4 si hay preprocesado propio).

Este servicio **no** llama al endpoint `/predict` del backend ni recibe datos precalculados de un
orquestador: carga su propia copia de los mismos artefactos entrenados desde el S3/MinIO
compartido (`app/infrastructure/storage/artifact_store.py`), porque SHAP necesita una función
`predict_fn(numpy) -> numpy`, no el contrato JSON de `ModelPluginPort`.

## Paso 0 — Localiza el ARTIFACT_FOLDER_NAME real en este repo

En **este** repo (inference-pan-model), abre `app/plugins/{{PLUGIN_NAME}}/constants.py` y copia
el valor exacto de `ARTIFACT_FOLDER_NAME`. Este es el nombre de carpeta S3
(`artifacts/fixed/<ARTIFACT_FOLDER_NAME>/`) que el servicio de explicabilidad usará para
descargar el artefacto — **nunca se adivina a partir del `model_id`**, tienen que coincidir letra
por letra. Varios modelos ya integrados en `{{XAI_REPO_PATH}}` tienen este mapeo marcado como
"UNVERIFIED" en comentarios de `plugin.py` precisamente por haberse asumido en vez de comprobado
— no repitas ese error.

## Paso 1 — Encuentra el modelo hermano más parecido ya integrado

En `app/domain/services/plugins/tabular/plugin.py` (o `vision/plugin.py` si es de imagen/vídeo),
busca un `elif self._model_id == "..."` de un modelo del mismo tipo de arquitectura que
`{{PLUGIN_NAME}}` y cópialo como plantilla:

| Tipo de modelo cargado | Explainer | Ejemplo ya integrado a copiar |
|---|---|---|
| sklearn/XGBoost/árbol o lineal puro | `shap.TreeExplainer`/`LinearExplainer` nativo, vía `explain_with_shap()` genérico — no hace falta wrapper | `modelo40_industrial` (RandomForest, feature names desde `model.feature_names_in_`) |
| PyTorch / red neuronal con scalers externos | `shap.KernelExplainer` sobre un `predict_fn` a medida, vía `explain_with_shap_callable()` | `dairy_pasteurization_energy` (ml34) — `_dairy_pasteurization_energy_predict_fn()` escala→infiere→des-escala |
| Imagen/vídeo (CNN) | `VisionXAIPlugin`, Grad-CAM | ver ramas ya existentes en `vision/plugin.py` |

## Paso 2 — Carga del artefacto (`_load()`)

Añade la rama `elif self._model_id == "{{PLUGIN_NAME}}":` en `_load()` replicando cómo el propio
`model_loader.py`/`predict.py` de `app/plugins/{{PLUGIN_NAME}}/` en este repo carga el modelo
(joblib/torch, scalers, orden de features). Reglas que no se saltan aquí:

- Nunca lanzar excepción si el artefacto no está disponible — solo `logger.warning(...)` y
  `return`, dejando `self._model = None` (patrón `_dummy_explain()`/`_dummy_regions()`). El
  servicio siempre responde algo, nunca revienta por artefacto ausente.
- El orden de features (`feature_names`) sale siempre de una fuente real: la config estática de
  `container.py`, un fichero que trae el propio artefacto (`feature_list.txt`,
  `model_config.json`), o `model.feature_names_in_` — nunca de `input_data.keys()` re-derivado
  a mano.
- Si el modelo tabular usa scalers (`scaler_X.pkl`/`scaler_y.pkl`) o metadata de arquitectura
  (`model_config.json`), cárgalos igual que el modelo hermano del paso 1.

## Paso 3 — Registro en `S3_ARTIFACT_NAMES` (solo tabular)

En `plugin.py`, añade `S3_ARTIFACT_NAMES["{{PLUGIN_NAME}}"] = "<ARTIFACT_FOLDER_NAME del paso 0>"`.
Para modelos de imagen, el equivalente (`artifact_s3_name`) se pasa directamente en el registro
de `container.py` del paso 5, no hace falta este diccionario.

## Paso 4 — Preprocesado propio (si aplica)

Si `{{PLUGIN_NAME}}` necesita ingeniería de features no trivial antes de poder llamar al modelo
(igual que `modelo40_industrial/features.py`), vendoriza ese preprocesado — copiado del
`preprocessing.py`/equivalente real de `app/plugins/{{PLUGIN_NAME}}/` en este repo, no
reescrito de memoria — en un submódulo nuevo
`app/domain/services/plugins/tabular/{{PLUGIN_NAME}}/`, no inline en `plugin.py`. Documenta con
un comentario que es una copia vendorizada del código fuente real, para que quien la toque sepa
que hay riesgo de divergencia si el original cambia.

## Paso 5 — `explain()` / `validate()`

Si el modelo entra por la vía `explain_with_shap()` genérica (árbol/lineal), no hace falta rama
nueva en `explain()`. Si necesita `explain_with_shap_callable()` (PyTorch/callable), escribe
`_<{{PLUGIN_NAME}}>_predict_fn()` siguiendo el patrón del ejemplo del paso 1, y añade la rama
correspondiente **tanto en `explain()` como en `validate()`** (el chequeo de dominancia usa la
misma rama). Sanea todo output numérico con el mismo helper `_safe_float()` ya usado en el resto
del fichero — nunca dejar que un NaN/Inf llegue a `json.dumps`.

No dupliques generación de gráfico ni el chequeo de calidad dentro de la rama nueva: `explain()`
ya los aplica de forma centralizada después de que la rama devuelva `top_features`/`base_value`/
`output_value` — la rama nueva solo debe devolver esa forma.

## Paso 6 — Registro en `container.py`

En `app/infrastructure/http/dependencies/container.py`:

- **Tabular** → nueva entrada en `TABULAR_MODELS["{{PLUGIN_NAME}}"]` con `artifact_dir`
  (termina exactamente en el `ARTIFACT_FOLDER_NAME` del paso 0, **no** en `{{PLUGIN_NAME}}`),
  `features` (lista estática, o `[]` con un comentario explicando por qué si solo se conocen tras
  cargar el artefacto), y `background` (ruta a CSV o `None`).
- **Imagen/vídeo** → nueva entrada en `VISION_MODELS["{{PLUGIN_NAME}}"]` con `artifact_dir`,
  `artifact_s3_name`, y `model_loader_fn` opcional si `torch.load()` solo no reconstruye el
  modelo.

Este es el registro real que usa `get_plugin()` — un `model_id` no presente aquí devuelve
`ModelNotSupportedError` (422) en el router, sea cual sea el estado de las ramas en `plugin.py`.

## Paso 7 — Background dataset (recomendado, tabular)

Si el modelo va por `KernelExplainer`/`LinearExplainer`, añade un CSV representativo en
`data/background/{{PLUGIN_NAME}}.csv` con las mismas columnas que `features`, y referencia su
ruta en el paso 6. Sin background dataset el `base_value` de la explicación no es realista.

## Paso 8 — Tests y despliegue

- Añade un test en `test/test_explain.py` que golpee `POST /explanations` para
  `{{PLUGIN_NAME}}`. Los tests existentes hacen monkeypatch de `router_mod.get_plugin` con
  `FakeTabularPlugin`/`FakeVisionPlugin` (`test/conftest.py`) y por tanto no ejercitan el registro
  real — añade también, si es posible, una prueba directa de
  `container.get_plugin("{{PLUGIN_NAME}}")` para validar que el registro de `container.py` y el
  mapeo de S3 funcionan de extremo a extremo.
- Sube el artefacto al bucket S3 compartido bajo `artifacts/fixed/<ARTIFACT_FOLDER_NAME>/` (o
  vendoriza localmente en `artifacts/<ARTIFACT_FOLDER_NAME>/` para desarrollo local).
- Despliega/reinicia el servicio y verifica `GET /stats` — debe listar `{{PLUGIN_NAME}}`.

## Reglas que nunca se saltan

- `artifact_dir`/`artifact_s3_name` deben terminar exactamente en el `ARTIFACT_FOLDER_NAME` real
  del paso 0, nunca en `{{PLUGIN_NAME}}` a secas — es el error más repetido en este repo según sus
  propios comentarios.
- Nunca se lanza excepción por artefacto ausente en `_load()` — se degrada a explicación dummy con
  `self._model = None` y `logger.warning`, nunca un crash.
- El orden de features nunca se re-deriva de `input_data.keys()` — sale de config estática, de
  metadata del propio artefacto, o de `model.feature_names_in_`.
- Todo output numérico pasa por `_safe_float()` antes de serializarse.
- El registro real y único que decide si un `model_id` está soportado es `container.py`
  (`TABULAR_MODELS`/`VISION_MODELS`) — `xai_runtime_service.py` es un registro legacy que ningún
  router usa hoy; no confundir uno con otro.
- Este skill no abre PR ni hace merge en el repo de explicabilidad — deja los cambios listos para
  revisión humana.

## Checklist de salida

```
[ ] ARTIFACT_FOLDER_NAME confirmado desde app/plugins/{{PLUGIN_NAME}}/constants.py de este repo
[ ] Modelo hermano identificado (tipo de explainer: nativo vs KernelExplainer sobre predict_fn)
[ ] _load() con rama nueva para {{PLUGIN_NAME}}, degradando a dummy si falta el artefacto
[ ] S3_ARTIFACT_NAMES actualizado (tabular) o artifact_s3_name en container.py (vision)
[ ] Preprocesado propio vendorizado en submódulo si aplica, con nota de origen
[ ] explain()/validate() con rama nueva si usa KernelExplainer; _safe_float() aplicado
[ ] TABULAR_MODELS/VISION_MODELS en container.py con artifact_dir/features/background correctos
[ ] Background CSV en data/background/ si aplica
[ ] Test en test/test_explain.py + (recomendado) test directo de container.get_plugin(...)
[ ] Artefacto desplegado en S3 fixed/<ARTIFACT_FOLDER_NAME>/, GET /stats lista {{PLUGIN_NAME}}
[ ] Cambios listos para revisión humana — sin commit/push/PR hechos por este skill
```
