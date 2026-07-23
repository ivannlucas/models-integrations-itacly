Usa el skill explainability-integration para conectar el plugin {PLUGIN_NAME} (ya integrado y
verificado en este repo, inference-pan-model) con el servicio de explicabilidad, ubicado en:

{XAI_REPO_PATH}

Alcance:
- Confirma primero que existe outputs/{PLUGIN_NAME}/verification_report.md y que dice "LISTO
  PARA PR". Si no existe o no está en verde, para y avísame — no conectes a explicabilidad un
  plugin sin verificar.
- Antes de tocar {XAI_REPO_PATH}, localiza en este repo app/plugins/{PLUGIN_NAME}/constants.py
  y copia el valor exacto de ARTIFACT_FOLDER_NAME — nunca lo derives del nombre del plugin a
  ojo, es el error más repetido en ese servicio según sus propios comentarios.
- Recuerda que ese repo no tiene una carpeta por modelo: es añadir una rama
  `elif self._model_id == "{PLUGIN_NAME}":` dentro de TabularXAIPlugin o VisionXAIPlugin
  (según el tipo de {PLUGIN_NAME}), más una entrada en TABULAR_MODELS/VISION_MODELS de
  app/infrastructure/http/dependencies/container.py. Busca primero un modelo hermano ya
  integrado del mismo tipo de arquitectura (sklearn/árbol con SHAP nativo vs PyTorch/callable
  con KernelExplainer) y cópialo como plantilla.
- Los nombres de features y cualquier preprocesado deben coincidir exactamente con
  predict_dto.py / preprocessing.py reales de app/plugins/{PLUGIN_NAME}/ en este repo — nunca
  inventados ni reescritos de memoria.
- Respeta el patrón de fallback silencioso ya usado en todo el servicio: si el artefacto no
  carga, self._model queda en None y se degrada a explicación dummy con logger.warning, nunca
  una excepción. Aplica _safe_float() a cualquier output numérico nuevo.
- Añade el CSV de background en data/background/{PLUGIN_NAME}.csv si el modelo va por
  KernelExplainer, y un test en test/test_explain.py para {PLUGIN_NAME}.
- Verifica GET /stats del servicio de explicabilidad tras el cambio — debe listar
  {PLUGIN_NAME}.

No abras PR ni hagas commit/push en el repo de explicabilidad. Deja los cambios listos para
revisión humana y resume al final: qué modelo hermano usaste de referencia, qué explainer
elegiste y por qué, qué ficheros tocaste, y cualquier ambigüedad (por ejemplo el mapeo de
ARTIFACT_FOLDER_NAME si no pudiste verificarlo con certeza) que necesite mi confirmación.
