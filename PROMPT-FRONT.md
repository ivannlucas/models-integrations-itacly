Usa el skill front-integration para conectar el plugin {PLUGIN_NAME} (ya integrado y verificado
en este repo, inference-pan-model) con el front de la plataforma, ubicado en:

{FRONT_REPO_PATH}

Alcance:
- Confirma primero que existe outputs/{PLUGIN_NAME}/verification_report.md y que dice "LISTO
  PARA PR". Si no existe o no está en verde, para y avísame — no conectes al front un plugin sin
  verificar.
- Sigue el proceso del skill en orden: encuentra el modelo hermano ya integrado más parecido
  (mismo tipo tabular/imagen, mismo sector) grepeando su model_key en
  src/service/predict-task-manager.ts, src/controllers/intelligence-controller.ts y
  static/js/vue/*.js dentro de {FRONT_REPO_PATH}, y usa sus 3 puntos de conexión como plantilla.
- Asigna el model_key de catálogo (modelo-<N>, el siguiente libre) y confirma el model_id real
  que devuelve app/registry.py de este repo para {PLUGIN_NAME} — deben poder resolverse entre sí
  en predict-task-manager.ts.
- Añade la resolución del vendor_payload.model_id + la extracción de prediction/score/
  xai_feature_values en predict-task-manager.ts, la entrada en MODEL_DESCRIPTIONS de
  intelligence-controller.ts, y la opción de modelo + schema de campos + condicionales en el
  bundle Vue del sector correspondiente en static/js/vue/. Los nombres de campo y columnas
  nunca se inventan: deben coincidir con predict_dto.py del plugin en este repo.
- Si {PLUGIN_NAME} es un modelo de imagen, confirma primero si de verdad hace falta un
  formulario nuevo en el bundle del sector o si el flujo genérico de subida de imagen ya
  integrado le sirve — no crees UI nueva sin comprobarlo contra un modelo hermano.
- Si no hay ningún modelo hermano parecido (primer modelo de un sector nuevo en el front), para
  y pregúntame antes de diseñar un bundle Vue nuevo desde cero.
- Tras los cambios, ejecuta npm run client-build (y npm run build si tocaste TypeScript) dentro
  de {FRONT_REPO_PATH} y arranca el front en local para probar el flujo completo: crear modelo →
  predict inline → predict batch (si aplica) → ficha de resultados con prediction/score/XAI
  correctos, contra el backend de inference-pan-model corriendo en local con {PLUGIN_NAME}
  cargado.

No abras PR ni hagas commit/push en el repo del front. Deja los cambios listos para revisión
humana y resume al final: qué modelo hermano usaste de referencia, qué ficheros tocaste, y
cualquier ambigüedad o decisión de UI que necesite mi confirmación.
