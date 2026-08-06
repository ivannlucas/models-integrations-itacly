Usa el skill odd-integration para conectar el plugin {PLUGIN_NAME} (ya integrado y verificado en
este repo, inference-pan-model) con el servicio de detección de drift (ODD), ubicado en:

{ODD_REPO_PATH}

Alcance:
- Confirma primero que existe outputs/{PLUGIN_NAME}/verification_report.md y que dice "LISTO
  PARA PR". Si no existe o no está en verde, para y avísame — no conectes a ODD un plugin sin
  verificar.
- Este servicio se integra sin tocar código Python: es una entrada nueva en
  models_config.json. Si en algún momento sientes que necesitas tocar main.py, detector.py,
  preprocessors.py o baseline_bootstrap.py para dar de alta {PLUGIN_NAME}, para y pregúntame —
  es señal de que el modelo no encaja en los tipos tabular/image tal cual están soportados hoy.
- Busca en models_config.json un modelo hermano ya integrado del mismo tipo (tabular vs
  imagen/vídeo) y copia su estructura exacta.
- El campo "name" de la nueva entrada debe ser idéntico byte a byte al model_key/model_id que
  usa (o usará) el front en /detect/{model_name} — si {PLUGIN_NAME} también se está conectando
  al front en paralelo, confirma que coincide con lo que se usó ahí (skill front-integration).
- feature_cols/categorical_cols (si es tabular) deben salir de predict_dto.py/manifest.yaml
  reales de app/plugins/{PLUGIN_NAME}/ en este repo, nunca inventados. categorical_cols
  siempre subconjunto de feature_cols.
- Sube a S3 los datos de entrenamiento de {PLUGIN_NAME} en datasets/{PLUGIN_NAME}/... — el
  mismo dataset que se usó como golden dataset en la skill verification, nunca datos
  sintéticos nuevos. No crees nada bajo la carpeta local datasets/ de {ODD_REPO_PATH}, es
  efímera.
- Incrementa el fichero VERSION de {ODD_REPO_PATH}.
- Antes de dar por bueno el despliegue, verifica que las variables de entorno de bucket S3 que
  usa el bootstrap (S3_BASELINE_BUCKET) y las que usa el servicio en tiempo de detección
  (OOD_BASELINE_BUCKET/DRIFT_BASELINE_BUCKET + CUSTOM_S3_ENDPOINT/CUSTOM_REGION) apuntan al
  mismo almacén en el entorno de destino — si divergen, el baseline generado queda invisible
  para /detect aunque el registro esté bien.
- Tras desplegar/reiniciar (o invocar /baseline/{PLUGIN_NAME}/from-source), verifica que
  aparece baselines/{PLUGIN_NAME}/baseline.json en S3 o en GET /baselines, y prueba
  POST /detect/{PLUGIN_NAME} con una fila real.

No abras PR ni hagas commit/push en el repo de ODD. Deja los cambios listos para revisión
humana y resume al final: qué modelo hermano usaste de referencia, la entrada añadida a
models_config.json, y cualquier ambigüedad (por ejemplo si el model_key todavía no está
decidido con el front) que necesite mi confirmación.
