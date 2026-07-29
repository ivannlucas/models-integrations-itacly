Usa el skill docs-generation para generar ÚNICAMENTE la ficha técnica y la ficha
funcional (NO la plantilla de metadatos) de los siguientes modelos ya integrados:
a34, a40 y a46.

Alcance estricto:
- No toques app/plugins/ ni ejecutes integración ni verificación — los plugins ya
  están integrados. Solo documentación.
- Para cada modelo, sigue el flujo de ficha-modelo/: lee primero
  reference/mapeo_contenido.md y reference/esquema_datos.md, construye un único
  datos_<model_id>.json (campos de ambos esquemas), y genera los dos .docx con
  scripts/generar_ficha.py (--tipo tecnica y --tipo funcional) en
  outputs/<model_id>/.
- Fuentes por modelo: inbox/<model_id>/manifest.yaml, la memoria original del
  equipo de IA en inbox/<model_id>/, y outputs/<model_id>/verification_report.md
  si existe (incluye en metricas[] tanto las métricas declaradas en la memoria
  como el resultado verificado del golden dataset, como entradas separadas).
- Si a algún modelo le falta el manifest.yaml o la memoria en inbox/, NO inventes
  contenido: repórtalo y sáltalo, y continúa con los demás.                                                                                                                         
- Mantén las advertencias de datos sintéticos en las cifras de negocio
  ("requiere validación con datos reales antes de despliegue").                                                                                                                     
- Nunca copies párrafos literales de la memoria: reescribe al formato de la
  plantillainstitucional.                                                                                                                                                          
Al terminar, dame un resumen por modelo: documentos generados, fuentes usaday cualquier campo que no pudiste rellenar y por qué.Dos avisos prácticos antes de lanzarlo:

- inbox/ no se commitea — si vas a generar fichas de modelos integrados haceox/<model_id>/ (manifest + memoria) sigue existiendo en tu máquina. Si noestá, el prompt hará que se reporte en vez de inventar, pero conviene saberl- Verificación visual en PDF: el skill pide renderizar a PDF para revisar lo WSL no hay pandoc ni libreoffice instalados. Si quieres esa comprobación,añade al prompt una línea tipo "si no puedes renderizar a PDF localmente, in lo revise yo manualmente".
