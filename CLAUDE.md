# CLAUDE.md — inference-pan-model

Repo de plugins de inferencia de DatagIA (arquitectura hexagonal). Este fichero es solo
navegación — el detalle de cada tarea vive en su skill correspondiente, no lo dupliques aquí.

## Stack

- Python, FastAPI, arquitectura hexagonal: `app/domain` (puertos), `app/application` (casos
  de uso genéricos), `app/infrastructure` (DI, router_factory, artifact store), `app/plugins/<nombre>`
  (implementación concreta de cada modelo).
- Cada modelo = 1 plugin autocontenido, expuesto en `/models/<model-id>/...`.
- Tests: `pytest`, en `tests/unit/`.
- Artefactos: local en `artifacts/<ARTIFACT_FOLDER_NAME>/` o en
  `s3://<STORAGE_BUCKET>/artifacts/fixed/<ARTIFACT_FOLDER_NAME>/`.

## Convención de carpetas de trabajo

Por modelo se usa el código `aNN` del caso de uso (`a07`, `a33`…):

| Ruta | Contenido | ¿Se commitea? |
|---|---|---|
| `inbox/aNN/manifest.yaml` | Manifest extraído (`manifest-extraction`) | **Sí** |
| `inbox/aNN/entregable/` | Memoria `.docx` entregada por el equipo de IA | **Sí** |
| `inbox/aNN/codigo/` | Código original del equipo de IA (datasets, checkpoints, a veces su propio `.git`) | **No** — excluido por `inbox/*/codigo/` en `.gitignore` |
| `outputs/aNN/` | Fichas institucionales, `datos_*.json` e informe de verificación | **Sí** |
| `outputs/*.md` | Informes transversales a varios modelos | **Sí** |
| `app/plugins/<nombre>/` | Plugin integrado | **Sí** |
| `artifacts/<ARTIFACT_FOLDER_NAME>/` | Copia local de los artefactos (la fuente es S3) | **No** |

**`codigo/` es lo único de `inbox/` que nunca se commitea.** Si al mover código entregado a
`codigo/` Git marca como borrados ficheros que antes estaban rastreados en `inbox/aNN/`, ese
borrado es correcto y se commitea.

## Skills disponibles

| Cuándo | Skill |
|---|---|
| Vas a integrar un modelo nuevo desde cero | `.claude/skills/manifest-extraction/SKILL.md` primero, luego `.claude/skills/plugin-integration/SKILL.md` |
| Vas a generar las fichas institucionales | `.claude/skills/docs-generation/SKILL.md` |
| Vas a validar que un plugin está listo para PR | `.claude/skills/verification/SKILL.md` |
| Vas a conectar un plugin ya verificado con el front de la plataforma | `.claude/skills/front-integration/SKILL.md` |
| Vas a conectar un plugin ya verificado con el servicio de explicabilidad (SHAP) | `.claude/skills/explainability-integration/SKILL.md` |
| Vas a conectar un plugin ya verificado con el servicio de detección de drift (ODD) | `.claude/skills/odd-integration/SKILL.md` |

## Orden de trabajo estándar para un modelo nuevo

1. `manifest-extraction` — genera `inbox/aNN/manifest.yaml`
2. `plugin-integration` — escribe `app/plugins/<nombre>/` y registra en `app/registry.py`
3. `verification` — checklist técnico + correctitud contra golden dataset, en bucle hasta verde
4. `docs-generation` — genera los 3 documentos institucionales en `outputs/aNN/`
5. Revisión humana del `verification_report.md` y los documentos antes de abrir PR

## Reglas que nunca se saltan

- Todo plugin lleva `mlflow_utils.py`, sin excepción, aunque el modelo no soporte hoy
  reentrenamiento por usuario.
- Nunca merge directo a `develop` — siempre PR + revisión humana.
- Nomenclatura obligatoria de plugin: carpeta `mlNN_<sector>_<desc>` (sector en inglés:
  `cereals`, `dairy`, `meat`, `wine`…), `MODEL_ID` = la misma cadena con guiones y
  `ARTIFACT_FOLDER_NAME` = el nombre de la carpeta. Detalle en `plugin-integration`.
- Nunca se inventa un golden case: sale del dataset real (`inbox/aNN/codigo/.../data/splits/`)
  o de una tabla de resultados auditada en la memoria — nunca de la imaginación del agente.
- Si un caso del golden dataset falla la tolerancia, no se silencia ni se ajusta la tolerancia
  para que pase — se investiga y se documenta.
- La ficha técnica y la ficha funcional se generan siempre con la plantilla paramétrica fija
  (`docs-generation/ficha-modelo/`, ver su `SKILL.md`), nunca redactando o editando el .docx
  del modelo anterior a mano — así se garantiza el mismo formato para los 48 modelos.