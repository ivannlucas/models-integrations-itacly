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

- `inbox/<model_id>/` — entradas para integrar un modelo (código del equipo de IA + memoria
  entregable). No se commitea — añadir a `.gitignore`.
- `outputs/<model_id>/` — documentos institucionales generados + informe de verificación.
  Tampoco se commitea a `develop`.
- `app/plugins/<nombre>/` — destino final del plugin ya integrado. Esto sí se commitea.

## Skills disponibles

| Cuándo | Skill |
|---|---|
| Vas a integrar un modelo nuevo desde cero | `.claude/skills/manifest-extraction/SKILL.md` primero, luego `.claude/skills/plugin-integration/SKILL.md` |
| Vas a generar las fichas institucionales | `.claude/skills/docs-generation/SKILL.md` |
| Vas a validar que un plugin está listo para PR | `.claude/skills/verification/SKILL.md` |

## Orden de trabajo estándar para un modelo nuevo

1. `manifest-extraction` — genera `inbox/<model_id>/manifest.yaml`
2. `plugin-integration` — escribe `app/plugins/<nombre>/` y registra en `app/registry.py`
3. `verification` — checklist técnico + correctitud contra golden dataset, en bucle hasta verde
4. `docs-generation` — genera los 3 documentos institucionales en `outputs/<model_id>/`
5. Revisión humana del `verification_report.md` y los documentos antes de abrir PR

## Reglas que nunca se saltan

- Todo plugin lleva `mlflow_utils.py`, sin excepción, aunque el modelo no soporte hoy
  reentrenamiento por usuario.
- Nunca merge directo a `develop` — siempre PR + revisión humana.
- Nunca se inventa un golden case: sale del dataset real (`inbox/<model_id>/codigo/.../data/splits/`)
  o de una tabla de resultados auditada en la memoria — nunca de la imaginación del agente.
- Si un caso del golden dataset falla la tolerancia, no se silencia ni se ajusta la tolerancia
  para que pase — se investiga y se documenta.
