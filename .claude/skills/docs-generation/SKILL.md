---
name: docs-generation
description: Usa este skill para generar los 3 documentos institucionales (plantilla de metadatos, ficha técnica, ficha funcional) de un modelo, a partir de su manifest.yaml y la memoria original. La ficha técnica y la ficha funcional se generan con la plantilla paramétrica fija de `ficha-modelo/` (docxtpl) — nunca redactando o editando el documento libremente. La plantilla de metadatos se apoya en el skill docx del sistema.
---

# Generación de documentos institucionales

## Requisito previo

Requiere `inbox/<model_id>/manifest.yaml` completo (skill `manifest-extraction`) y, si ya está
disponible, `outputs/<model_id>/verification_report.md` (skill `verification`) para poder
incluir el estado de validación real, no solo lo declarado por el equipo de IA.

## Antes de generar cualquier .docx

Consultar primero `/mnt/skills/public/docx/SKILL.md` (gotchas de docx-js, cómo verificar el
resultado renderizando a PDF). No se duplica ese contenido aquí.

## Convención de nombres

`a{número_modelo:02d}_metadatos`, `a{número_modelo:02d}_ficha_tecnica`,
`a{número_modelo:02d}_ficha_funcional`.

- **Plantilla de metadatos**: partir siempre de la plantilla corporativa ITACYL existente
  (editar, no crear desde cero salvo que no exista plantilla previa para ese tipo de documento).
- **Ficha técnica / ficha funcional**: **nunca** se editan a mano ni se parte del `.docx` del
  modelo anterior. Se generan siempre con la plantilla paramétrica de `ficha-modelo/` — ver
  sección siguiente. Esto es lo que garantiza el mismo formato exacto para los 48 modelos, en
  vez de ir arrastrando (y a veces rompiendo) el XML de un modelo al siguiente.

## Ficha técnica y ficha funcional: flujo con `ficha-modelo/`

Este skill incluye, en `ficha-modelo/`, las plantillas y el generador para estos dos documentos:

```
ficha-modelo/
├── assets/ficha_tecnica_template.docx
├── assets/ficha_funcional_template.docx
├── scripts/generar_ficha.py
└── reference/
    ├── esquema_datos.md          # esquema completo de campos esperados
    ├── mapeo_contenido.md        # de qué sección de la memoria sale cada campo, y el tono correcto
    └── notas_tecnicas_plantilla.md
```

Pasos:

1. Lee `ficha-modelo/reference/mapeo_contenido.md` y `ficha-modelo/reference/esquema_datos.md`.
2. Construye **un único** `datos_<model_id>.json` con los campos de ambos esquemas (técnica +
   funcional), usando como fuente el `manifest.yaml`, la memoria original, y — si ya existe —
   `verification_report.md`. Ver la tabla de mapeo memoria → documento más abajo para saber de
   qué sección sale cada bloque antes de traducirlo al esquema campo a campo de
   `esquema_datos.md`.
3. Genera los dos documentos:
   ```bash
   pip install docxtpl --break-system-packages   # si no está ya instalado
   python3 ficha-modelo/scripts/generar_ficha.py --tipo tecnica   --datos datos_<model_id>.json --out outputs/<model_id>/<model_id>_ficha_tecnica.docx
   python3 ficha-modelo/scripts/generar_ficha.py --tipo funcional --datos datos_<model_id>.json --out outputs/<model_id>/<model_id>_ficha_funcional.docx
   ```
4. Verifica visualmente (renderizar a PDF y mirar cada página, ver `/mnt/skills/public/docx/SKILL.md`)
   antes de dar el documento por bueno — en particular que las tablas/listas de longitud
   variable (inputs, outputs, métricas, limitaciones, KPIs) tienen tantas filas/bullets como
   elementos pusiste en el JSON, ni uno más ni uno menos.

No se toca `ficha-modelo/assets/*.docx` a mano. Si hay que cambiar el diseño, seguir
`ficha-modelo/reference/notas_tecnicas_plantilla.md` (el patrón de filas/párrafos repetidos con
`{%tr%}`/`{%p%}` es frágil si no se respeta exactamente).

## Mapeo memoria → documento

| Documento | Se rellena con (sección de la memoria) |
|---|---|
| **Plantilla de metadatos** | Título/nombre, sector, versión, listado de artefactos y librerías ("Requisitos de Hardware y Entorno"), tipo de modelo |
| **Ficha técnica** | "Arquitectura y Algoritmos", "Entradas y Salidas" (formato físico, estructura), "Hiperparámetros Clave", tabla de métricas de test, requisitos de hardware — mapeo detallado campo a campo en `ficha-modelo/reference/mapeo_contenido.md` |
| **Ficha funcional** | "Objetivos" de negocio, KPI y su cumplimiento (tabla de resultados/escenarios), "Modo de Despliegue" y "Mecanismo de Inferencia" en lenguaje no técnico, "Limitaciones, Riesgos y Consideraciones" — mapeo detallado campo a campo en `ficha-modelo/reference/mapeo_contenido.md` |

## Reglas

- Nunca reproducir párrafos completos de la memoria tal cual — reescribir/resumir al formato de
  la plantilla institucional; la memoria es la fuente, no el contenido final. Esto aplica tanto
  al construir el JSON de `ficha-modelo/` como a la plantilla de metadatos.
- Si `verification` ya se ejecutó, incluir en `metricas` (ficha técnica) el resultado real del
  golden dataset además de las métricas que reporta la memoria — son dos evidencias distintas y
  ambas importan: "métricas declaradas por IA" vs. "resultado verificado tras integración". Usa
  dos entradas de `metricas[]` si hace falta para no perder ninguna de las dos.
- Cualquier cifra de negocio (ahorro %, KPI) que provenga de datos sintéticos debe llevar la
  misma advertencia que ya trae la memoria ("requiere validación con datos reales antes de
  despliegue") — no se elimina al resumir. En la ficha funcional esto va típicamente en
  `puede_equivocarse` o como una entrada más de `limitaciones[]`.

## Salida

```
outputs/<model_id>/
├── <model_id>_metadatos.docx
├── <model_id>_ficha_tecnica.docx
└── <model_id>_ficha_funcional.docx
```
