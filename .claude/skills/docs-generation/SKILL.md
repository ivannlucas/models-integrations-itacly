---
name: docs-generation
description: Usa este skill para generar los 3 documentos institucionales (plantilla de metadatos, ficha técnica, ficha funcional) de un modelo, a partir de su manifest.yaml y la memoria original. Se apoya en el skill docx del sistema para la creación/edición del .docx.
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
`a{número_modelo:02d}_ficha_funcional` — partir siempre de la plantilla corporativa ITACYL
existente (editar, no crear desde cero salvo que no exista plantilla previa para ese tipo de
documento).

## Mapeo memoria → documento

| Documento | Se rellena con (sección de la memoria) |
|---|---|
| **Plantilla de metadatos** | Título/nombre, sector, versión, listado de artefactos y librerías ("Requisitos de Hardware y Entorno"), tipo de modelo |
| **Ficha técnica** | "Arquitectura y Algoritmos", "Entradas y Salidas" (formato físico, estructura), "Hiperparámetros Clave", tabla de métricas de test, requisitos de hardware |
| **Ficha funcional** | "Objetivos" de negocio, KPI y su cumplimiento (tabla de resultados/escenarios), "Modo de Despliegue" y "Mecanismo de Inferencia" en lenguaje no técnico, "Limitaciones, Riesgos y Consideraciones" |

## Reglas

- Nunca reproducir párrafos completos de la memoria tal cual — reescribir/resumir al formato de
  la plantilla institucional; la memoria es la fuente, no el contenido final.
- Si `verification` ya se ejecutó, incluir en la ficha técnica el resultado real del golden
  dataset, no solo las métricas que reporta la memoria — son dos evidencias distintas y ambas
  importan: "métricas declaradas por IA" vs. "resultado verificado tras integración".
- Cualquier cifra de negocio (ahorro %, KPI) que provenga de datos sintéticos debe llevar la
  misma advertencia que ya trae la memoria ("requiere validación con datos reales antes de
  despliegue") — no se elimina al resumir.

## Salida

```
outputs/<model_id>/
├── <model_id>_metadatos.docx
├── <model_id>_ficha_tecnica.docx
└── <model_id>_ficha_funcional.docx
```
