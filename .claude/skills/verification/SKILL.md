---
name: verification
description: Usa este skill para validar que un plugin ya integrado está listo para PR — checklist técnico local (lint, tests, seguridad) y verificación de correctitud contra el golden dataset del manifest. Produce outputs/<model_id>/verification_report.md. Requiere el plugin integrado (skill plugin-integration) y el manifest con golden_cases (skill manifest-extraction).
---

# Verificación pre-PR

## Requisito previo

Plugin integrado en `app/plugins/<nombre>/` (skill `plugin-integration`) y
`inbox/<model_id>/manifest.yaml` con `golden_cases` (skill `manifest-extraction`).

## Parte A — Checklist técnico (wiring)

Reproduce el pipeline de Bitbucket en local, en este orden, y **repite hasta que todos pasen**
antes de continuar a la Parte B. Si algo falla: leer el error, corregir el fichero
correspondiente en `app/plugins/<nombre>/`, repetir.

```bash
flake8 . --extend-exclude=dist,build --show-source --statistics
python -m pytest tests/unit/ --cov=app -v
pylint app/plugins/<nombre>/ --disable=import-error
pip-audit -r requirements.txt

MODEL=<model_id> python main.py &
curl http://localhost:8000/models/<model_id>/health
curl -X POST http://localhost:8000/models/<model_id>/predict \
  -H "Content-Type: application/json" -d '<payload de ejemplo, sacado del manifest>'
curl http://localhost:8000/models/<model_id>/stats
curl -X POST http://localhost:8000/models/<model_id>/train \
  -H "Content-Type: application/json" -d '<data_path apuntando a un CSV con training.required_columns del manifest>'
```

Comportamiento esperado de `/train` según `manifest.training`:
- `supported: true` → 200 con `TrainResponse` conteniendo las métricas de `training.metrics_returned`.
- `supported: false` → 501 (`TrainingNotSupportedError`). Un 501 aquí NO es un fallo del
  checklist si el manifest ya declaraba `supported: false` — sí lo es si el manifest decía
  `true` y el endpoint devuelve 501 (train() quedó sin implementar).

No se pasa a la Parte B con la Parte A en rojo.

## Parte B — Correctitud contra golden dataset

Para cada entrada en `golden_cases` del manifest:

1. Llamar al endpoint real (`/predict` inline o batch, según corresponda) con el `input` del caso.
2. Comparar el resultado contra `expected` con una tolerancia derivada de `metrics_reported`
   (p. ej. tolerancia = 2× el MAE relativo reportado en la memoria; si no hay métrica de
   referencia disponible, usar un 5% por defecto y señalarlo explícitamente en el informe).
3. Registrar cada caso: `id`, `esperado`, `obtenido`, `diferencia`, `¿dentro de tolerancia?`.

**Si algún caso falla la tolerancia**: no ajustar la tolerancia para que pase. Investigar si el
fallo es de wiring (preprocesado distinto al del código original — el sospechoso más común) o
si es un problema real del modelo/artefacto entregado, y documentarlo en el informe. Nunca
marcar un plugin como verificado con casos fallidos silenciados.

## Parte C — Informe de salida

Generar `outputs/<model_id>/verification_report.md`:

```markdown
# Verificación — <model_id>

## Checklist técnico
- [x] flake8: 0 errores
- [x] pytest: X/X passed, cobertura Y%
- [x] pylint: sin issues nuevos
- [x] pip-audit: sin CVEs nuevas
- [x] Arranque local + health + predict + stats: OK
- [x] /train: <200 con métricas OK | 501 esperado (training.supported=false)>

## Correctitud (golden dataset)
| Caso | Esperado | Obtenido | Diferencia | ¿OK? |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

Tolerancia usada: <valor> (basada en <métrica de la memoria | default>)
Resultado: X/Y casos dentro de tolerancia

## Estado final
[ LISTO PARA PR | REQUIERE REVISIÓN — ver detalle arriba ]
```

## Regla de cierre

Este skill nunca abre el PR ni hace merge por sí mismo — deja la rama y el informe listos para
que una persona los revise y decida. El gate humano es siempre el último paso.
