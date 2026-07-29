# Verification report — a31 (ml31-cereals-residue-optimizer) v2.0 LP

**Fecha:** 2026-07-15
**Modelo:** Optimizador de Programación Lineal (PuLP/CBC) para reducción de residuo vegetal cerealista.
**Reemplaza:** el plugin surrogate v1.x (MLPRegressor) previamente integrado, que la memoria v2.0 abandonó.

## Correctitud contra el golden dataset (manifest, 12 casos, tolerancia 0.5%)

Ejecutado el plugin real (`Ml31CerealsResidueOptimizerPlugin`) contra los 12 golden cases del
manifest. Resultado: **12/12 reproducidos dentro de tolerancia.**

| Golden case | Modo | Métrica clave | Esperado | Obtenido | Resultado |
|---|---|---|---|---|---|
| caso_optimize_2023_ref | optimize (min residuo) | total_residue_t | 4 481 067.28 | 4 481 067.28 | 0.000% ✅ |
| " | " | total_benefit_eur | 340 734 196.59 | 340 734 196.59 | 0.000% ✅ |
| " | " | residue_reduction_pct | 31.18 | 31.18 | 0.000% ✅ |
| caso_optimize_2021 | optimize | \|residue_reduction_pct\| | 7.4 | 7.44 | 0.04 pp ✅ (¹) |
| caso_optimize_2022 | optimize | \|residue_reduction_pct\| | 7.3 | 7.30 | 0.00 pp ✅ (¹) |
| caso_pareto_bound_min_residue | optimize | total_residue_t | 3 679 010.51 | 3 679 010.51 | 0.000% ✅ |
| caso_pareto_bound_max_benefit | optimize (max beneficio) | total_benefit_eur | 397 321 163.38 | 397 321 163.38 | 0.000% ✅ |
| " | " | total_residue_t | 5 094 873.39 | 5 094 873.39 | 0.000% ✅ |
| caso_pareto_knee | pareto | knee benefit/residuo/prod | 250.94M / 3.78M / 5.18M | 250.94M / 3.78M / 5.18M | ✅ |
| caso_pareto_p01/p03/p09/p12/p15/p19 | pareto | frontera (19 no dominados) | 19 puntos | 19 puntos | ✅ |

(¹) **Convención de signo documentada:** `residue_reduction_pct` se devuelve POSITIVO cuando hay
reducción (definición del campo en el DTO); la memoria lo tabula NEGATIVO. Misma magnitud, signo
opuesto. No es un error: es la convención declarada del modelo integrado. Anotado en el manifest
(`known_issues`) y en la ficha. NO se ajustó tolerancia para acomodarlo.

## Wiring / contrato

- Tests de wiring: `tests/unit/test_ml31_cereals_residue_optimizer.py` — 6/6 en verde
  (`pytest -p no:flask`). Cubren health, stats, predict optimize, predict pareto, predict batch,
  y train→501.
- Modos expuestos vía unión discriminada `mode`: `optimize` (minimize_residue / maximize_benefit),
  `pareto`, `batch`. `train()` lanza `TrainingNotSupportedError` (→501): es un optimizador
  determinista, no se entrena.
- Excepción de dominio: `InfeasibleOptimizationError` → 422 cuando el LP no tiene solución óptima
  (restricciones incompatibles). Registrada en `extra_predict_exceptions`.
- `mlflow_utils.py` presente (obligatorio por convención del repo).
- Determinismo: LP exacto (Simplex/CBC), sin semillas aleatorias — reproducibilidad exacta.

## Dependencias

- `PuLP==3.3.2` disponible en el entorno (añadida a requirements.txt). Solver CBC incluido.

## Pendiente antes de PR (revisión humana)

1. **Regenerar las fichas de a31**: las fichas actuales en `outputs/a31/` describen el modelo
   surrogate viejo — deben regenerarse para el contrato del optimizador LP (modos optimize/pareto).
2. Confirmar el artefacto de configuración (crop_economics.json, harvest_index.json) que se
   desplegará a S3 en `artifacts/fixed/ml31_cereals_residue_optimizer/`.
3. Revisar la convención de signo de `residue_reduction_pct` (positivo=reducción) — decidir si se
   mantiene o se alinea con la memoria.
4. Identidad del modelo: id/carpeta dicen "neuroevolutivo" pero el v2.0 es LP determinista
   (ni neuroevolución ni GA). Considerar renombrar o documentar de forma prominente.
