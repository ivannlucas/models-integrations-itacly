# Reporte de consistencia de métricas CU28

- Scope: `mixed_context`
- Fecha de referencia: `2026-05-18`
- Checks superados: `26/26`
- Estado: `PASS`

## Resultado de checks

| Check | Estado | Detalle |
|---|---|---|
| upstream linear_regression validation: predictions vs run JSON | PASS | match |
| upstream linear_regression validation: run JSON vs summary | PASS | match |
| upstream linear_regression test: predictions vs run JSON | PASS | match |
| upstream linear_regression test: run JSON vs summary | PASS | match |
| upstream neuroevolution validation: predictions vs run JSON | PASS | match |
| upstream neuroevolution validation: run JSON vs summary | PASS | match |
| upstream neuroevolution test: predictions vs run JSON | PASS | match |
| upstream neuroevolution test: run JSON vs summary | PASS | match |
| purchase trigger train: scalar metrics | PASS | match |
| purchase trigger train: confusion matrix | PASS | expected={'true_negative': 359, 'false_positive': 11, 'false_negative': 28, 'true_positive': 414} actual={'true_negative': 359, 'false_positive': 11, 'false_negative': 28, 'true_positive': 414} |
| purchase trigger validation: scalar metrics | PASS | match |
| purchase trigger validation: confusion matrix | PASS | expected={'true_negative': 44, 'false_positive': 6, 'false_negative': 5, 'true_positive': 119} actual={'true_negative': 44, 'false_positive': 6, 'false_negative': 5, 'true_positive': 119} |
| purchase trigger test: scalar metrics | PASS | match |
| purchase trigger test: confusion matrix | PASS | expected={'true_negative': 4, 'false_positive': 6, 'false_negative': 1, 'true_positive': 164} actual={'true_negative': 4, 'false_positive': 6, 'false_negative': 1, 'true_positive': 164} |
| quantity optimizer train: model metrics | PASS | match |
| quantity optimizer train: baseline metrics | PASS | match |
| quantity optimizer validation: model metrics | PASS | match |
| quantity optimizer validation: baseline metrics | PASS | match |
| quantity optimizer test: model metrics | PASS | match |
| quantity optimizer test: baseline metrics | PASS | match |
| policy simulation: period CSV vs summary JSON | PASS | match |
| formula code vs config: pressure_snapshot_blend_weight | PASS | config=0.08 code=0.08 |
| formula code vs config: pressure_coverage_gap_blend_weight | PASS | config=0.06 code=0.06 |
| formula generated report vs config: pressure_snapshot_blend_weight | PASS | config=0.08 report=0.08 |
| formula generated report vs config: pressure_coverage_gap_blend_weight | PASS | config=0.06 report=0.06 |
| obsolete values absent from current docs/reports | PASS | none |

## Métricas auditadas

- Linear validation RMSE: `76.47469717064605`
- Linear test RMSE: `105.67102742272888`
- Neuroevolution validation RMSE: `127.61424716008125`
- Neuroevolution test RMSE: `213.46256623034822`
- Trigger train confusion matrix: `{'true_negative': 359, 'false_positive': 11, 'false_negative': 28, 'true_positive': 414}`
- Quantity Optimizer test MAE/RMSE/R²: `31.064144337979435` / `37.0957340841414` / `0.9978737931736812`
- Excedente baseline/política: `37374.49442683941` / `29260.17626877957`
- Reducción porcentual de excedente: `21.710843939156476`

Las métricas vigentes deben copiarse desde los artefactos oficiales regenerados, nunca desde corridas antiguas.
