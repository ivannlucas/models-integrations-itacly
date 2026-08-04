# Métricas oficiales CU28 / mixed_context

Documento generado desde los JSON/CSV oficiales. No copiar métricas desde corridas antiguas.

- Run oficial: `mixed_context_20260518_seed42_smoke`
- Fecha de referencia: `2026-05-18`
- Modo: `smoke`
- Split: `train=812`, `validation=174`, `test=175`

## Predictor upstream

| Modelo | Feature set | Split | n | MAE | RMSE | R² |
|---|---|---|---:|---:|---:|---:|
| linear_regression | ablation_reduced_context | validation | 174 | 47.707647049152065 | 76.47469717064605 | 0.589011676816338 |
| linear_regression | ablation_reduced_context | test | 175 | 72.82239979926281 | 105.67102742272888 | 0.12312950024740887 |
| neuroevolution | ablation_reduced_context | validation | 174 | 94.9272365870896 | 127.61424716008125 | -0.14443801861018368 |
| neuroevolution | ablation_reduced_context | test | 175 | 184.5997901476652 | 213.46256623034822 | -2.578220107110586 |

## Purchase Trigger

| Split | n | Accuracy | Balanced accuracy | Precision BUY | Precision DO_NOT_BUY | Recall BUY | Recall DO_NOT_BUY | F1 BUY | F1 DO_NOT_BUY | FNR | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 812 | 0.9519704433497537 | 0.9534609269903387 | 0.9741176470588235 | 0.9276485788113695 | 0.9366515837104072 | 0.9702702702702702 | 0.9550173010380624 | 0.9484808454425363 | 0.06334841628959276 | 359 | 11 | 28 | 414 |
| validation | 174 | 0.9367816091954023 | 0.9198387096774194 | 0.952 | 0.8979591836734694 | 0.9596774193548387 | 0.88 | 0.9558232931726908 | 0.888888888888889 | 0.04032258064516129 | 44 | 6 | 5 | 119 |
| test | 175 | 0.96 | 0.696969696969697 | 0.9647058823529412 | 0.8 | 0.9939393939393939 | 0.4 | 0.9791044776119404 | 0.5333333333333333 | 0.006060606060606061 | 4 | 6 | 1 | 164 |

## Quantity Optimizer

Target: `quantity_optimizer_target_tons`. Baseline funcional: `baseline_order_quantity_tons`.

| Split | n | MAE | RMSE | R² | Baseline MAE | Baseline RMSE | Baseline R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 442 | 3.9994645620436904 | 6.63576813977863 | 0.9997257534492018 | 4.93507267051678 | 12.065701140216483 | 0.9990932994146666 |
| validation | 124 | 12.900121645653218 | 19.703367623362663 | 0.9992717128955891 | 16.420628261821772 | 25.03340956316879 | 0.9988243942306193 |
| test | 165 | 31.064144337979435 | 37.0957340841414 | 0.9978737931736812 | 31.679930543505172 | 36.30726355666127 | 0.9979632177372101 |

### Comparacion supervisada DummyRegressor vs Ridge

Target: `quantity_optimizer_target_tons`. Filtro: `purchase_trigger_label == 1`. Test se usa solo para evaluacion final.

| Modelo | Split | n_rows | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|
| DummyRegressor | train | 442 | 272.0314738724874 | 400.7011749302037 | 0.0 |
| Ridge | train | 442 | 3.9994645620436904 | 6.63576813977863 | 0.9997257534492018 |
| DummyRegressor | validation | 124 | 536.3048285513383 | 824.1777654783837 | -0.27427595210661004 |
| Ridge | validation | 124 | 12.900121645653218 | 19.703367623362663 | 0.9992717128955891 |
| DummyRegressor | test | 165 | 901.0212475737835 | 1189.0638023229546 | -1.1845796593242532 |
| Ridge | test | 165 | 31.064144337979435 | 37.0957340841414 | 0.9978737931736812 |

## Policy simulation

- Periodos evaluados: `175`
- Excedente agregado baseline: `37374.49442683941`
- Excedente agregado política: `29260.17626877957`
- Reducción absoluta: `8114.318158059839`
- Reducción porcentual: `21.710843939156476`
- Stockout agregado baseline: `0.0`
- Stockout agregado política: `0.0`
- Guardrail: `{'name': 'bounded_stockout_increase', 'allowed_stockout_increase_pct': 5.0}`
- Guardrail superado: `True`

Fuente: `models/metrics/summary/*__mixed_context.json` y `models/metrics/official/*__mixed_context.csv`.
