# CU28 doc-code metrics alignment

- Scope: `mixed_context`
- Reference date: `2026-05-18`
- Status: `PASS`
- Checks: `40/40`

## Checks

| Check | Status | Detail |
|---|---|---|
| upstream linear_regression validation: predictions vs run JSON | PASS | match |
| upstream linear_regression validation: run JSON vs summary | PASS | match |
| upstream linear_regression test: predictions vs run JSON | PASS | match |
| upstream linear_regression test: run JSON vs summary | PASS | match |
| upstream neuroevolution validation: predictions vs run JSON | PASS | match |
| upstream neuroevolution validation: run JSON vs summary | PASS | match |
| upstream neuroevolution test: predictions vs run JSON | PASS | match |
| upstream neuroevolution test: run JSON vs summary | PASS | match |
| purchase trigger train: predictions vs summary JSON | PASS | match |
| purchase trigger train: confusion matrix sums to n | PASS | actual_buy=442 actual_do_not_buy=370 n=812 |
| purchase trigger train: BUY pct from counts | PASS | buy_pct=0.5443349753694581 positive_rate_actual=0.5443349753694581 |
| purchase trigger validation: predictions vs summary JSON | PASS | match |
| purchase trigger validation: confusion matrix sums to n | PASS | actual_buy=124 actual_do_not_buy=50 n=174 |
| purchase trigger validation: BUY pct from counts | PASS | buy_pct=0.7126436781609196 positive_rate_actual=0.7126436781609196 |
| purchase trigger test: predictions vs summary JSON | PASS | match |
| purchase trigger test: confusion matrix sums to n | PASS | actual_buy=165 actual_do_not_buy=10 n=175 |
| purchase trigger test: BUY pct from counts | PASS | buy_pct=0.9428571428571428 positive_rate_actual=0.9428571428571428 |
| quantity optimizer train: predictions vs summary JSON | PASS | match |
| quantity optimizer train: baseline_order_quantity_tons vs summary JSON | PASS | match |
| quantity optimizer validation: predictions vs summary JSON | PASS | match |
| quantity optimizer validation: baseline_order_quantity_tons vs summary JSON | PASS | match |
| quantity optimizer test: predictions vs summary JSON | PASS | match |
| quantity optimizer test: baseline_order_quantity_tons vs summary JSON | PASS | match |
| quantity optimizer supervised comparison contains DummyRegressor and Ridge on all splits | PASS | pairs=[('DummyRegressor', 'test'), ('DummyRegressor', 'train'), ('DummyRegressor', 'validation'), ('Ridge', 'test'), ('Ridge', 'train'), ('Ridge', 'validation')] |
| quantity optimizer supervised comparison excludes target and trigger label from features | PASS | feature_count=24 |
| quantity optimizer supervised DummyRegressor train: recalculated vs comparison JSON | PASS | match |
| quantity optimizer supervised Ridge train: recalculated vs comparison JSON | PASS | match |
| quantity optimizer supervised DummyRegressor validation: recalculated vs comparison JSON | PASS | match |
| quantity optimizer supervised Ridge validation: recalculated vs comparison JSON | PASS | match |
| quantity optimizer supervised DummyRegressor test: recalculated vs comparison JSON | PASS | match |
| quantity optimizer supervised Ridge test: recalculated vs comparison JSON | PASS | match |
| policy simulation: period CSV vs summary JSON | PASS | match |
| formula canonical_variant: config vs effective code parameters | PASS | config='pressure' effective='pressure' |
| formula canonical_variant: config vs generated formula report | PASS | config='pressure' report='pressure' |
| formula pressure_snapshot_blend_weight: config vs effective code parameters | PASS | config=0.08 effective=0.08 |
| formula pressure_snapshot_blend_weight: config vs generated formula report | PASS | config=0.08 report=0.08 |
| formula pressure_coverage_gap_blend_weight: config vs effective code parameters | PASS | config=0.06 effective=0.06 |
| formula pressure_coverage_gap_blend_weight: config vs generated formula report | PASS | config=0.06 report=0.06 |
| documentation scan: no known stale metrics in README/docs/reports/official | PASS | none |
| metrics summary exposes quantity optimizer DummyRegressor comparison | PASS | target='quantity_optimizer_target_tons' |

## Upstream Predictor

| Model | Split | n | MAE | RMSE | R2 | MAPE | Source |
|---|---|---:|---:|---:|---:|---:|---|
| linear_regression | validation | 174 | 47.707647049152065 | 76.47469717064605 | 0.589011676816338 | 0.16698156168532938 | models/metrics/summary/baseline_comparison_latest__mixed_context.json; models/metrics/baseline_comparison_mixed_context_20260518_seed42_smoke__synthetic_procurement_need__linear_regression__ablation_reduced_context.json |
| linear_regression | test | 175 | 72.82239979926281 | 105.67102742272888 | 0.12312950024740887 | 0.1600149297070126 | models/metrics/summary/baseline_comparison_latest__mixed_context.json; models/metrics/baseline_comparison_mixed_context_20260518_seed42_smoke__synthetic_procurement_need__linear_regression__ablation_reduced_context.json |
| neuroevolution | validation | 174 | 94.9272365870896 | 127.61424716008125 | -0.14443801861018368 | 0.33718344562232344 | models/metrics/summary/baseline_comparison_latest__mixed_context.json; models/metrics/baseline_comparison_mixed_context_20260518_seed42_smoke__synthetic_procurement_need__neuroevolution__ablation_reduced_context.json |
| neuroevolution | test | 175 | 184.5997901476652 | 213.46256623034822 | -2.578220107110586 | 0.38353199703633345 | models/metrics/summary/baseline_comparison_latest__mixed_context.json; models/metrics/baseline_comparison_mixed_context_20260518_seed42_smoke__synthetic_procurement_need__neuroevolution__ablation_reduced_context.json |

## Purchase Trigger

| Split | n | BUY pct | Accuracy | Balanced accuracy | Recall BUY | FNR BUY | Recall DO_NOT_BUY | Precision DO_NOT_BUY | F1 DO_NOT_BUY | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 812 | 0.5443349753694581 | 0.9519704433497537 | 0.9534609269903387 | 0.9366515837104072 | 0.06334841628959276 | 0.9702702702702702 | 0.9276485788113695 | 0.9484808454425363 | 359 | 11 | 28 | 414 |
| validation | 174 | 0.7126436781609196 | 0.9367816091954023 | 0.9198387096774194 | 0.9596774193548387 | 0.04032258064516129 | 0.88 | 0.8979591836734694 | 0.888888888888889 | 44 | 6 | 5 | 119 |
| test | 175 | 0.9428571428571428 | 0.96 | 0.696969696969697 | 0.9939393939393939 | 0.006060606060606061 | 0.4 | 0.8 | 0.5333333333333333 | 4 | 6 | 1 | 164 |

## Quantity Optimizer

| Model | Split | n | MAE | RMSE | R2 | Baseline MAE | Baseline RMSE | Baseline R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | train | 442 | 3.9994645620436904 | 6.63576813977863 | 0.9997257534492018 | 4.93507267051678 | 12.065701140216483 | 0.9990932994146666 |
| Ridge | validation | 124 | 12.900121645653218 | 19.703367623362663 | 0.9992717128955891 | 16.420628261821772 | 25.03340956316879 | 0.9988243942306193 |
| Ridge | test | 165 | 31.064144337979435 | 37.0957340841414 | 0.9978737931736812 | 31.679930543505172 | 36.30726355666127 | 0.9979632177372101 |

## Quantity Optimizer Dummy Comparison

| Model | Split | RMSE | MAE | R2 | n_rows |
|---|---|---:|---:|---:|---:|
| DummyRegressor | train | 400.7011749302037 | 272.0314738724874 | 0.0 | 442 |
| Ridge | train | 6.63576813977863 | 3.9994645620436904 | 0.9997257534492018 | 442 |
| DummyRegressor | validation | 824.1777654783837 | 536.3048285513383 | -0.27427595210661004 | 124 |
| Ridge | validation | 19.703367623362663 | 12.900121645653218 | 0.9992717128955891 | 124 |
| DummyRegressor | test | 1189.0638023229546 | 901.0212475737835 | -1.1845796593242532 | 165 |
| Ridge | test | 37.0957340841414 | 31.064144337979435 | 0.9978737931736812 | 165 |

## Policy Simulation

- `baseline_excess_tons`: `37374.49442683941`
- `policy_excess_tons`: `29260.17626877957`
- `absolute_excess_reduction_tons`: `8114.318158059839`
- `aggregate_excess_reduction_pct`: `21.710843939156476`
- `baseline_stockout_tons`: `0.0`
- `policy_stockout_tons`: `0.0`
- `aggregate_stockout_change_pct`: `0.0`
- `stockout_guardrail_pass`: `True`

## synthetic_procurement_need Formula

- `canonical_variant`: `pressure`
- `effective_variant_column`: `synthetic_procurement_need_pressure`
- `pressure_core_weight`: `0.8600000000000001`
- `pressure_snapshot_blend_weight`: `0.08`
- `pressure_coverage_gap_blend_weight`: `0.06`
- `forward_requirement_weights`: `[0.4, 0.3, 0.2, 0.1]`
- `formula_note`: `For canonical_variant=pressure, synthetic_procurement_need is the pressure variant: pressure_core_weight * pressure_core + pressure_snapshot_blend_weight * legacy_snapshot_need + pressure_coverage_gap_blend_weight * synthetic_procurement_need_coverage_gap.`

## Sources

- `baseline_summary_json`: `models/metrics/summary/baseline_comparison_latest__mixed_context.json`
- `trigger_metrics_json`: `models/metrics/summary/trigger_metrics_latest__mixed_context.json`
- `quantity_optimizer_metrics_json`: `models/metrics/summary/quantity_optimizer_latest__mixed_context.json`
- `quantity_optimizer_baseline_comparison_json`: `models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.json`
- `policy_simulation_summary_json`: `models/metrics/summary/policy_simulation_latest__mixed_context.json`
- `official_formula_json`: `reports/official/synthetic_procurement_need_formula__mixed_context.json`
- `metrics_summary_json`: `models/metrics/summary/metrics_summary__mixed_context.json`
- `config`: `config/config.yaml`
