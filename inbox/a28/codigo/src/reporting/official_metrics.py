"""Generated reports backed only by official mixed_context artifacts."""

from __future__ import annotations

from typing import Any

from src.reproducibility.runtime import official_paths
from src.utils import ensure_directory, read_json, write_json


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _find_run(summary: dict[str, Any], model_family: str, feature_set: str) -> dict[str, Any]:
    for run in summary.get("runs", []):
        if run.get("model_family") == model_family and run.get("feature_set") == feature_set:
            return dict(run)
    raise KeyError(f"Missing official run model_family={model_family} feature_set={feature_set}")


def write_official_reports(config: dict[str, Any]) -> dict[str, str]:
    paths = official_paths(config)
    ensure_directory(paths["official_report_dir"])
    baseline = read_json(paths["baseline_summary_json"])
    trigger = read_json(paths["trigger_metrics_json"])
    quantity = read_json(paths["quantity_optimizer_metrics_json"])
    quantity_baseline_comparison = (
        read_json(paths["quantity_optimizer_baseline_comparison_json"])
        if paths["quantity_optimizer_baseline_comparison_json"].exists()
        else {}
    )
    policy = read_json(paths["policy_simulation_summary_json"])
    manifest = read_json(paths["repro_manifest"]) if paths["repro_manifest"].exists() else {}

    target_parameters = (
        config["synthetic_data"]["simulation_parameters"]["procurement_target_parameters"]
    )
    formula_payload = {
        "scope": "mixed_context",
        "source": "config/config.yaml",
        "config_path": "synthetic_data.simulation_parameters.procurement_target_parameters",
        "canonical_variant": target_parameters["canonical_variant"],
        "pressure_snapshot_blend_weight": float(
            target_parameters["pressure_snapshot_blend_weight"]
        ),
        "pressure_coverage_gap_blend_weight": float(
            target_parameters["pressure_coverage_gap_blend_weight"]
        ),
        "pressure_core_weight": float(
            1.0
            - float(target_parameters["pressure_snapshot_blend_weight"])
            - float(target_parameters["pressure_coverage_gap_blend_weight"])
        ),
        "forward_requirement_weights": [
            float(value) for value in target_parameters["forward_requirement_weights"]
        ],
    }
    write_json(paths["official_formula_json"], formula_payload)
    paths["official_formula_report"].write_text(
        "\n".join(
            [
                "# Fórmula oficial de synthetic_procurement_need",
                "",
                "Documento generado desde `config/config.yaml`. No editar valores manualmente.",
                "",
                f"- `canonical_variant`: `{formula_payload['canonical_variant']}`",
                f"- `pressure_core_weight`: `{_fmt(formula_payload['pressure_core_weight'])}`",
                f"- `pressure_snapshot_blend_weight`: `{_fmt(formula_payload['pressure_snapshot_blend_weight'])}`",
                f"- `pressure_coverage_gap_blend_weight`: `{_fmt(formula_payload['pressure_coverage_gap_blend_weight'])}`",
                f"- `forward_requirement_weights`: `{formula_payload['forward_requirement_weights']}`",
                "",
                "La configuración versionada es la fuente canónica. El código falla si falta alguno de los parámetros requeridos.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    best = dict(baseline["best_baseline_run"])
    linear = _find_run(baseline, "linear_regression", "ablation_reduced_context")
    neuro = _find_run(baseline, "neuroevolution", "ablation_reduced_context")
    lines = [
        "# Métricas oficiales CU28 / mixed_context",
        "",
        "Documento generado desde los JSON/CSV oficiales. No copiar métricas desde corridas antiguas.",
        "",
        f"- Run oficial: `{manifest.get('official_run', {}).get('id', baseline.get('comparison_run_id'))}`",
        f"- Fecha de referencia: `{config['official_release']['reference_date']}`",
        f"- Modo: `{manifest.get('official_run', {}).get('mode', config.get('runtime', {}).get('reproducibility_mode'))}`",
        f"- Split: `train={baseline['split_rows']['train']}`, `validation={baseline['split_rows']['valid']}`, `test={baseline['split_rows']['test']}`",
        "",
        "## Predictor upstream",
        "",
        "| Modelo | Feature set | Split | n | MAE | RMSE | R² |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    upstream_runs = []
    seen_run_ids = set()
    for run in [best, linear, neuro]:
        run_id = run.get("run_id")
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        upstream_runs.append(run)
    for run in upstream_runs:
        for split in ["validation", "test"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(run["model_family"]),
                        str(run["feature_set"]),
                        split,
                        _fmt(run[f"{split}_n_samples"]),
                        _fmt(run[f"{split}_mae"]),
                        _fmt(run[f"{split}_rmse"]),
                        _fmt(run[f"{split}_r2"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Purchase Trigger",
            "",
            "| Split | n | Accuracy | Balanced accuracy | Precision BUY | Precision DO_NOT_BUY | Recall BUY | Recall DO_NOT_BUY | F1 BUY | F1 DO_NOT_BUY | FNR | TN | FP | FN | TP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ["train", "validation", "test"]:
        metrics = trigger[split]
        confusion = metrics["confusion_matrix"]
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    _fmt(metrics["n_samples"]),
                    _fmt(metrics["accuracy"]),
                    _fmt(metrics["balanced_accuracy"]),
                    _fmt(metrics["precision_by_class"]["BUY"]),
                    _fmt(metrics["precision_by_class"]["DO_NOT_BUY"]),
                    _fmt(metrics["recall_buy"]),
                    _fmt(metrics["recall_do_not_buy"]),
                    _fmt(metrics["f1_buy"]),
                    _fmt(metrics["f1_do_not_buy"]),
                    _fmt(metrics["false_negative_rate"]),
                    _fmt(confusion["true_negative"]),
                    _fmt(confusion["false_positive"]),
                    _fmt(confusion["false_negative"]),
                    _fmt(confusion["true_positive"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Quantity Optimizer",
            "",
            "Target: "
            f"`{quantity['target_column']}`. Baseline funcional: "
            f"`{quantity['baseline_comparison'].get('column') or quantity['baseline_comparison'].get('name')}`.",
            "",
            "| Split | n | MAE | RMSE | R² | Baseline MAE | Baseline RMSE | Baseline R² |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ["train", "validation", "test"]:
        metrics = quantity[split]
        baseline_metrics = metrics["baseline_comparison"]
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    _fmt(metrics["n_samples"]),
                    _fmt(metrics["mae"]),
                    _fmt(metrics["rmse"]),
                    _fmt(metrics["r2"]),
                    _fmt(baseline_metrics["mae"]),
                    _fmt(baseline_metrics["rmse"]),
                    _fmt(baseline_metrics["r2"]),
                ]
            )
            + " |"
        )

    if quantity_baseline_comparison:
        lines.extend(
            [
                "",
                "### Comparacion supervisada DummyRegressor vs Ridge",
                "",
                f"Target: `{quantity_baseline_comparison['target']}`. "
                f"Filtro: `{quantity_baseline_comparison['evaluation_filter']}`. "
                "Test se usa solo para evaluacion final.",
                "",
                "| Modelo | Split | n_rows | MAE | RMSE | R2 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in quantity_baseline_comparison.get("metrics", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(row["model"]),
                        _fmt(row["split"]),
                        _fmt(row["n_rows"]),
                        _fmt(row["mae"]),
                        _fmt(row["rmse"]),
                        _fmt(row["r2"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Policy simulation",
            "",
            f"- Periodos evaluados: `{policy['n_periods']}`",
            f"- Excedente agregado baseline: `{_fmt(policy['baseline_excess_tons'])}`",
            f"- Excedente agregado política: `{_fmt(policy['policy_excess_tons'])}`",
            f"- Reducción absoluta: `{_fmt(policy['absolute_excess_reduction_tons'])}`",
            f"- Reducción porcentual: `{_fmt(policy['aggregate_excess_reduction_pct'])}`",
            f"- Stockout agregado baseline: `{_fmt(policy['baseline_stockout_tons'])}`",
            f"- Stockout agregado política: `{_fmt(policy['policy_stockout_tons'])}`",
            f"- Guardrail: `{policy['guardrail']}`",
            f"- Guardrail superado: `{policy['stockout_guardrail_pass']}`",
            "",
            "Fuente: `models/metrics/summary/*__mixed_context.json` y `models/metrics/official/*__mixed_context.csv`.",
            "",
        ]
    )
    paths["official_metrics_report"].write_text("\n".join(lines), encoding="utf-8")
    return {
        "metrics_report": str(paths["official_metrics_report"]),
        "formula_report": str(paths["official_formula_report"]),
        "formula_json": str(paths["official_formula_json"]),
    }
