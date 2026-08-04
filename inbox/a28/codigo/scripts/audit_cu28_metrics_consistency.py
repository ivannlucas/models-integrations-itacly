from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing.synthetic_plant import _merge_procurement_target_parameters
from src.evaluation.metrics import (
    compute_policy_metrics,
    compute_regression_metrics,
    compute_trigger_metrics,
)
from src.reproducibility import SMOKE_MODE, build_reproducibility_config
from src.reproducibility.runtime import official_paths
from src.utils import read_json


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return left == right


def _compare_fields(
    checks: list[Check],
    *,
    prefix: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    fields: list[str],
) -> None:
    mismatches = [
        f"{field}: expected={expected.get(field)!r} actual={actual.get(field)!r}"
        for field in fields
        if not _same_number(expected.get(field), actual.get(field))
    ]
    checks.append(
        Check(
            prefix,
            not mismatches,
            "match" if not mismatches else "; ".join(mismatches),
        )
    )


def _find_run(summary: dict[str, Any], model_family: str, feature_set: str) -> dict[str, Any]:
    for record in summary["runs"]:
        if record["model_family"] == model_family and record["feature_set"] == feature_set:
            return record
    raise KeyError(f"Missing run {model_family}/{feature_set}")


def _read_relative_csv(path_value: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / path_value)


def _audit_upstream(checks: list[Check], summary: dict[str, Any]) -> None:
    for model_family in ["linear_regression", "neuroevolution"]:
        record = _find_run(summary, model_family, "ablation_reduced_context")
        payload = read_json(REPO_ROOT / record["metrics_path"])
        for split in ["validation", "test"]:
            predictions = _read_relative_csv(payload["prediction_paths"][split])
            recalculated = compute_regression_metrics(
                predictions["actual"],
                predictions["prediction"],
                split=split,
                model=payload["model_name"],
                target=payload["target_column"],
                include_mape=True,
                include_distribution=True,
            )
            _compare_fields(
                checks,
                prefix=f"upstream {model_family} {split}: predictions vs run JSON",
                expected=payload["metrics"][split],
                actual=recalculated,
                fields=["n_samples", "mae", "rmse", "r2", "mape", "prediction_bias"],
            )
            summary_metrics = {
                key: record[f"{split}_{key}"]
                for key in ["n_samples", "mae", "rmse", "r2", "mape", "prediction_bias"]
            }
            _compare_fields(
                checks,
                prefix=f"upstream {model_family} {split}: run JSON vs summary",
                expected=payload["metrics"][split],
                actual=summary_metrics,
                fields=["n_samples", "mae", "rmse", "r2", "mape", "prediction_bias"],
            )


def _audit_trigger(checks: list[Check], trigger: dict[str, Any]) -> None:
    for split in ["train", "validation", "test"]:
        predictions = _read_relative_csv(trigger["prediction_paths"][split])
        recalculated = compute_trigger_metrics(
            predictions["actual"],
            predictions["prediction"],
            predictions["probability_buy"],
            split=split,
        )
        _compare_fields(
            checks,
            prefix=f"purchase trigger {split}: scalar metrics",
            expected=trigger[split],
            actual=recalculated,
            fields=[
                "n_samples",
                "accuracy",
                "balanced_accuracy",
                "recall_buy",
                "recall_do_not_buy",
                "f1_buy",
                "f1_do_not_buy",
                "false_negative_rate",
            ],
        )
        checks.append(
            Check(
                f"purchase trigger {split}: confusion matrix",
                trigger[split]["confusion_matrix"] == recalculated["confusion_matrix"],
                f"expected={trigger[split]['confusion_matrix']} actual={recalculated['confusion_matrix']}",
            )
        )


def _audit_quantity(checks: list[Check], quantity: dict[str, Any]) -> None:
    for split in ["train", "validation", "test"]:
        predictions = _read_relative_csv(quantity["prediction_paths"][split])
        recalculated = compute_regression_metrics(
            predictions["actual"],
            predictions["prediction"],
            split=split,
            model=quantity["model_family"],
            target=quantity["target_column"],
            include_mape=True,
            include_distribution=True,
        )
        _compare_fields(
            checks,
            prefix=f"quantity optimizer {split}: model metrics",
            expected=quantity[split],
            actual=recalculated,
            fields=["n_samples", "mae", "rmse", "r2", "mape", "prediction_bias"],
        )
        baseline = compute_regression_metrics(
            predictions["actual"],
            predictions["baseline_prediction"],
            split=split,
            model=quantity["baseline_comparison"]["name"],
            target=quantity["target_column"],
            include_mape=True,
            include_distribution=False,
        )
        _compare_fields(
            checks,
            prefix=f"quantity optimizer {split}: baseline metrics",
            expected=quantity[split]["baseline_comparison"],
            actual=baseline,
            fields=["n_samples", "mae", "rmse", "r2", "mape"],
        )


def _audit_policy(
    checks: list[Check],
    config: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    period_df = _read_relative_csv(policy["period_metrics_csv_path"])
    allowed = float(
        config["policy_simulation"]["kpi_definition"]["allowed_stockout_increase_pct"]
    )
    recalculated = compute_policy_metrics(
        period_df,
        guardrail_name="bounded_stockout_increase",
        allowed_stockout_increase_pct=allowed,
    )
    _compare_fields(
        checks,
        prefix="policy simulation: period CSV vs summary JSON",
        expected=policy,
        actual=recalculated,
        fields=[
            "n_periods",
            "baseline_excess_tons",
            "policy_excess_tons",
            "absolute_excess_reduction_tons",
            "aggregate_excess_reduction_pct",
            "baseline_stockout_tons",
            "policy_stockout_tons",
            "aggregate_stockout_change_pct",
            "stockout_guardrail_pass",
        ],
    )


def _audit_formula(
    checks: list[Check],
    config: dict[str, Any],
    paths: dict[str, Any],
) -> None:
    configured = config["synthetic_data"]["simulation_parameters"][
        "procurement_target_parameters"
    ]
    merged = _merge_procurement_target_parameters(
        config["synthetic_data"]["simulation_parameters"]
    )
    for field in [
        "pressure_snapshot_blend_weight",
        "pressure_coverage_gap_blend_weight",
    ]:
        checks.append(
            Check(
                f"formula code vs config: {field}",
                _same_number(configured[field], merged[field]),
                f"config={configured[field]!r} code={merged[field]!r}",
            )
        )

    generated = read_json(paths["official_formula_json"])
    for field in [
        "pressure_snapshot_blend_weight",
        "pressure_coverage_gap_blend_weight",
    ]:
        checks.append(
            Check(
                f"formula generated report vs config: {field}",
                _same_number(configured[field], generated[field]),
                f"config={configured[field]!r} report={generated[field]!r}",
            )
        )


def _audit_obsolete_values(checks: list[Check]) -> None:
    files = [
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
        *sorted((REPO_ROOT / "reports" / "official").rglob("*.md")),
    ]
    patterns = {
        "legacy regression metric": re.compile(
            r"(77[,.]24|98[,.]80|129[,.]43|208[,.]15)"
        ),
        "legacy train confusion matrix": re.compile(
            r"(true_negative[^\n]{0,40}366|false_positive[^\n]{0,40}5\b|"
            r"false_negative[^\n]{0,40}36\b|true_positive[^\n]{0,40}407\b)",
            re.IGNORECASE,
        ),
        "legacy formula weights": re.compile(
            r"(pressure_snapshot_blend_weight[^\n]{0,40}0[,.]22|"
            r"pressure_coverage_gap_blend_weight[^\n]{0,40}0[,.]18)",
            re.IGNORECASE,
        ),
        "legacy policy metrics": re.compile(
            r"(reduction|reducci[oó]n|excess|excedente)[^\n]{0,80}"
            r"(21[,.]5\b|37686|37\.686|29569|29\.569)",
            re.IGNORECASE,
        ),
    }
    findings: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {label}")
    checks.append(
        Check(
            "obsolete values absent from current docs/reports",
            not findings,
            "none" if not findings else "; ".join(findings),
        )
    )


def _render_report(
    checks: list[Check],
    config: dict[str, Any],
    baseline: dict[str, Any],
    trigger: dict[str, Any],
    quantity: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    linear = _find_run(baseline, "linear_regression", "ablation_reduced_context")
    neuro = _find_run(baseline, "neuroevolution", "ablation_reduced_context")
    passed = sum(check.passed for check in checks)
    lines = [
        "# Reporte de consistencia de métricas CU28",
        "",
        f"- Scope: `mixed_context`",
        f"- Fecha de referencia: `{config['official_release']['reference_date']}`",
        f"- Checks superados: `{passed}/{len(checks)}`",
        f"- Estado: `{'PASS' if passed == len(checks) else 'FAIL'}`",
        "",
        "## Resultado de checks",
        "",
        "| Check | Estado | Detalle |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {check.detail.replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Métricas auditadas",
            "",
            f"- Linear validation RMSE: `{linear['validation_rmse']}`",
            f"- Linear test RMSE: `{linear['test_rmse']}`",
            f"- Neuroevolution validation RMSE: `{neuro['validation_rmse']}`",
            f"- Neuroevolution test RMSE: `{neuro['test_rmse']}`",
            f"- Trigger train confusion matrix: `{trigger['train']['confusion_matrix']}`",
            f"- Quantity Optimizer test MAE/RMSE/R²: `{quantity['test']['mae']}` / `{quantity['test']['rmse']}` / `{quantity['test']['r2']}`",
            f"- Excedente baseline/política: `{policy['baseline_excess_tons']}` / `{policy['policy_excess_tons']}`",
            f"- Reducción porcentual de excedente: `{policy['aggregate_excess_reduction_pct']}`",
            "",
            "Las métricas vigentes deben copiarse desde los artefactos oficiales regenerados, nunca desde corridas antiguas.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit CU28 official mixed_context metrics.")
    parser.add_argument("--scope", default="mixed_context")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if args.scope != "mixed_context":
        raise ValueError("Only scope=mixed_context is supported.")

    config = build_reproducibility_config(args.config, mode=SMOKE_MODE)
    paths = official_paths(config)
    required = [
        paths["baseline_summary_json"],
        paths["trigger_metrics_json"],
        paths["quantity_optimizer_metrics_json"],
        paths["policy_simulation_summary_json"],
        paths["official_formula_json"],
        paths["official_metrics_report"],
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing official artifacts:", *missing, sep="\n- ")
        return 2

    baseline = read_json(paths["baseline_summary_json"])
    trigger = read_json(paths["trigger_metrics_json"])
    quantity = read_json(paths["quantity_optimizer_metrics_json"])
    policy = read_json(paths["policy_simulation_summary_json"])
    checks: list[Check] = []
    _audit_upstream(checks, baseline)
    _audit_trigger(checks, trigger)
    _audit_quantity(checks, quantity)
    _audit_policy(checks, config, policy)
    _audit_formula(checks, config, paths)
    _audit_obsolete_values(checks)

    report = _render_report(checks, config, baseline, trigger, quantity, policy)
    paths["audit_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["audit_report"].write_text(report, encoding="utf-8")

    failed = [check for check in checks if not check.passed]
    print(
        json.dumps(
            {
                "scope": args.scope,
                "checks": len(checks),
                "passed": len(checks) - len(failed),
                "failed": len(failed),
                "report": str(paths["audit_report"]),
                "failures": [check.detail for check in failed],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed and args.fail_on_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
