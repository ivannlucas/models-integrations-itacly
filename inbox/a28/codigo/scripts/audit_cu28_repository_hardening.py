from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_PATH = REPO_ROOT / "reports" / "audit" / "cu28_repository_hardening_report.md"
OFFICIAL_SCOPE = "mixed_context"
OFFICIAL_REFERENCE_DATE = "2026-05-18"
OFFICIAL_RUN_ID = "mixed_context_20260518_seed42_smoke"
OFFICIAL_BRANCH = "hardening/cu28-remove-legacy-stale-artifacts"

OFFICIAL_ARTIFACTS = [
    "config/config.yaml",
    "DELIVERY_README_CU28.md",
    "reproducibility_manifest__mixed_context.json",
    "data_blob_manifest.json",
    "models/artifacts/model_manifest__mixed_context.json",
    "models/artifacts/upstream_predictor_latest__mixed_context.pkl",
    "models/artifacts/purchase_trigger_latest__mixed_context.pkl",
    "models/artifacts/quantity_optimizer_latest__mixed_context.pkl",
    "models/metrics/summary/baseline_comparison_latest__mixed_context.csv",
    "models/metrics/summary/baseline_comparison_latest__mixed_context.json",
    "models/metrics/summary/neuroevolution_comparison_latest__mixed_context.csv",
    "models/metrics/summary/neuroevolution_comparison_latest__mixed_context.json",
    "models/metrics/summary/trigger_metrics_latest__mixed_context.json",
    "models/metrics/summary/quantity_optimizer_latest__mixed_context.json",
    "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.csv",
    "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.json",
    "models/metrics/summary/policy_simulation_latest__mixed_context.csv",
    "models/metrics/summary/policy_simulation_latest__mixed_context.json",
    "models/metrics/summary/metrics_summary__mixed_context.csv",
    "models/metrics/summary/metrics_summary__mixed_context.json",
    "models/metrics/official/policy_simulation_period_latest__mixed_context.csv",
    "models/metrics/official/policy_simulation_scenario_latest__mixed_context.csv",
    "models/metrics/official/purchase_trigger_predictions_train__mixed_context.csv",
    "models/metrics/official/purchase_trigger_predictions_validation__mixed_context.csv",
    "models/metrics/official/purchase_trigger_predictions_test__mixed_context.csv",
    "models/metrics/official/quantity_optimizer_predictions_train__mixed_context.csv",
    "models/metrics/official/quantity_optimizer_predictions_validation__mixed_context.csv",
    "models/metrics/official/quantity_optimizer_predictions_test__mixed_context.csv",
    "reports/official/cu28_metrics_official__mixed_context.md",
    "reports/official/synthetic_procurement_need_formula__mixed_context.json",
    "reports/official/synthetic_procurement_need_formula__mixed_context.md",
    "reports/audit/cu28_metrics_consistency_report.md",
    "reports/audit/cu28_doc_metrics_alignment.json",
    "reports/audit/cu28_doc_metrics_alignment.md",
    "scripts/audit_cu28_metrics_consistency.py",
    "scripts/audit_doc_metrics_alignment.py",
    "dist/cu28_data_blob_20260518.manifest.json",
    "dist/cu28_data_blob_20260518.sha256",
]

SUMMARY_WHITELIST = {
    "baseline_comparison_latest__mixed_context.csv",
    "baseline_comparison_latest__mixed_context.json",
    "metrics_summary__mixed_context.csv",
    "metrics_summary__mixed_context.json",
    "neuroevolution_comparison_latest__mixed_context.csv",
    "neuroevolution_comparison_latest__mixed_context.json",
    "policy_simulation_latest__mixed_context.csv",
    "policy_simulation_latest__mixed_context.json",
    "quantity_optimizer_baseline_comparison_latest__mixed_context.csv",
    "quantity_optimizer_baseline_comparison_latest__mixed_context.json",
    "quantity_optimizer_latest__mixed_context.json",
    "trigger_metrics_latest__mixed_context.json",
}

DELETED_GROUPS = [
    ("legacy/deprecated_before_platform_reset/", 430, "eliminado"),
    ("legacy/metrics_snapshots/", 9, "eliminado"),
    ("data/interim/external/legacy_raw_cache/", 7, "eliminado; duplicados SHA-256 de raw oficial"),
    ("artefactos y métricas de modelo run 20260604", 12, "eliminado"),
    ("summaries *_comparison.{csv,json}", 4, "eliminado y generación desactivada"),
    ("policy CSV históricos fuera de official/", 4, "eliminado y generación desactivada"),
    ("dist/cu28_data_blob_20260604.*", 3, "eliminado"),
    ("docs/audit/ manual previo", 5, "eliminado por estar sustituido"),
    ("wrappers scripts/{data_processing,get_stats,policy_simulation,predict,train}.py", 5, "eliminado"),
    ("data/metrics/README.md", 1, "eliminado junto con la ruta histórica vacía"),
]

EXCLUDED_ROUTES = [
    "outputs/",
    "data/interim/external/source_cache/",
    "tmp/",
    "temp/",
    "cache/",
    ".ipynb_checkpoints/",
    "__pycache__/",
    "models/metrics/experiments/",
    "models/artifacts/experiments/",
    "internal_archive/not_for_delivery/",
]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return [path for path in tracked if (REPO_ROOT / path).exists()]


def _add(checks: list[Check], name: str, passed: bool, detail: str) -> None:
    checks.append(Check(name=name, passed=passed, detail=detail))


def _check_required_artifacts(checks: list[Check]) -> None:
    missing = [path for path in OFFICIAL_ARTIFACTS if not (REPO_ROOT / path).is_file()]
    _add(
        checks,
        "artefactos oficiales requeridos",
        not missing,
        "todos presentes" if not missing else f"ausentes={missing}",
    )
    zip_path = REPO_ROOT / "dist" / "cu28_data_blob_20260518.zip"
    _add(
        checks,
        "bundle oficial 20260518",
        zip_path.is_file(),
        str(zip_path.relative_to(REPO_ROOT)) if zip_path.is_file() else "zip ausente",
    )


def _check_canonical_identity(checks: list[Check]) -> None:
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    reproducibility = _read_json("reproducibility_manifest__mixed_context.json")
    model_manifest = _read_json("models/artifacts/model_manifest__mixed_context.json")
    identities = [
        ("config reference date", config.get("official_release", {}).get("reference_date"), OFFICIAL_REFERENCE_DATE),
        ("reproducibility scope", reproducibility.get("scope"), OFFICIAL_SCOPE),
        ("reproducibility reference date", reproducibility.get("reference_date"), OFFICIAL_REFERENCE_DATE),
        ("reproducibility run id", reproducibility.get("official_run", {}).get("id"), OFFICIAL_RUN_ID),
        ("model manifest scope", model_manifest.get("scope"), OFFICIAL_SCOPE),
        ("model manifest run id", model_manifest.get("official_run", {}).get("id"), OFFICIAL_RUN_ID),
        ("model manifest run kind", model_manifest.get("official_run", {}).get("kind"), "end_to_end"),
        ("model manifest publish_latest", model_manifest.get("official_run", {}).get("publish_latest"), True),
    ]
    mismatches = [
        f"{name}: actual={actual!r} expected={expected!r}"
        for name, actual, expected in identities
        if actual != expected
    ]
    _add(checks, "identidad canónica mixed_context", not mismatches, "match" if not mismatches else "; ".join(mismatches))


def _check_latest_manifest(checks: list[Check]) -> None:
    manifest = _read_json("models/artifacts/model_manifest__mixed_context.json")
    mismatches: list[str] = []
    for entry in [*manifest.get("artifacts", []), *manifest.get("summary_metrics", [])]:
        relative = entry.get("path")
        if not relative:
            mismatches.append("entrada sin path")
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            mismatches.append(f"ausente={relative}")
            continue
        if entry.get("sha256") != _sha256(path):
            mismatches.append(f"hash={relative}")
    baseline = _read_json("models/metrics/summary/baseline_comparison_latest__mixed_context.json")
    run_ids = {
        str(baseline.get("comparison_run_id", "")),
        *[str(run.get("comparison_run_id", "")) for run in baseline.get("runs", [])],
    }
    if not run_ids or any(OFFICIAL_RUN_ID not in run_id for run_id in run_ids):
        mismatches.append(f"comparison_run_ids={sorted(run_ids)}")
    _add(
        checks,
        "latest ligado al manifest y corrida oficial",
        not mismatches,
        "hashes y run IDs válidos" if not mismatches else "; ".join(mismatches),
    )


def _check_tracked_structure(checks: list[Check], tracked: list[str]) -> None:
    stale_patterns = [
        re.compile(r"(^|/)legacy(/|$)", re.IGNORECASE),
        re.compile(r"(^|/)(old|backup|archive|tmp|temp)(/|$)", re.IGNORECASE),
        re.compile(r"20260604|20260331"),
        re.compile(r"models/metrics/summary/.+_comparison\.(csv|json)$"),
        re.compile(r"models/metrics/policy_simulation_.+_(period|scenario)_policy_metrics\.csv$"),
        re.compile(r"\.(bak|old|tmp)$", re.IGNORECASE),
    ]
    stale = [
        path
        for path in tracked
        if any(pattern.search(path) for pattern in stale_patterns)
        and not path.startswith("reports/audit/")
    ]
    _add(
        checks,
        "estructura versionada sin legacy/snapshots",
        not stale,
        "sin rutas obsoletas" if not stale else f"rutas={stale}",
    )

    summary_files = {
        path.name
        for path in (REPO_ROOT / "models" / "metrics" / "summary").iterdir()
        if path.is_file()
    }
    extras = sorted(summary_files - SUMMARY_WHITELIST)
    missing = sorted(SUMMARY_WHITELIST - summary_files)
    _add(
        checks,
        "familia única de summaries oficiales",
        not extras and not missing,
        "whitelist exacta" if not extras and not missing else f"extras={extras}; missing={missing}",
    )

    dist_sidecars = [
        path for path in tracked if path.startswith("dist/cu28_data_blob_") and not path.endswith(".zip")
    ]
    wrong_dist = [path for path in dist_sidecars if "20260518" not in path]
    _add(
        checks,
        "dist contiene solo sidecars 20260518",
        not wrong_dist,
        f"sidecars={dist_sidecars}" if not wrong_dist else f"obsoletos={wrong_dist}",
    )


def _iter_current_text_files() -> Iterable[Path]:
    roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "DELIVERY_README_CU28.md",
        REPO_ROOT / "docs",
        REPO_ROOT / "reports" / "official",
    ]
    for root in roots:
        if root.is_file():
            yield root
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".yaml", ".yml"}:
                yield path


def _check_stale_content(checks: list[Check]) -> None:
    patterns = {
        "old regression metrics": re.compile(
            r"(validation|test|rmse|linear|neuro)[^\n]{0,100}"
            r"(77[,.]24|98[,.]80|129[,.]43|208[,.]15)",
            re.IGNORECASE,
        ),
        "old trigger matrix": re.compile(
            r"(true_negative[^\n]{0,50}366|false_positive[^\n]{0,50}5\b|"
            r"false_negative[^\n]{0,50}36\b|true_positive[^\n]{0,50}407\b)",
            re.IGNORECASE,
        ),
        "old formula weights": re.compile(
            r"(pressure_snapshot_blend_weight[^\n]{0,50}0[,.]22|"
            r"pressure_coverage_gap_blend_weight[^\n]{0,50}0[,.]18)",
            re.IGNORECASE,
        ),
        "old policy metrics": re.compile(
            r"(reduction|reducci[oó]n|excess|excedente)[^\n]{0,80}"
            r"(21[,.]5\b|37686|37\.686|29569|29\.569)",
            re.IGNORECASE,
        ),
        "old run/date": re.compile(r"20260604|2026-06-04|20260331"),
    }
    findings: list[str] = []
    for path in _iter_current_text_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {label}")
    _add(
        checks,
        "contenido vigente sin valores obsoletos",
        not findings,
        "sin coincidencias contextuales" if not findings else "; ".join(findings),
    )


def _check_notebooks(checks: list[Check]) -> None:
    findings: list[str] = []
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs"):
                findings.append(f"{path.name}: cell {index} contiene outputs")
            if cell.get("execution_count") is not None:
                findings.append(f"{path.name}: cell {index} execution_count no nulo")
    _add(
        checks,
        "notebooks oficiales sin outputs embebidos",
        len(notebooks) == 8 and not findings,
        f"notebooks={len(notebooks)}; outputs=0" if len(notebooks) == 8 and not findings else f"notebooks={len(notebooks)}; {findings}",
    )


def _check_official_report(checks: list[Check]) -> None:
    reports = sorted((REPO_ROOT / "reports" / "official").glob("cu28_metrics_official__mixed_context*.md"))
    _add(
        checks,
        "un único informe oficial de métricas",
        len(reports) == 1,
        f"reports={[path.name for path in reports]}",
    )
    if len(reports) != 1:
        return

    metrics = _read_json("models/metrics/summary/metrics_summary__mixed_context.json")
    formula = _read_json("reports/official/synthetic_procurement_need_formula__mixed_context.json")
    report_text = reports[0].read_text(encoding="utf-8")
    trigger_matrix = metrics["trigger"]["train"]["confusion_matrix"]
    expected_values = [
        metrics["upstream"]["baseline_reference_run"]["validation_rmse"],
        metrics["upstream"]["baseline_reference_run"]["test_rmse"],
        metrics["upstream"]["neuroevolution_reference_run"]["validation_rmse"],
        metrics["upstream"]["neuroevolution_reference_run"]["test_rmse"],
        trigger_matrix["true_negative"],
        trigger_matrix["false_positive"],
        trigger_matrix["false_negative"],
        trigger_matrix["true_positive"],
        metrics["quantity_optimizer"]["test"]["mae"],
        metrics["quantity_optimizer"]["test"]["rmse"],
        metrics["quantity_optimizer"]["test"]["r2"],
        metrics["policy_simulation"]["baseline_excess_tons"],
        metrics["policy_simulation"]["policy_excess_tons"],
        metrics["policy_simulation"]["aggregate_excess_reduction_pct"],
        formula["pressure_snapshot_blend_weight"],
        formula["pressure_coverage_gap_blend_weight"],
    ]
    missing = [str(value) for value in expected_values if str(value) not in report_text and str(value) not in json.dumps(formula)]
    _add(
        checks,
        "informe oficial deriva de JSON oficiales",
        not missing,
        "valores oficiales presentes" if not missing else f"valores ausentes={missing}",
    )


def _check_delivery_contract(checks: list[Check]) -> None:
    text = (REPO_ROOT / "DELIVERY_README_CU28.md").read_text(encoding="utf-8")
    required_fragments = [
        OFFICIAL_BRANCH,
        OFFICIAL_SCOPE,
        OFFICIAL_REFERENCE_DATE,
        "scripts\\reproduce_mixed_context.py",
        "models/metrics/official/",
        "models/metrics/summary/",
        "reports/official/",
        "internal_archive/not_for_delivery/",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    _add(
        checks,
        "ruta oficial documentada inequívocamente",
        not missing,
        "contrato completo" if not missing else f"faltan={missing}",
    )

    manifest = _read_json("data_blob_manifest.json")
    docs = set(manifest.get("required_paths", {}).get("docs_snapshot", []))
    _add(
        checks,
        "README de entrega incluido en data blob",
        "DELIVERY_README_CU28.md" in docs,
        f"docs_snapshot_count={len(docs)}",
    )

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing_exclusions = [route for route in EXCLUDED_ROUTES if route not in gitignore]
    _add(
        checks,
        "rutas no entregables bloqueadas",
        not missing_exclusions,
        "exclusiones presentes" if not missing_exclusions else f"faltan={missing_exclusions}",
    )


def _run_metrics_audit(checks: list[Check]) -> None:
    from scripts import audit_cu28_metrics_consistency

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = audit_cu28_metrics_consistency.main(
            ["--scope", OFFICIAL_SCOPE, "--fail-on-mismatch"]
        )
    detail = output.getvalue().strip().replace("\n", " ")
    _add(
        checks,
        "auditoría de consistencia de métricas",
        result == 0,
        "PASS" if result == 0 else detail[-500:],
    )


def _render_report(checks: list[Check], tracked: list[str]) -> str:
    passed = sum(check.passed for check in checks)
    status = "PASS" if passed == len(checks) else "FAIL"
    lines = [
        "# Auditoría de hardening del repositorio CU28",
        "",
        f"- Fecha de ejecución: `{datetime.now(timezone.utc).date().isoformat()}`",
        f"- Rama objetivo: `{OFFICIAL_BRANCH}`",
        f"- Scope: `{OFFICIAL_SCOPE}`",
        f"- Fecha de referencia: `{OFFICIAL_REFERENCE_DATE}`",
        f"- Checks superados: `{passed}/{len(checks)}`",
        f"- Resultado final: **{status}**",
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

    lines.extend(["", "## Ficheros eliminados", "", "| Grupo | Cantidad | Tratamiento |", "|---|---:|---|"])
    for group, count, treatment in DELETED_GROUPS:
        lines.append(f"| `{group}` | {count} | {treatment} |")

    lines.extend(
        [
            "",
            "## Ficheros movidos a internal_archive/not_for_delivery",
            "",
            "- Ninguno. No se identificó contenido que justificara conservar una copia no entregable.",
            "",
            "## Ficheros oficiales conservados",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in OFFICIAL_ARTIFACTS)

    lines.extend(
        [
            "",
            "## Valores obsoletos encontrados y tratamiento",
            "",
            "- Métricas de regresión anteriores: eliminadas con snapshots y runs previos.",
            "- Matriz de confusión anterior: eliminada de documentación y snapshots.",
            "- Pesos anteriores de synthetic_procurement_need: eliminados de defaults y documentación vigente.",
            "- Reducción y agregados de policy anteriores: eliminados de reports y summaries históricos.",
            "- Coincidencias numéricas legítimas en datos fila a fila o hiperparámetros no se clasifican como métricas obsoletas.",
            "",
            "## Rutas excluidas de entrega",
            "",
        ]
    )
    lines.extend(f"- `{route}`" for route in EXCLUDED_ROUTES)
    lines.extend(
        [
            "",
            "## Resumen de inventario",
            "",
            f"- Ficheros versionados auditados: `{len(tracked)}`",
            f"- Summary files permitidos: `{len(SUMMARY_WHITELIST)}`",
            f"- Artefactos oficiales requeridos: `{len(OFFICIAL_ARTIFACTS)}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit() -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    tracked = _tracked_files()
    _check_required_artifacts(checks)
    _check_canonical_identity(checks)
    _check_latest_manifest(checks)
    _check_tracked_structure(checks, tracked)
    _check_stale_content(checks)
    _check_notebooks(checks)
    _check_official_report(checks)
    _check_delivery_contract(checks)
    _run_metrics_audit(checks)
    return checks, tracked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit CU28 repository hardening.")
    parser.add_argument("--scope", default=OFFICIAL_SCOPE)
    parser.add_argument("--fail-on-stale", action="store_true")
    args = parser.parse_args(argv)
    if args.scope != OFFICIAL_SCOPE:
        raise ValueError("Only scope=mixed_context is supported.")

    checks, tracked = run_audit()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(checks, tracked), encoding="utf-8")
    failed = [check for check in checks if not check.passed]
    print(
        json.dumps(
            {
                "scope": args.scope,
                "checks": len(checks),
                "passed": len(checks) - len(failed),
                "failed": len(failed),
                "report": str(REPORT_PATH),
                "failures": [
                    {"check": check.name, "detail": check.detail}
                    for check in failed
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed and args.fail_on_stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
