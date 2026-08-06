from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.build_data_manifest import OFFICIAL_DOCS
except ImportError:  # pragma: no cover - direct script execution fallback
    from build_data_manifest import OFFICIAL_DOCS


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
REPORT_PATH = REPO_ROOT / "reports" / "audit" / "cu28_docs_minimal_contract_report.md"
OFFICIAL_SCOPE = "mixed_context"

DOCS_WHITELIST = (
    "docs/README.md",
    "docs/reproducibility.md",
    "docs/repository_outputs.md",
    "docs/data_lineage.md",
    "docs/data_sources_registry.md",
    "docs/data_blob_inventory.md",
    "docs/etl_pipeline.md",
    "docs/feature_engineering.md",
    "docs/input_contract.md",
    "docs/output_contract.md",
    "docs/leakage_policy.md",
    "docs/model_card_cu28.md",
    "docs/platform_usage.md",
    "docs/simulation_assumptions.md",
    "docs/simulation_data_basis.md",
)

DOCS_SNAPSHOT = ("README.md", "DELIVERY_README_CU28.md", *DOCS_WHITELIST)

REQUIRED_EVIDENCE = (
    "reports/official/cu28_metrics_official__mixed_context.md",
    "reports/audit/cu28_metrics_consistency_report.md",
    "reports/audit/cu28_doc_metrics_alignment.md",
    "reports/audit/cu28_doc_metrics_alignment.json",
    "reports/audit/cu28_repository_hardening_report.md",
)

LEGACY_NAME_PATTERN = re.compile(
    r"(^|[._-])(legacy|deprecated|old|draft|backup)([._-]|$)",
    re.IGNORECASE,
)

OBSOLETE_PATTERNS = {
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
    "old run or intervention date": re.compile(r"20260604|2026-06-04|20260331"),
}

INDEPENDENT_METRIC_ASSIGNMENT = re.compile(
    r"(validation_rmse|test_rmse|aggregate_excess_reduction_pct|"
    r"aggregate_stockout_change_pct|baseline_excess_tons|policy_excess_tons)"
    r"\s*[:=|]\s*`?-?\d",
    re.IGNORECASE,
)

CONTRADICTION_PATTERNS = {
    "platform_run presented as official route": re.compile(
        r"((ruta|linea|línea)\s+oficial[^\n]{0,100}platform_run|"
        r"platform_run[^\n]{0,100}(ruta|linea|línea)\s+oficial)",
        re.IGNORECASE,
    ),
    "outputs presented as official metrics source": re.compile(
        r"(outputs/[^\n]{0,120}(fuente|origen)[^\n]{0,40}metricas oficiales|"
        r"(fuente|origen)[^\n]{0,40}metricas oficiales[^\n]{0,120}outputs/)",
        re.IGNORECASE,
    ),
    "demo CSV presented as official training data": re.compile(
        r"(customer_upload_example\.csv[^\n]{0,120}"
        r"(dataset|datos)[^\n]{0,40}(oficial|entrenamiento)|"
        r"(dataset|datos)[^\n]{0,40}(oficial|entrenamiento)[^\n]{0,120}"
        r"customer_upload_example\.csv)",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add(checks: list[Check], name: str, passed: bool, detail: str) -> None:
    checks.append(Check(name=name, passed=passed, detail=detail))


def _actual_docs_files() -> set[str]:
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in DOCS_ROOT.rglob("*")
        if path.is_file()
    }


def _read_docs() -> dict[str, str]:
    return {
        relative: (REPO_ROOT / relative).read_text(encoding="utf-8")
        for relative in DOCS_WHITELIST
        if (REPO_ROOT / relative).is_file()
    }


def _check_whitelist(checks: list[Check]) -> None:
    expected = set(DOCS_WHITELIST)
    actual = _actual_docs_files()
    extras = sorted(actual - expected)
    missing = sorted(expected - actual)
    _add(
        checks,
        "docs contiene exactamente la whitelist",
        not extras and not missing,
        "whitelist exacta" if not extras and not missing else f"extras={extras}; missing={missing}",
    )
    _add(
        checks,
        "docs/audit no existe",
        not (DOCS_ROOT / "audit").exists(),
        "ausente" if not (DOCS_ROOT / "audit").exists() else "directorio presente",
    )

    legacy = sorted(
        relative
        for relative in actual
        if any(LEGACY_NAME_PATTERN.search(part) for part in Path(relative).parts)
    )
    _add(
        checks,
        "docs sin nombres legacy/deprecated/old/draft/backup",
        not legacy,
        "sin nombres obsoletos" if not legacy else f"rutas={legacy}",
    )


def _check_required_contract_statements(checks: list[Check], docs: dict[str, str]) -> None:
    requirements = {
        "docs/reproducibility.md": (
            "ruta oficial reproducible de CU28 es `mixed_context`",
            "`platform_run` es solo inferencia batch/offline",
        ),
        "docs/platform_usage.md": (
            "exclusivamente una utilidad batch/offline de inferencia",
            "No es la ruta oficial de",
        ),
        "docs/repository_outputs.md": (
            "`models/metrics/summary/`",
            "`models/metrics/official/`",
            "`reports/official/`",
            "`outputs/` es una zona local y regenerable",
        ),
        "docs/leakage_policy.md": (
            "`purchase_trigger_label` es una etiqueta supervisada",
            "prohibida como input de inferencia",
        ),
        "docs/data_lineage.md": (
            "proxies externos no deben reinterpretarse como historicos reales de",
        ),
        "docs/simulation_data_basis.md": (
            "No constituye validacion industrial final",
        ),
    }
    missing: list[str] = []
    for relative, fragments in requirements.items():
        text = docs.get(relative, "")
        for fragment in fragments:
            if fragment not in text:
                missing.append(f"{relative}: {fragment}")
    _add(
        checks,
        "declaraciones contractuales obligatorias",
        not missing,
        "presentes" if not missing else f"faltan={missing}",
    )


def _check_content(checks: list[Check], docs: dict[str, str]) -> None:
    obsolete: list[str] = []
    independent_metrics: list[str] = []
    contradictions: list[str] = []
    for relative, text in docs.items():
        for label, pattern in OBSOLETE_PATTERNS.items():
            if pattern.search(text):
                obsolete.append(f"{relative}: {label}")
        if INDEPENDENT_METRIC_ASSIGNMENT.search(text):
            independent_metrics.append(relative)
        for label, pattern in CONTRADICTION_PATTERNS.items():
            if pattern.search(text):
                contradictions.append(f"{relative}: {label}")

    _add(
        checks,
        "docs sin resultados o metricas obsoletas hardcodeadas",
        not obsolete and not independent_metrics,
        "sin valores manuales"
        if not obsolete and not independent_metrics
        else f"obsoletos={obsolete}; asignaciones={independent_metrics}",
    )
    _add(
        checks,
        "docs sin contradicciones de ruta oficial",
        not contradictions,
        "mixed_context coherente" if not contradictions else f"hallazgos={contradictions}",
    )


def _check_evidence(checks: list[Check]) -> None:
    missing = [relative for relative in REQUIRED_EVIDENCE if not (REPO_ROOT / relative).is_file()]
    official_files = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "reports" / "official").glob("*")
        if path.is_file()
    )
    audit_files = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "reports" / "audit").glob("*")
        if path.is_file()
    )
    _add(
        checks,
        "reports contiene evidencia oficial y auditorias vigentes",
        not missing and bool(official_files) and bool(audit_files),
        f"official={len(official_files)}; audit={len(audit_files)}"
        if not missing
        else f"ausentes={missing}",
    )


def _check_data_blob_snapshot(checks: list[Check]) -> None:
    expected = set(DOCS_SNAPSHOT)
    declared = set(OFFICIAL_DOCS)
    _add(
        checks,
        "configuracion del data blob usa el snapshot documental minimo",
        declared == expected,
        "snapshot exacto"
        if declared == expected
        else f"extras={sorted(declared - expected)}; missing={sorted(expected - declared)}",
    )

    manifest_path = REPO_ROOT / "data_blob_manifest.json"
    if not manifest_path.is_file():
        _add(checks, "data blob incluye snapshot documental actualizado", False, "manifest ausente")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = set(manifest.get("required_paths", {}).get("docs_snapshot", []))
    entries = {
        entry.get("path"): entry
        for entry in manifest.get("files", {}).get("docs_snapshot", [])
        if entry.get("path")
    }
    mismatches: list[str] = []
    if required != expected:
        mismatches.append(
            f"required extras={sorted(required - expected)} missing={sorted(expected - required)}"
        )
    if set(entries) != expected:
        mismatches.append(
            f"entries extras={sorted(set(entries) - expected)} missing={sorted(expected - set(entries))}"
        )
    for relative in sorted(expected & set(entries)):
        path = REPO_ROOT / relative
        if not path.is_file():
            mismatches.append(f"missing file={relative}")
            continue
        if entries[relative].get("sha256") != _sha256(path):
            mismatches.append(f"stale hash={relative}")
        if entries[relative].get("size_bytes") != path.stat().st_size:
            mismatches.append(f"stale size={relative}")
    _add(
        checks,
        "data blob incluye snapshot documental actualizado",
        not mismatches,
        "rutas y hashes vigentes" if not mismatches else "; ".join(mismatches),
    )


def run_audit() -> tuple[list[Check], str]:
    checks: list[Check] = []
    _check_whitelist(checks)
    docs = _read_docs()
    _check_required_contract_statements(checks, docs)
    _check_content(checks, docs)
    _check_evidence(checks)
    _check_data_blob_snapshot(checks)
    return checks, _render_report(checks)


def _render_report(checks: list[Check]) -> str:
    passed = sum(check.passed for check in checks)
    status = "PASS" if passed == len(checks) else "FAIL"
    lines = [
        "# Auditoria documental minima CU28",
        "",
        f"- Fecha de ejecucion: `{datetime.now(timezone.utc).date().isoformat()}`",
        f"- Scope: `{OFFICIAL_SCOPE}`",
        f"- Checks superados: `{passed}/{len(checks)}`",
        f"- Estado: **{status}**",
        "",
        "## Checks",
        "",
        "| Check | Estado | Detalle |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | "
            f"{check.detail.replace('|', '/')} |"
        )
    lines.extend(["", "## Whitelist", ""])
    lines.extend(f"- `{relative}`" for relative in DOCS_WHITELIST)
    lines.extend(
        [
            "",
            "`docs/` conserva contrato tecnico estable. `reports/` conserva "
            "resultados, auditorias y evidencias generadas.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the minimal CU28 documentation contract.")
    parser.add_argument("--scope", default=OFFICIAL_SCOPE)
    parser.add_argument("--fail-on-extra-docs", action="store_true")
    args = parser.parse_args(argv)
    if args.scope != OFFICIAL_SCOPE:
        raise ValueError("Only scope=mixed_context is supported.")

    checks, report = run_audit()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

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
    return 1 if failed and args.fail_on_extra_docs else 0


if __name__ == "__main__":
    raise SystemExit(main())
