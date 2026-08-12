from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from scripts.audit_cu28_repository_hardening import (
    OFFICIAL_BRANCH,
    OFFICIAL_REFERENCE_DATE,
    OFFICIAL_RUN_ID,
    SUMMARY_WHITELIST,
    run_audit,
)
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def _read_json(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


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


def test_repository_hardening_audit_passes() -> None:
    ensure_repro_smoke_pipeline()
    checks, _ = run_audit()
    failed = [f"{check.name}: {check.detail}" for check in checks if not check.passed]
    assert failed == []


def test_no_legacy_snapshots_or_stale_result_files_are_tracked() -> None:
    tracked = _tracked_files()
    prohibited = [
        path
        for path in tracked
        if (
            path.startswith("legacy/")
            or "legacy/metrics_snapshots/" in path
            or "20260604" in path
            or "20260331" in path
            or re.search(r"models/metrics/summary/.+_comparison\.(csv|json)$", path)
            or re.search(
                r"models/metrics/policy_simulation_.+_(period|scenario)_policy_metrics\.csv$",
                path,
            )
        )
        and not path.startswith("reports/audit/")
    ]
    assert prohibited == []


def test_summary_directory_has_only_the_official_family() -> None:
    actual = {
        path.name
        for path in (REPO_ROOT / "models/metrics/summary").iterdir()
        if path.is_file()
    }
    assert actual == SUMMARY_WHITELIST


def test_exactly_one_official_mixed_context_metrics_report_exists() -> None:
    reports = sorted(
        (REPO_ROOT / "reports/official").glob(
            "cu28_metrics_official__mixed_context*.md"
        )
    )
    assert [path.name for path in reports] == [
        "cu28_metrics_official__mixed_context.md"
    ]


def test_latest_artifacts_match_end_to_end_manifest() -> None:
    ensure_repro_smoke_pipeline()
    manifest = _read_json("models/artifacts/model_manifest__mixed_context.json")
    assert manifest["scope"] == "mixed_context"
    assert manifest["official_run"] == {
        "id": OFFICIAL_RUN_ID,
        "kind": "end_to_end",
        "publish_latest": True,
        "reference_date": OFFICIAL_REFERENCE_DATE,
        "mode": "smoke",
    }
    for entry in [*manifest["artifacts"], *manifest["summary_metrics"]]:
        path = REPO_ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]


def test_official_report_values_come_from_official_json() -> None:
    ensure_repro_smoke_pipeline()
    metrics = _read_json("models/metrics/summary/metrics_summary__mixed_context.json")
    report = (
        REPO_ROOT / "reports/official/cu28_metrics_official__mixed_context.md"
    ).read_text(encoding="utf-8")

    trigger = metrics["trigger"]["train"]["confusion_matrix"]
    expected = [
        metrics["upstream"]["baseline_reference_run"]["validation_rmse"],
        metrics["upstream"]["baseline_reference_run"]["test_rmse"],
        metrics["upstream"]["neuroevolution_reference_run"]["validation_rmse"],
        metrics["upstream"]["neuroevolution_reference_run"]["test_rmse"],
        trigger["true_negative"],
        trigger["false_positive"],
        trigger["false_negative"],
        trigger["true_positive"],
        metrics["quantity_optimizer"]["test"]["mae"],
        metrics["quantity_optimizer"]["test"]["rmse"],
        metrics["quantity_optimizer"]["test"]["r2"],
        metrics["policy_simulation"]["baseline_excess_tons"],
        metrics["policy_simulation"]["policy_excess_tons"],
        metrics["policy_simulation"]["aggregate_excess_reduction_pct"],
    ]
    assert [value for value in expected if str(value) not in report] == []


def test_current_docs_do_not_publish_stale_metrics_or_weights() -> None:
    paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "DELIVERY_README_CU28.md",
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
        *sorted((REPO_ROOT / "reports/official").rglob("*.md")),
    ]
    patterns = [
        re.compile(
            r"(validation|test|rmse|linear|neuro)[^\n]{0,100}"
            r"(77[,.]24|98[,.]80|129[,.]43|208[,.]15)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(true_negative[^\n]{0,50}366|false_positive[^\n]{0,50}5\b|"
            r"false_negative[^\n]{0,50}36\b|true_positive[^\n]{0,50}407\b)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(pressure_snapshot_blend_weight[^\n]{0,50}0[,.]22|"
            r"pressure_coverage_gap_blend_weight[^\n]{0,50}0[,.]18)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(reduction|reducci[oó]n|excess|excedente)[^\n]{0,80}"
            r"(21[,.]5\b|37686|37\.686|29569|29\.569)",
            re.IGNORECASE,
        ),
    ]
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in patterns):
            findings.append(path.relative_to(REPO_ROOT).as_posix())
    assert findings == []


def test_docs_do_not_define_independent_metric_values() -> None:
    metric_assignment = re.compile(
        r"(validation_rmse|test_rmse|aggregate_excess_reduction_pct|"
        r"baseline_excess_tons|policy_excess_tons)"
        r"\s*[:=|]\s*`?-?\d",
        re.IGNORECASE,
    )
    findings = []
    for path in [
        REPO_ROOT / "README.md",
        REPO_ROOT / "DELIVERY_README_CU28.md",
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
    ]:
        if metric_assignment.search(path.read_text(encoding="utf-8")):
            findings.append(path.relative_to(REPO_ROOT).as_posix())
    assert findings == []


def test_official_notebooks_are_output_free() -> None:
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 8
    findings = []
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                findings.append(f"{path.name}:{index}")
    assert findings == []


def test_delivery_readme_fixes_branch_scope_and_reference_date() -> None:
    delivery = (REPO_ROOT / "DELIVERY_README_CU28.md").read_text(
        encoding="utf-8"
    )
    assert OFFICIAL_BRANCH in delivery
    assert "mixed_context" in delivery
    assert OFFICIAL_REFERENCE_DATE in delivery
    assert "internal_archive/not_for_delivery/" in delivery
