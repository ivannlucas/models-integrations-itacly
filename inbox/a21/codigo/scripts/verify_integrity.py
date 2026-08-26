"""
Verificacion de integridad de artefactos DATAGIA-21.
Ejecutar desde la raiz del repo: python -m scripts.verify_integrity
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "models" / "artifacts" / "model_metadata.json"
FULL_REPORT_PATH = ROOT / "models" / "metrics" / "full_report.json"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"

EXPECTED_SPLITS = [
    f"{split}_h{h}_{task}.csv"
    for split in ("train", "test")
    for h in (1, 2, 3)
    for task in ("regresion", "clasificacion")
]

EXPECTED_JOBLIBS = [
    f"datagia_best_h{h}_{task}.joblib"
    for h in (1, 2, 3)
    for task in ("reg", "clf")
]

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def _sha256_normalized(path: Path) -> str:
    """SHA-256 con CRLF normalizado a LF.

    Misma logica que _file_sha256 en get_stats.py: garantiza el mismo
    resultado en Windows (autocrlf=true) y Linux/Mac.
    """
    with path.open("rb") as f:
        content = f.read()
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def _stable_metadata_digest(metadata: dict) -> str:
    def to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_jsonable(v) for v in obj]
        return obj

    clean = dict(metadata)
    clean.pop("evidence_manifest_sha256", None)
    payload = json.dumps(to_jsonable(clean), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", str(path.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _git_modified(path: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", str(path.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return ok


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def main() -> int:
    failures = 0

    print("\n" + "=" * 60)
    print("  DATAGIA-21 -- Verificacion de integridad de artefactos")
    print("=" * 60)

    section("1. Existencia de archivos clave")
    for path in [METADATA_PATH, FULL_REPORT_PATH]:
        ok = path.exists()
        if not check(path.relative_to(ROOT).as_posix(), ok):
            failures += 1

    section("2. Splits (12 CSV esperados)")
    for name in EXPECTED_SPLITS:
        p = SPLITS_DIR / name
        if not check(name, p.exists()):
            failures += 1

    section("3. Modelos serializados (6 .joblib esperados)")
    for name in EXPECTED_JOBLIBS:
        p = ARTIFACTS_DIR / name
        if not check(name, p.exists(),
                     f"{p.stat().st_size / 1024 / 1024:.2f} MB" if p.exists() else "NO EXISTE"):
            failures += 1

    section("4. Coherencia de hashes (normalizados LF)")

    if not METADATA_PATH.exists() or not FULL_REPORT_PATH.exists():
        print("  No se pueden verificar hashes: faltan archivos base.")
        return 1

    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    with open(FULL_REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)

    meta_id = meta.get("evidence_bundle_id")
    report_id = report.get("evidence_bundle_id")
    ok = meta_id == report_id
    if not check(
        "evidence_bundle_id coincide en metadata y full_report",
        ok,
        f"metadata={meta_id!r}  report={report_id!r}" if not ok else meta_id,
    ):
        failures += 1

    real_sha = _sha256_normalized(METADATA_PATH)
    declared_sha = report.get("source_model_metadata_sha256", "")
    ok = real_sha == declared_sha
    if not check(
        "SHA-256 (LF) de model_metadata.json == source_model_metadata_sha256 en full_report",
        ok,
        f"\n    real=     {real_sha}\n    declarado={declared_sha}" if not ok else real_sha[:16] + "...",
    ):
        failures += 1

    stored_manifest = meta.get("evidence_manifest_sha256", "")
    recomputed_manifest = _stable_metadata_digest(meta)
    ok = stored_manifest == recomputed_manifest
    if not check(
        "evidence_manifest_sha256 en metadata es internamente coherente",
        ok,
        f"\n    stored=     {stored_manifest}\n    recomputed={recomputed_manifest}" if not ok else stored_manifest[:16] + "...",
    ):
        failures += 1

    section("5. Coherencia de metricas metadata vs full_report")
    mismatches = []
    for h_key in ("H1", "H2", "H3"):
        for task, task_key in [("regression", "regression"), ("classification", "classification")]:
            meta_metrics = (
                meta.get("selected_models", {})
                .get(h_key, {})
                .get(task_key, {})
                .get("metrics_test", {})
            )
            report_test = (
                report.get("horizons", {})
                .get(h_key, {})
                .get(task_key, {})
                .get("test", {})
            )
            for metric_m, metric_r in [("Pearson", "Pearson_R"), ("AUC", "AUC"), ("DA", "DA"), ("MAE", "MAE")]:
                v_m = meta_metrics.get(metric_m)
                v_r = report_test.get(metric_r)
                if v_m is None and v_r is None:
                    continue
                if v_m is None or v_r is None:
                    continue
                if abs(v_m - v_r) > 1e-10:
                    mismatches.append(f"{h_key}/{task}/{metric_m}: metadata={v_m} report={v_r}")

    if mismatches:
        for m in mismatches:
            print(f"  {FAIL}  {m}")
        failures += len(mismatches)
    else:
        print(f"  {PASS}  Todas las metricas coinciden entre metadata y full_report")

    section("6. Estado git de artefactos clave")
    key_files = [
        METADATA_PATH, FULL_REPORT_PATH,
        *[ARTIFACTS_DIR / n for n in EXPECTED_JOBLIBS],
        *[SPLITS_DIR / n for n in EXPECTED_SPLITS],
    ]
    for path in key_files:
        if not path.exists():
            continue
        tracked = _git_tracked(path)
        modified = _git_modified(path) if tracked else False
        short = path.relative_to(ROOT).as_posix()
        if not tracked:
            print(f"  {FAIL}  {short}  (NO esta en git)")
            failures += 1
        elif modified:
            print(f"  {WARN}  {short}  (modificado pero NO commiteado)")
        else:
            print(f"  {PASS}  {short}")

    print("\n" + "=" * 60)
    if failures == 0:
        print("  INTEGRIDAD OK -- todos los checks pasaron")
    else:
        print(f"  {failures} check(s) FALLARON -- revisar antes de entregar")
    print("=" * 60 + "\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
