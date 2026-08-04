from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repro_smoke_pipeline() -> None:
    required_paths = [
        REPO_ROOT / "reproducibility_manifest__mixed_context.json",
        REPO_ROOT / "data/predictions/predictions_latest__mixed_context.csv",
        REPO_ROOT / "models/metrics/summary/policy_simulation_latest__mixed_context.json",
        REPO_ROOT / "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.json",
        REPO_ROOT / "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.csv",
        REPO_ROOT / "reports/eda/eda_summary__mixed_context.json",
        REPO_ROOT / "dist/cu28_data_blob_20260518.zip",
    ]
    if all(path.exists() for path in required_paths):
        return

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_mixed_context.py",
            "--config",
            "config/config.yaml",
            "--scope",
            "mixed_context",
            "--smoke",
            "--run-notebooks",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
