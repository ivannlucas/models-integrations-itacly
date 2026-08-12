from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_platform_run_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "outputs" / "test_demo_run"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "platform_run",
            "--input",
            "data/demo/customer_upload_example.csv",
            "--output",
            "outputs/test_demo_run/",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PURCHASE SUMMARY BY RAW MATERIAL" in completed.stdout
    assert "raw_material_id" in completed.stdout
    assert "destination_profile" in completed.stdout
    assert "total_order_quantity_tons" in completed.stdout
    assert "RECOMMENDED PURCHASES" in completed.stdout
    assert "recommended_action" in completed.stdout
    assert "order_quantity_tons" in completed.stdout
    assert "DATA USE" in completed.stdout
    assert "RM_PORK_SHOULDER_B" in completed.stdout
    assert "fresh_short_shelf_life" in completed.stdout
    assert (output_dir / "validation_report.json").exists()
    assert (output_dir / "recommendations.csv").exists()
    assert (output_dir / "policy_simulation_results.csv").exists()
    assert (output_dir / "summary_metrics.json").exists()
