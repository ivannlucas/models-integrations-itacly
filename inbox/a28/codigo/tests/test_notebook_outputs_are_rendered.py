from __future__ import annotations

from src.utils import read_json
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_notebook_outputs_are_rendered_in_html() -> None:
    ensure_repro_smoke_pipeline()

    execution_summary = read_json(REPO_ROOT / "reports/eda/notebook_execution_summary__mixed_context.json")
    eda_summary = read_json(REPO_ROOT / "reports/eda/eda_summary__mixed_context.json")

    assert len(eda_summary["notebook_html_paths"]) == 8

    for rel_path in eda_summary["notebook_html_paths"]:
        html_text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "<table" in html_text, rel_path
        assert "<img" in html_text or "image/png" in html_text, rel_path

    for notebook_result in execution_summary["notebook_results"]:
        render_stats = notebook_result.get("render_stats", {})
        assert render_stats.get("table_outputs", 0) >= 3, notebook_result["notebook"]
        assert render_stats.get("stream_outputs", 0) >= 1, notebook_result["notebook"]
        if notebook_result.get("figures"):
            assert render_stats.get("visual_outputs", 0) >= 3, notebook_result["notebook"]
