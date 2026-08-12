from __future__ import annotations

from src.utils import read_json
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_notebook_outputs_exist() -> None:
    ensure_repro_smoke_pipeline()

    execution_summary_path = REPO_ROOT / "reports" / "eda" / "notebook_execution_summary__mixed_context.json"
    eda_summary_json = REPO_ROOT / "reports" / "eda" / "eda_summary__mixed_context.json"
    eda_summary_md = REPO_ROOT / "reports" / "eda" / "eda_summary__mixed_context.md"

    assert execution_summary_path.exists()
    assert eda_summary_json.exists()
    assert eda_summary_md.exists()

    execution_summary = read_json(execution_summary_path)
    eda_summary = read_json(eda_summary_json)

    assert len(execution_summary["executed_notebooks"]) == 8
    assert execution_summary["failed_notebooks"] == []
    assert len(execution_summary["notebook_results"]) == 8

    for notebook_result in execution_summary["notebook_results"]:
        render_stats = notebook_result.get("render_stats", {})
        assert render_stats.get("table_outputs", 0) >= 3, notebook_result["notebook"]
        assert render_stats.get("stream_outputs", 0) >= 1, notebook_result["notebook"]
        if notebook_result.get("figures"):
            assert render_stats.get("visual_outputs", 0) >= 3, notebook_result["notebook"]

    notebook_html_paths = eda_summary["notebook_html_paths"]
    figure_paths = eda_summary["figure_paths"]
    table_paths = eda_summary["table_paths"]

    assert len(notebook_html_paths) >= 8
    assert len(figure_paths) >= 24
    assert len(table_paths) >= 16

    for rel_path in notebook_html_paths:
        assert (REPO_ROOT / rel_path).exists(), rel_path
    for rel_path in figure_paths[:10]:
        assert (REPO_ROOT / rel_path).exists(), rel_path
    for rel_path in table_paths[:10]:
        assert (REPO_ROOT / rel_path).exists(), rel_path
