from __future__ import annotations

import argparse
import html
import io
import json
import os
import sys
import builtins
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reproducibility.eda import build_eda_summary
from src.reproducibility.runtime import repo_root
from src.utils import ensure_directory, write_json


NOTEBOOK_ORDER = [
    "00_data_sources_audit.ipynb",
    "01_raw_data_profile.ipynb",
    "02_external_context_eda.ipynb",
    "03_synthetic_plant_layer_eda.ipynb",
    "04_feature_engineering_audit.ipynb",
    "05_modeling_dataset_eda.ipynb",
    "06_split_validation_and_leakage_audit.ipynb",
    "07_training_and_policy_results_eda.ipynb",
]


def _render_markdown(source: str) -> str:
    escaped = html.escape(source)
    return "<section class='markdown'><pre>{}</pre></section>".format(escaped)


def _render_code(source: str, rendered_outputs: list[str]) -> str:
    output_html = "".join(rendered_outputs) or "<pre></pre>"
    return (
        "<section class='code'><h3>Code</h3><pre>{code}</pre><h3>Output</h3>{output}</section>".format(
            code=html.escape(source),
            output=output_html,
        )
    )


def _display_to_text(value: Any) -> str:
    if value is None:
        return "None"
    to_markdown = getattr(value, "to_markdown", None)
    if callable(to_markdown):
        try:
            return to_markdown(index=False)
        except Exception:
            try:
                return to_markdown()
            except Exception:
                pass
    to_string = getattr(value, "to_string", None)
    if callable(to_string):
        try:
            return to_string()
        except Exception:
            pass
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return repr(value)
    return str(value)


def _render_stream_output(output: str) -> str:
    return f"<pre>{html.escape(output)}</pre>"


def _relative_output_asset(asset_path: str | Path, html_dir: Path) -> str:
    path = Path(asset_path).resolve()
    return Path(os.path.relpath(path, start=html_dir.resolve())).as_posix()


def _serialize_display_object(value: Any, html_dir: Path) -> tuple[str, str]:
    class_name = value.__class__.__name__
    module_name = value.__class__.__module__

    if class_name == "DataFrame":
        return "table", value.to_html(index=False, border=1)
    if class_name == "Series":
        return "table", value.to_frame().to_html(border=1)

    filename = getattr(value, "filename", None)
    if class_name == "Image" and filename:
        src = _relative_output_asset(filename, html_dir)
        alt = html.escape(Path(filename).name)
        return "image", f"<img src='{src}' alt='{alt}' style='max-width:100%;height:auto;' />"

    repr_html = getattr(value, "_repr_html_", None)
    if callable(repr_html):
        try:
            return "html", repr_html()
        except Exception:
            pass

    if module_name.startswith("pandas"):
        return "table", f"<pre>{html.escape(_display_to_text(value))}</pre>"

    return "text", f"<pre>{html.escape(_display_to_text(value))}</pre>"


def _execute_notebook(notebook_path: Path, *, scope: str, html_dir: Path) -> dict[str, Any]:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    display_items: list[tuple[str, str]] = []
    render_stats = {
        "table_outputs": 0,
        "visual_outputs": 0,
        "stream_outputs": 0,
        "other_outputs": 0,
    }

    def _display(*objects: Any) -> None:
        for obj in objects:
            kind, rendered = _serialize_display_object(obj, html_dir)
            display_items.append((kind, rendered))
            if kind == "table":
                render_stats["table_outputs"] += 1
            elif kind == "image":
                render_stats["visual_outputs"] += 1
            else:
                render_stats["other_outputs"] += 1

    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "scope": scope,
        "display": _display,
    }
    previous_display_hook = getattr(builtins, "_cu28_notebook_display", None)
    builtins._cu28_notebook_display = _display
    html_parts = [
        "<html><head><meta charset='utf-8'><title>{}</title></head><body>".format(html.escape(notebook_path.name))
    ]
    result_payload: dict[str, Any] | None = None
    try:
        for cell_index, cell in enumerate(payload.get("cells", []), start=1):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "markdown":
                html_parts.append(_render_markdown(source))
                continue
            if cell.get("cell_type") != "code":
                continue
            buffer = io.StringIO()
            display_items.clear()
            try:
                with redirect_stdout(buffer):
                    exec(source, namespace, namespace)
            except Exception as exc:
                rendered_outputs: list[str] = []
                if buffer.getvalue():
                    rendered_outputs.append(_render_stream_output(buffer.getvalue()))
                rendered_outputs.extend(item[1] for item in display_items)
                html_parts.append(_render_code(source, rendered_outputs))
                raise RuntimeError(f"{notebook_path.name} failed at code cell {cell_index}: {exc}") from exc
            rendered_outputs = []
            if buffer.getvalue():
                render_stats["stream_outputs"] += 1
                rendered_outputs.append(_render_stream_output(buffer.getvalue()))
            rendered_outputs.extend(item[1] for item in display_items)
            html_parts.append(_render_code(source, rendered_outputs))
            if "RESULT" in namespace:
                result_payload = namespace["RESULT"]
    finally:
        if previous_display_hook is None:
            try:
                del builtins._cu28_notebook_display
            except AttributeError:
                pass
        else:
            builtins._cu28_notebook_display = previous_display_hook

    html_parts.append("</body></html>")
    html_path = html_dir / notebook_path.with_suffix(".html").name
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    return {
        "notebook": notebook_path.name,
        "html_path": html_path.resolve().relative_to(repo_root().resolve()).as_posix(),
        "result": result_payload or {},
        "render_stats": render_stats,
    }


def run_notebooks(
    *,
    scope: str = "mixed_context",
    notebook_dir: str | Path = "notebooks",
    output_dir: str | Path = "reports/notebooks",
    continue_on_error: bool = False,
    skip_missing_inputs: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    notebook_root = repo_root() / Path(notebook_dir)
    html_dir = repo_root() / Path(output_dir)
    ensure_directory(html_dir)
    reports_eda_dir = repo_root() / "reports" / "eda"
    ensure_directory(reports_eda_dir)
    figures_dir = repo_root() / "reports" / "figures" / "eda"
    tables_dir = repo_root() / "reports" / "tables" / "eda"
    ensure_directory(figures_dir)
    ensure_directory(tables_dir)

    for directory, pattern in [(html_dir, "*.html"), (figures_dir, "*.png"), (tables_dir, "*.csv")]:
        for path in directory.glob(pattern):
            path.unlink()

    selected_order = NOTEBOOK_ORDER if not smoke else NOTEBOOK_ORDER
    executed = []
    failures = []
    result_payloads = []
    for notebook_name in selected_order:
        notebook_path = notebook_root / notebook_name
        if not notebook_path.exists():
            error = f"Notebook not found: {notebook_name}"
            failures.append({"notebook": notebook_name, "error": error})
            if continue_on_error:
                continue
            raise FileNotFoundError(error)
        try:
            executed_payload = _execute_notebook(notebook_path, scope=scope, html_dir=html_dir)
            executed.append(executed_payload["notebook"])
            notebook_result = dict(executed_payload["result"])
            notebook_result["render_stats"] = executed_payload["render_stats"]
            result_payloads.append(notebook_result)
        except FileNotFoundError as exc:
            failures.append({"notebook": notebook_name, "error": str(exc)})
            if not skip_missing_inputs and not continue_on_error:
                raise
        except Exception as exc:  # pragma: no cover - smoke path covers happy path
            failures.append({"notebook": notebook_name, "error": str(exc)})
            if not continue_on_error:
                raise

    eda_summary = build_eda_summary(result_payloads)
    summary = {
        "scope": scope,
        "executed_notebooks": executed,
        "failed_notebooks": failures,
        "notebook_results": result_payloads,
        "continue_on_error": continue_on_error,
        "skip_missing_inputs": skip_missing_inputs,
        "eda_summary_path": "reports/eda/eda_summary__mixed_context.json",
        "generated_at_utc": eda_summary["executed_at_utc"],
    }
    write_json(reports_eda_dir / "notebook_execution_summary__mixed_context.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute CU28 audit notebooks headlessly and export HTML.")
    parser.add_argument("--scope", default="mixed_context", help="Notebook scope. Only mixed_context is supported.")
    parser.add_argument("--notebook-dir", default="notebooks", help="Directory containing the notebooks.")
    parser.add_argument("--output-dir", default="reports/notebooks", help="Directory where HTML exports will be written.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue executing the remaining notebooks after a failure.")
    parser.add_argument("--skip-missing-inputs", action="store_true", help="Skip notebooks whose required inputs are missing.")
    parser.add_argument("--smoke", action="store_true", help="Reserved reduced mode for CI smoke execution.")
    args = parser.parse_args(argv)

    result = run_notebooks(
        scope=args.scope,
        notebook_dir=args.notebook_dir,
        output_dir=args.output_dir,
        continue_on_error=args.continue_on_error,
        skip_missing_inputs=args.skip_missing_inputs,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["failed_notebooks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
