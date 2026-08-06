from __future__ import annotations

import builtins
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from src.utils import find_repo_root


def project_root() -> Path:
    cwd_candidate = find_repo_root(Path.cwd())
    if (cwd_candidate / "src").exists() and (cwd_candidate / "notebooks").exists():
        return cwd_candidate

    module_candidate = find_repo_root(Path(__file__).resolve().parent)
    if (module_candidate / "src").exists() and (module_candidate / "notebooks").exists():
        return module_candidate

    return module_candidate


def execution_metadata(scope: str) -> dict[str, str]:
    return {
        "scope": scope,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
    }


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def ensure_eda_dirs() -> dict[str, Path]:
    root = project_root()
    directories = {
        "figures": root / "reports" / "figures" / "eda",
        "tables": root / "reports" / "tables" / "eda",
        "notebooks": root / "reports" / "notebooks",
        "eda": root / "reports" / "eda",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def figure_path(filename: str) -> Path:
    return ensure_eda_dirs()["figures"] / filename


def table_path(filename: str) -> Path:
    return ensure_eda_dirs()["tables"] / filename


def notebook_html_path(filename: str) -> Path:
    return ensure_eda_dirs()["notebooks"] / filename


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(project_root().resolve()).as_posix()


def _display_object(value: Any) -> None:
    custom_display = getattr(builtins, "_cu28_notebook_display", None)
    if callable(custom_display):
        custom_display(value)
        return

    from IPython.display import display as ipy_display

    ipy_display(value)


def display_and_save_table(df: pd.DataFrame, output_path: str | Path, max_rows: int = 20) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    if len(df) > max_rows:
        _display_object(df.head(max_rows))
        print(f"Showing first {max_rows} rows of {len(df)}. Full table saved to: {output_path}")
    else:
        _display_object(df)
        print(f"Table saved to: {output_path}")
    return relative_to_root(output_path)


def display_and_save_figure(fig, output_path: str | Path, dpi: int = 150) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    try:
        from IPython.display import Image

        _display_object(Image(filename=str(output_path)))
    except Exception:
        print(f"Figure rendered to file only: {output_path}")
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass

    print(f"Figure saved to: {output_path}")
    return relative_to_root(output_path)


def save_table(df: pd.DataFrame, filename: str) -> str:
    return display_and_save_table(df, table_path(filename))


def save_figure(fig, filename: str) -> str:
    return display_and_save_figure(fig, figure_path(filename))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_frame(title: str, frame: pd.DataFrame, rows: int = 10) -> None:
    print(title)
    if frame.empty:
        print("[empty dataframe]")
        return
    print(frame.head(rows).to_string(index=False))


def print_series(title: str, series: pd.Series, rows: int = 20) -> None:
    print(title)
    if series.empty:
        print("[empty series]")
        return
    print(series.head(rows).to_string())


def parse_markdown_table(path: str | Path) -> pd.DataFrame:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    table_lines = []
    capture = False
    for line in lines:
        if line.startswith("| source_id |"):
            capture = True
        if capture and line.startswith("|"):
            table_lines.append(line)
        elif capture and not line.startswith("|"):
            break
    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found in {path}")
    rows = []
    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    for raw_line in table_lines[2:]:
        values = [cell.strip() for cell in raw_line.strip("|").split("|")]
        rows.append(values)
    return pd.DataFrame(rows, columns=header)


def load_source_manifests(raw_root: str | Path) -> list[dict[str, Any]]:
    manifests = []
    for manifest_path in sorted(Path(raw_root).glob("*/source_manifest.json")):
        manifests.append(read_json(manifest_path))
    return manifests


def load_tabular_file(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.read_csv(file_path, sep=None, engine="python")
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=0)
    raise ValueError(f"Unsupported tabular file format: {file_path.suffix}")


def detect_temporal_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if any(token in str(column).lower() for token in ["date", "week", "month", "year", "period"])
    ]


def first_valid_temporal_range(frame: pd.DataFrame) -> dict[str, str | None]:
    for column in detect_temporal_columns(frame):
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().any():
            return {
                "column": str(column),
                "date_min": str(parsed.min().date()),
                "date_max": str(parsed.max().date()),
            }
    return {"column": None, "date_min": None, "date_max": None}
