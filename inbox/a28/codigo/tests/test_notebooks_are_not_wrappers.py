from __future__ import annotations

import json
import unicodedata

from tests.conftest import REPO_ROOT
from tests.test_notebooks_exist import EXPECTED_NOTEBOOKS


FORBIDDEN_PHRASES = [
    "validacion industrial final",
    "datos reales de fabrica",
    "tiempo real",
    "cantidad optima real",
    "compra autonoma",
]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character)).lower()


def test_notebooks_are_narrative_and_not_wrappers() -> None:
    notebook_dir = REPO_ROOT / "notebooks"
    required_sections = [
        "objetivo",
        "inputs",
        "carga de datos",
        "interpretacion",
        "limitaciones",
        "outputs esperados",
    ]
    quality_markers = ["calidad de datos", "validaciones", "inspeccion inicial"]
    visual_markers = ["graficas", "visualizacion", "figura"]

    for notebook_name in EXPECTED_NOTEBOOKS:
        payload = json.loads((notebook_dir / notebook_name).read_text(encoding="utf-8"))
        cells = payload["cells"]
        markdown_cells = [cell for cell in cells if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]

        assert len(cells) >= 18, notebook_name
        assert len(markdown_cells) >= 8, notebook_name
        assert len(code_cells) >= 8, notebook_name

        code_sources = ["".join(cell.get("source", [])) for cell in code_cells]
        full_code = "\n".join(code_sources)
        normalized_code = _normalize(full_code)
        assert "from src.reproducibility.eda import run_" not in full_code, notebook_name
        assert "result = run_" not in normalized_code, notebook_name

        total_code_chars = sum(len(source) for source in code_sources)
        largest_code_chars = max(len(source) for source in code_sources)
        assert total_code_chars > 0, notebook_name
        assert largest_code_chars / total_code_chars <= 0.25, notebook_name

        serialized = json.dumps(payload, ensure_ascii=False)
        normalized = _normalize(serialized)
        assert "c:\\" not in normalized, notebook_name
        assert "/users/" not in normalized, notebook_name

        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in normalized, (notebook_name, phrase)

        for section in required_sections:
            assert section in normalized, (notebook_name, section)
        assert any(marker in normalized for marker in quality_markers), notebook_name
        assert any(marker in normalized for marker in visual_markers), notebook_name
