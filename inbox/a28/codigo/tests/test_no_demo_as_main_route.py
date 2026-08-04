from __future__ import annotations

from pathlib import Path


def test_readme_promotes_mixed_context_as_main_route() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "## Comando demo" not in readme
    assert "## CU28 reproducibility quick path" in readme
    assert "## Comandos oficiales por fase" in readme
    assert "## Entrenamiento y reentrenamiento" in readme
    assert "## Reconstrucción desde fuentes raw" in readme
