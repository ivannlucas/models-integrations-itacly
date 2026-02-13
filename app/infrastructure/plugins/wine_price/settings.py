from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class WinePaths:
    base: Path
    data_raw: Path
    models_prod: Path
    output: Path

def default_wine_paths() -> WinePaths:
    base = Path(__file__).resolve().parents[2] / "vendor" / "wine_model"
    return WinePaths(
        base=base,
        data_raw=base / "data" / "raw",
        models_prod=base / "models" / "prod",
        output=base / "output",
    )