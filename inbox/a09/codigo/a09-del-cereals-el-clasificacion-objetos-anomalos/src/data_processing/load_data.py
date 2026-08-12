from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe dataset: {p}")
    return pd.read_csv(p)
