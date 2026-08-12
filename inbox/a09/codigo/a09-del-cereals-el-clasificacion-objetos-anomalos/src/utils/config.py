from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe config: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config invalida: se esperaba un diccionario YAML.")
    return cfg
