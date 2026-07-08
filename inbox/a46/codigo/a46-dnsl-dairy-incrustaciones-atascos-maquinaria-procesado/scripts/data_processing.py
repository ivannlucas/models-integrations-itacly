from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import data_processing


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/process CU07 data into the repository structure.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--assets", type=int, default=None)
    parser.add_argument("--cycles-per-asset", type=int, dest="cycles_per_asset", default=None)
    parser.add_argument("--dt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--emit-idle-phase", action="store_true")
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()

    result = data_processing(
        config_path=args.config,
        assets=args.assets,
        cycles_per_asset=args.cycles_per_asset,
        dt=args.dt,
        seed=args.seed,
        emit_idle_phase=(True if args.emit_idle_phase else None),
        no_noise=(True if args.no_noise else None),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
