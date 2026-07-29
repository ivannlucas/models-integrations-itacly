from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import get_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA/statistics outputs for CU07.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--telemetry", default=None)
    parser.add_argument("--maintenance", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seq-len", type=int, dest="seq_len", default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--sample-cycles", type=int, dest="sample_cycles", default=None)
    args = parser.parse_args()

    result = get_stats(
        config_path=args.config,
        telemetry=args.telemetry,
        maintenance=args.maintenance,
        outdir=args.outdir,
        seq_len=args.seq_len,
        stride=args.stride,
        sample_cycles=args.sample_cycles,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
