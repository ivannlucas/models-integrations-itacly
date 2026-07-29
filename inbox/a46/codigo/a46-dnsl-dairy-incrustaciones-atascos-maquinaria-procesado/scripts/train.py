from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CU07 predictive model.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--telemetry", default=None)
    parser.add_argument("--maintenance", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    parser.add_argument("--seq-len", type=int, dest="seq_len", default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-ablate-clocks", action="store_true")
    args = parser.parse_args()

    result = train(
        config_path=args.config,
        telemetry=args.telemetry,
        maintenance=args.maintenance,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        ablate_clocks=(False if args.no_ablate_clocks else None),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
