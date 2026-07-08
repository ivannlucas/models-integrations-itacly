from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained CU07 model.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", dest="telemetry_input", default=None, help="Processed telemetry CSV.")
    parser.add_argument("--scenario", default=None, help="Scenario to use: auto, full or no_clock.")
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    result = predict(
        config_path=args.config,
        telemetry_input=args.telemetry_input,
        scenario=args.scenario,
        batch_size=args.batch_size,
        device=args.device,
        output_name=args.output_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
