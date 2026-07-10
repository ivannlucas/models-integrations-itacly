"""
Script: Predict

Runs inference on a dataset using the trained model and saves predictions.

Usage:
    python scripts/predict.py
    python scripts/predict.py --input data/splits/test.csv --output data/predictions/my_preds.csv
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import predict


def main():
    parser = argparse.ArgumentParser(description="DATAGIA — Run inference")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()
    predict(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
