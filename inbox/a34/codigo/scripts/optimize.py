"""
Script: Optimize

Runs single-objective GA optimization per scenario using the trained model artifacts.

Usage:
    python scripts/optimize.py
    python scripts/optimize.py --input data/splits/test.csv --output data/predictions/evaluation_rt_hist_vs_ia.csv
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import optimize


def main():
    parser = argparse.ArgumentParser(description="DATAGIA — Run GA optimization")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--pop-size", type=int, default=None, help="GA population size")
    parser.add_argument("--n-gen", type=int, default=None, help="GA generations")
    parser.add_argument("--cxpb", type=float, default=None, help="GA crossover probability")
    parser.add_argument("--mutpb", type=float, default=None, help="GA mutation probability")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed")
    args = parser.parse_args()

    optimize(
        input_path=args.input,
        output_path=args.output,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()