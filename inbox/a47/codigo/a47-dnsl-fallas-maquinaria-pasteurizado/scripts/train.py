import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import run_train

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="store_true", help="Si se incluye, se calculan y guardan las métricas en Test.")
    args = parser.parse_args()
    
    run_train(metrics=args.metrics)
