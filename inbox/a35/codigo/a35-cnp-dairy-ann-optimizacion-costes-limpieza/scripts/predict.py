import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from src.main import evaluate_test, predict_external
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación del modelo o inferencia sobre datos externos")
    parser.add_argument("--input_path", type=str, default=None, help="CSV externo para inferencia")
    parser.add_argument("--output_path", type=str, default=None, help="Ruta de salida para predicciones externas")
    args = parser.parse_args()

    if args.input_path:
        predict_external(args.input_path, args.output_path)
    else:
        evaluate_test()
        print("Evaluación finalizada con éxito")
        print("Siguiente paso: ejecuta 'python scripts/optimize.py'")
