import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import fine_tune


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibración/fine-tuning del modelo con datos etiquetados del cliente"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="CSV con variables de entrada y columna consumo_agua_l",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default=None,
        help="Ruta donde guardar el modelo calibrado",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=None,
        help="Ruta donde guardar las métricas de calibración",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de épocas de fine-tuning",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate del fine-tuning",
    )
    args = parser.parse_args()

    fine_tune(
        input_path=args.input_path,
        output_model_path=args.output_model_path,
        metrics_path=args.metrics_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
