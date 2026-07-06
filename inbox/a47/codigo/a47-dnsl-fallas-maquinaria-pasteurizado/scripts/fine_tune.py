import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import run_fine_tuning

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Calibración del modelo DNSL a datos reales de planta (fine-tuning).\n"
            "Adapta las capas de decisión del modelo preentrenado con aceite a las\n"
            "propiedades físicas del fluido real (leche u otro fluido de producción).\n\n"
            "Ejemplo de uso:\n"
            "  python scripts/fine_tune.py \\\n"
            "      --train_input data/planta/ciclos_train.csv \\\n"
            "      --val_input   data/planta/ciclos_val.csv\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--train_input",
        type=str,
        required=True,
        help=(
            "Ruta al CSV con los ciclos reales de planta para entrenamiento. "
            "Debe incluir las columnas de sensores y las etiquetas (Target_Fouling, "
            "Target_Valvula, Target_Bomba, Target_Acumulador)."
        ),
    )
    parser.add_argument(
        "--val_input",
        type=str,
        required=True,
        help=(
            "Ruta al CSV con los ciclos reales de planta para validación "
            "(Early Stopping). Mismo formato que --train_input."
        ),
    )
    parser.add_argument(
        "--fluid_density",
        type=float,
        default=None,
        help=(
            "Densidad del fluido real en kg/L. "
            "Si no se indica, se usa el valor del config.yaml "
            "(por defecto: 1.03, leche entera aprox.). "
            "Aceite hidráulico de referencia: ~0.87."
        ),
    )
    parser.add_argument(
        "--fluid_cp",
        type=float,
        default=None,
        help=(
            "Calor específico del fluido real en kJ/(kg·K). "
            "Si no se indica, se usa el valor del config.yaml "
            "(por defecto: 3.93, leche entera aprox.). "
            "Aceite hidráulico de referencia: ~1.88."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número máximo de épocas de calibración. Si no se indica, se usa el valor del config.yaml (por defecto: 50).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Paciencia del Early Stopping: número de épocas sin mejora en "
            "validación antes de detener la calibración. Si no se indica, "
            "se usa el valor del config.yaml (por defecto: 7)."
        ),
    )

    args = parser.parse_args()

    run_fine_tuning(
        train_csv=args.train_input,
        val_csv=args.val_input,
        fluid_density_kg_l=args.fluid_density,
        fluid_cp_kj_kgK=args.fluid_cp,
        ft_epochs=args.epochs,
        ft_patience=args.patience,
    )
