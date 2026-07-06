import sys
import os
import argparse
import yaml

# Añadir el directorio raíz al path para poder importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import optimize

def run_optimization():
    """
    Punto de entrada para la optimización de setpoints (GA) - Datagia.
    Permite ejecución individual (vía config) o por lotes (vía CSV).
    """
    # 1. Configurar Argumentos de Línea de Comandos
    parser = argparse.ArgumentParser(description="Motor de Optimización Genético - Proyecto Datagia")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="single_mode",
        choices=["single_mode", "massive_mode", "csv_mode"],
        help="Modo de ejecución"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="CSV de escenarios cuando mode=csv_mode"
    )

    args = parser.parse_args()
    print("ARGUMENTOS:", args)

    print("====================================================")
    print("INICIANDO MOTOR DE OPTIMIZACIÓN (GA) - DATAGIA")
    print("====================================================")

    try:
        optimize(mode=args.mode, batch_input_path=args.input_path)
        print("\nProceso de optimización completado con éxito.")

    except Exception as e:
        print(f"\nError crítico durante la optimización: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_optimization()
