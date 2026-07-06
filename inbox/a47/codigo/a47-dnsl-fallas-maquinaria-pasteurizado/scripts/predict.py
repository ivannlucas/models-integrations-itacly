import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import run_predict

if __name__ == "__main__":
    # Puede recibir argumentos, pero por simplicidad de nuestro wrapper usaremos los defautls o pasaremos args.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cycle", type=int, default=None, help="Si se incluye filtrará por ese Cycle_ID específico.")
    parser.add_argument("--apply_digital_twin", action="store_true", help="Si está activo, aplica el desplazamiento térmico del Gemelo Digital al baseline de 65ºC. Útil para datos de laboratorio (UCI). Por defecto se usan las temperaturas reales.")
    args = parser.parse_args()
    
    run_predict(args.input, args.output, target_cycle=args.cycle, apply_digital_twin=args.apply_digital_twin)
