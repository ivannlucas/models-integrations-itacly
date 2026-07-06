import sys
import os
import yaml
import pandas as pd
import argparse

# Configuración del path para importar desde la estructura de carpetas src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import data_processing


if __name__ == "__main__":
    data_processing()
    print("Datos generados y splits guardados con éxito.")
    print("Siguiente paso: ejecuta 'python scripts/train.py'")
