import sys
import os
import yaml
import argparse
# Añadimos el path para poder importar desde src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import data_generating

if __name__ == "__main__":
    data_generating()
    print("Datos generados y splits guardados con éxito.")
    print("Siguiente paso: ejecuta 'python scripts/data_processing.py'")

