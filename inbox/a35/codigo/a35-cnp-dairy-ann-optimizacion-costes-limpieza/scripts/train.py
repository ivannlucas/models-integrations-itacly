import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import train
import yaml


if __name__ == "__main__":
    train()
    print("Proceso de entrenamiento finalizado con éxito.")
    print("Siguiente paso: ejecuta 'python scripts/predict.py'")
