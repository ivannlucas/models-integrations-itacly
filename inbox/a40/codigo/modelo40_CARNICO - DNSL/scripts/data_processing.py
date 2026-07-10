# data_processing.py
"""
Launcher para ejecutar el preprocesamiento completo del dataset.

Ejecuta la función `data_processing()` definida en src/main.py
"""
import os
import sys

# Añadir carpeta raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import data_processing

if __name__ == "__main__":
    data_processing()