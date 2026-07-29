# scripts/train.py
"""
Launcher para ejecutar el entrenamiento de los modelos (Aireado o Refrigeración).

Ejecuta la función `train()` definida en src/main.py utilizando la 
configuración especificada en config/config.yaml
"""
import os
import sys

# Añadir carpeta raíz al path para poder importar desde 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.main import train
except ImportError as e:
    print(f"Error: No se pudo importar la función train desde src.main. {e}")
    sys.exit(1)

if __name__ == "__main__":
    # Ejecuta el pipeline de entrenamiento configurado
    train()