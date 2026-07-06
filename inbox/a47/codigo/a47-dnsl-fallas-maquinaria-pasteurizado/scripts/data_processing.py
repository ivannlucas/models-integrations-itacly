import os
import sys

# Añadir el directorio raíz al path para que Python encuentre 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import run_data_processing

if __name__ == "__main__":
    run_data_processing()
