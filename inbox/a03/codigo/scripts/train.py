import sys
import os
from pathlib import Path

# Añadimos la raíz del proyecto al sys.path para importaciones de "src"
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.main import main

if __name__ == "__main__":
    # Trucamos los argumentos para que main parseé 'train' por defecto
    if len(sys.argv) == 1:
        sys.argv.append("train")
    
    main()
