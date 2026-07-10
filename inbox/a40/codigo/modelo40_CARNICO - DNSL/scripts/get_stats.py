# scripts/get_stats.py
"""
Launcher para obtener estadísticas y descriptores del dataset procesado.

Ejecuta la función `get_stats()` definida en src/main.py para generar
un informe de las features, tipos de datos y distribución del target.
"""
import os
import sys

# Añadir carpeta raíz al path para poder importar desde 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.main import get_stats
except ImportError as e:
    print(f"Error: No se pudo importar la función get_stats desde src.main. {e}")
    sys.exit(1)

if __name__ == "__main__":
    # Ejecuta la obtención de estadísticas del sistema configurado en config.yaml
    get_stats()