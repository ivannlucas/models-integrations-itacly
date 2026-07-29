# scripts/predict.py
"""
Launcher para ejecutar la inferencia (predicciones) del modelo.

Ejecuta la función `predict()` definida en src/main.py. 
Utiliza el modelo guardado en models/artifacts/ y los datos de entrada
para generar un archivo de predicciones en data/predictions/.
"""
import os
import sys

# Añadir carpeta raíz al path para poder importar desde 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.main import predict
except ImportError as e:
    print(f"Error: No se pudo importar la función predict desde src.main. {e}")
    sys.exit(1)

if __name__ == "__main__":
    # Ejecuta el pipeline de inferencia
    # Se puede pasar una ruta de archivo específica si se desea: predict(input_data="data/test_new.csv")
    predict()