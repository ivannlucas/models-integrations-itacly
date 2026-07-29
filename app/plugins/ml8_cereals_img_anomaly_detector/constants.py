MODEL_ID = "ml8-cereals-img-anomaly-detector"
ARTIFACT_FOLDER_NAME = "modelo8_cereales"
MODEL_FILENAME = "mobilenet_v3_large_cereales_multitask.pth"

IMAGE_SIZE = 224

# El orden DEBE coincidir con el índice de clase del checkpoint entrenado
# (idx_to_class / idx_to_cereal), que se generó alfabéticamente vía ImageFolder:
#   categoría -> {0: hongos, 1: insectos, 2: otros, 3: sano}
#   cereal    -> {0: arroz,  1: maiz,     2: sorgo, 3: trigo}
# Estas listas se usan como fallback si el checkpoint no trae el mapeo y para
# construir el mapeo al reentrenar (train); un orden distinto reetiqueta mal.
CATEGORY_NAMES = ["hongos", "insectos", "otros", "sano"]
CEREAL_NAMES = ["arroz", "maiz", "sorgo", "trigo"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
