"""Static configuration for the fungal leaf-disease CNN plugin."""

MODEL_ID = "ml2-fungal-cnn-disease-detection"
ARTIFACT_FOLDER_NAME = "ml2_fungal_cnn_disease_detection"
MODEL_FILENAME = "leafcnn_best.pth"

IMAGE_SIZE = 224

# Clases en el mismo orden que el repositorio de entrenamiento (LeafCNN).
CLASS_NAMES = ["black_rot", "downy_mildew", "healthy", "powdery_mildew", "trunk_disease"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
