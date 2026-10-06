"""
Constantes y configuraciones para el plugin Ml10DairyDiseaseVectorDetection.
"""
DETECTOR_FILENAME = "detector_best.pt"
CLASSIFIER_FILENAME = "best_classifier.pth"
CLASS_NAMES_FILENAME = "class_names.json"
ARTIFACT_FOLDER_NAME = "ml10_dairy_disease_vector_detection"

MODEL_ID = "ml10-dairy-disease-vector-detection"
FRAMEWORK = "pytorch+ultralytics"
VERSION = "1.0.0"

# Umbrales por defecto (coinciden con los del modelo fuente)
DEFAULT_DET_CONF = 0.2
DEFAULT_CLS_CONF = 0.5
