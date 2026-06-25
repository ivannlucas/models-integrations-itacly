"""Static configuration for the grain pest-detection YOLO plugin."""

MODEL_ID = "ml7-cereals-grain-pest-detection"
ARTIFACT_FOLDER_NAME = "ml7_cereals_grain_pest_detection"
MODEL_FILENAME = "best.pt"

FRAMEWORK = "ultralytics"
VERSION = "1.0.0"

CONF_THRESHOLD = 0.28
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
