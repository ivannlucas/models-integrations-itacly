"""Static configuration for the thermal-udder subclinical-mastitis CNN plugin."""

MODEL_ID = "ml4-lactic-cnn-thermal-early-disease-detection"
ARTIFACT_FOLDER_NAME = "ml4_lactic_cnn_thermal_early_disease_detection"
MODEL_FILENAME = "baseline_efficientnet_final_model.pth"

FRAMEWORK = "pytorch + timm"
VERSION = "1.0.0"

# Architecture — must match the training configuration exactly.
BACKBONE = "efficientnet_b0"
NUM_CLASSES = 2
DROPOUT = 0.3
CLASS_NAMES = ["Healthy", "SCM"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
