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

# Training (fine-tuning) hyperparameters — must match configs/baseline_efficientnet.yaml and
# src/training/{trainer,losses}.py from the delivered training code exactly (manifest.training).
TRAIN_LEARNING_RATE = 0.0001
TRAIN_WEIGHT_DECAY = 0.0001
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_EPOCHS = 100
TRAIN_EARLY_STOPPING_PATIENCE = 15
TRAIN_SEED = 42
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 0.000001
TRAIN_VAL_SPLIT = 0.2
MIN_TRAIN_SAMPLES = 10
