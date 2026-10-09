"""Static configuration for the fungal leaf-disease CNN plugin."""

MODEL_ID = "ml2-fungal-cnn-disease-detection"
ARTIFACT_FOLDER_NAME = "ml2_fungal_cnn_disease_detection"
MODEL_FILENAME = "leafcnn_best.pth"

IMAGE_SIZE = 224

# Clases en el mismo orden que el repositorio de entrenamiento (LeafCNN).
CLASS_NAMES = ["black_rot", "downy_mildew", "healthy", "powdery_mildew", "trunk_disease"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# -----------------------------------------------------------------------------
# Retraining hyperparameters — copied verbatim from the delivered training
# repository (config/config.py + src/training/model.py:get_optimizer_scheduler),
# never chosen ad hoc by this plugin. See inbox/a02/manifest.yaml#training.
# -----------------------------------------------------------------------------
TRAIN_LR = 0.001065
TRAIN_WEIGHT_DECAY = 0.000001
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_EPOCHS = 100
TRAIN_PATIENCE = 10
TRAIN_SCHEDULER_FACTOR = 0.3
TRAIN_SCHEDULER_PATIENCE = 3
# The delivered pipeline stratifies a dedicated train/val/test split upstream
# (data/splits/, 74/16/10 %, seed 42) before scripts/train.py ever runs; this
# endpoint instead receives a single labelled ZIP, so it reproduces the same
# fixed-seed stratified-split *idea* with an 80/20 train/val ratio (no held-out
# test slice here — same auto-split pattern as ml10_dairy_disease_vector_detection/ml8).
TRAIN_VAL_SPLIT = 0.2
TRAIN_SPLIT_SEED = 42
