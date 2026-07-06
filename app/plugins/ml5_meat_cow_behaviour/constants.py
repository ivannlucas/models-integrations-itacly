"""Static configuration for the cow-behaviour recognition plugin."""

MODEL_ID = "ml5-meat-cow-behaviour"
ARTIFACT_FOLDER_NAME = "ml5_meat_cow_behaviour"

DETECTOR_FILENAME = "detector_model.pth"
CLASSIFIER_FILENAME = "classifier_model.pth"

FRAMEWORK = "torch + detectron2 + pytorchvideo"
VERSION = "1.0.0"

# SlowFast clip configuration (must match the training pipeline).
CLIP_LENGTH = 32   # frames per SlowFast clip
ALPHA = 4          # slow/fast pathway ratio → 8 slow + 32 fast frames
CROP_SIZE = 224    # spatial resolution fed to SlowFast

DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_ANOMALY_THRESHOLD = 0.5

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
