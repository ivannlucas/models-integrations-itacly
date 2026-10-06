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

# --- /train (classifier only) ZIP contract ---
#
# Matches the structure scripts/data/format.py:load_all_clips already consumes in the
# delivered training repo (inbox/a05/codigo/) — a per-clip CVAT/COCO export, NOT an
# invented structure. The detector (bbox-annotation ZIP) remains out of scope.
RAW_FRAMES_DIRNAME = "raw_frames"
ANNOTATIONS_DIRNAME = "annotations"
CVAT_ANNOTATIONS_FILENAME = "instances_default.json"

# Real hyperparameters from the delivered train_classifier.py / memoria §6.4
# (see inbox/a05/manifest.yaml training.reason) — not chosen independently.
TRAIN_LR = 0.0005
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 30
TRAIN_WARMUP_EPOCHS = 2
FOCAL_LOSS_ALPHA = 0.5
FOCAL_LOSS_GAMMA = 1.5
TRAIN_SEED = 42
TRAIN_SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

# --- Classifier index -> behaviour decoding (BUGFIX, see model_loader._load_classifier) ---
#
# The delivered training code has TWO different, inconsistent class-order dicts:
#   - configs/slowfast_cow_behavior.py:BEHAVIOR_TO_IDX (12 classes) — this is the dict
#     `scripts/training/train_classifier.py` imports and embeds into the checkpoint
#     (`torch.save(..., 'behavior_to_idx': BEHAVIOR_TO_IDX)`).
#   - utils/data_utils.py:BehaviorClassificationDataset.__init__ — a DIFFERENT hardcoded
#     dict with a different index order, used to compute the integer `label` that is
#     actually fed to the loss/optimizer in `__getitem__` (`label = self.behavior_to_idx[...]`).
#
# The model's output head therefore learned the data_utils.py order, but the checkpoint's
# embedded `behavior_to_idx` (configs order) is what both `main.py` and this plugin used to
# decode predictions — WRONG for every class except index 0 (grazing) and index 6
# (ruminating-standing), the only two indices where the two dicts happen to agree.
#
# Confirmed by running the real delivered checkpoint on 1,500 real test-split clips:
#   - decoding with the checkpoint's embedded dict (configs order): accuracy=0.44, macro F1=0.17
#   - decoding with this dict (data_utils.py order, i.e. what the model actually learned):
#     accuracy=0.91, macro F1=0.87
# (both numbers share the same frame-level train/val/test leakage in format.py — see
# inbox/a05/manifest.yaml known_issues — so they are a fair A/B of the two decodings, not a
# clean generalization estimate.)
#
# This is the correct mapping to decode with. Do NOT replace it with
# configs/slowfast_cow_behavior.py's BEHAVIOR_TO_IDX, and do NOT trust the checkpoint's own
# embedded `behavior_to_idx` for decoding (it is only used as a model_state_dict/shape probe).
TRAINING_BEHAVIOR_TO_IDX: dict[str, int] = {
    "grazing": 0,
    "drinking": 1,
    "walking": 2,
    "resting-lying": 3,
    "resting-standing": 4,
    "ruminating-lying": 5,
    "ruminating-standing": 6,
    "running": 7,
    "grooming": 8,
    "hidden": 9,
    "other": 10,
    "none": 11,
}
