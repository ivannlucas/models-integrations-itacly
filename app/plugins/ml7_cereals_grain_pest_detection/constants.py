"""Static configuration for the grain pest-detection YOLO plugin."""

MODEL_ID = "ml7-cereals-grain-pest-detection"
ARTIFACT_FOLDER_NAME = "ml7_cereals_grain_pest_detection"
MODEL_FILENAME = "best.pt"

FRAMEWORK = "ultralytics"
VERSION = "1.0.0"

CONF_THRESHOLD = 0.28
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --- Class id -> species code decoding (BUGFIX, see postprocessing.yolo_results_to_dict) ---
#
# The delivered preprocessing pipeline has TWO different, inconsistent class-order sources:
#   - config/config.yaml::valid_classes = [cf, sz, rd, tc, os] — this is the order
#     src/data_processing/preprocess.py::convert_xmls_to_yolo_labels() actually uses to
#     assign the integer class id written into every YOLO label .txt (class2id =
#     {name: i for i, name in enumerate(classes)}), i.e. the ids the detector is really
#     trained against: cf=0, sz=1, rd=2, tc=3, os=4.
#   - src/data_processing/preprocess.py::create_dataset_yaml() instead computes
#     `class_names = sorted(valid_classes_found_in_xmls)` — alphabetical order — and writes
#     THAT into dataset.yaml's `names:`, which YOLO then embeds into the checkpoint as
#     `model.names` at training time: cf=0, os=1, rd=2, sz=3, tc=4.
#
# The shipped best.pt therefore has the detector head correctly trained on the first
# (config.yaml) id order, but its own embedded `model.names` reports the second
# (alphabetical) order — wrong for ids 1, 3 and 4 (ids 0 and 2 happen to coincide: cf and rd
# are in the same position in both orderings). Using `results.names` to decode predictions
# (as the delivered src/predict/predictor.py and the previous version of this plugin both
# did) silently mislabels roughly 3 of every 5 detections.
#
# Confirmed by running the real delivered checkpoint on 500 real test-split images
# (data/splits/images/test/, 100 stratified per class, inbox/a07/codigo/):
#   - decoding with the checkpoint's embedded `model.names` (alphabetical order):
#     species accuracy = 195/500 = 39.0%
#   - decoding with this dict (config.yaml order, i.e. what the detector actually learned):
#     species accuracy = 472/500 = 94.4%
# (model.val() on the full 2.700-image real test split gives mAP50=0.871/precision=0.876/
# recall=0.856, consistent with the memoria's 0.87/0.84/0.81 — confirms the detector/box
# regression itself is correct; the bug is purely in the id->name decoding layer, which
# IoU-based detection metrics never exercise.)
#
# This is the correct mapping to decode with. Do NOT decode via `results.names` /
# `model.names` (only safe to use as a shape/sanity probe), and do NOT regenerate this from
# `sorted()` over any class list.
TRAINING_CLASS_ID_TO_CODE: dict[int, str] = {
    0: "cf",
    1: "sz",
    2: "rd",
    3: "tc",
    4: "os",
}

# Inference image size — the delivered predictor.py (predict_folder) always passes
# imgsz=img_size (512, config/config.yaml::model.img_size) explicitly to model.predict().
# Omitting it makes ultralytics fall back to its own default (640, letterbox-resized),
# which does not match what the model was trained/evaluated on. Must be passed on every
# model.predict() call (see inbox/a07 manifest known_issues).
IMG_SIZE = 512

# ── Training hyperparameters (fine-tuning) ──────────────────────────────────────
# Mirrors config/config.yaml::model + model.hyperparameters from the delivered code
# (src/training/train.py::train_model), confirmed against memoria sección 6.2 Tabla 6.
# Never exposed on TrainRequest — same values the AI team used for the original best.pt.
TRAIN_EPOCHS = 300
TRAIN_PATIENCE = 20
TRAIN_BATCH_SIZE = 32
TRAIN_WORKERS = 0
TRAIN_SEED = 42
TRAIN_HYPERPARAMS = {
    "lr0": 0.00868,
    "momentum": 0.97,
    "weight_decay": 0.00027,
    "box": 8.19212,
    "cls": 0.72124,
    "dfl": 1.82105,
    "hsv_h": 0.0017,
    "hsv_s": 0.64489,
    "hsv_v": 0.36131,
    "degrees": 0.00145,
    "translate": 0.16011,
    "scale": 0.3,
    "fliplr": 0.4361,
    "mosaic": 0.72567,
    "mixup": 0.00512,
}
# Evaluation confidence used by src/training/validation.py::validate_model() after
# training — same eval.conf as inference (config/config.yaml).
TRAIN_EVAL_CONF = CONF_THRESHOLD
