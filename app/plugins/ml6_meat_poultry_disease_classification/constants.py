"""Static configuration for the poultry disease deep-ensemble classifier (use case a06).

Two independent majority-voting ensembles, as delivered by the AI team
(inbox/a06/codigo/config/config.yaml, src/main.py::predict):

- ``thermal``: DenseNet-121 + EfficientNetV2-S + ResNet-50 on thermal images (healthy / sick).
- ``rgb``: VGG19-BN + EfficientNet-B7 + MobileNetV3-L on RGB fecal images
  (healthy / cocci / ncd / salmo).

The request selects the ensemble with ``model_key`` (predict) or ``task`` (train).
"""

MODEL_ID = "ml6-meat-poultry-disease-classification"
ARTIFACT_FOLDER_NAME = "ml6_meat_poultry_disease_classification"
VERSION = "1.0.0"

FRAMEWORK = "pytorch + torchvision"

TASK_THERMAL = "thermal"
TASK_RGB = "rgb"
TASKS = (TASK_THERMAL, TASK_RGB)

# Base models of each ensemble, in config.yaml order (tasks.<task>.models.names). The order
# matters: src/main.py breaks a three-way tie with Counter.most_common, i.e. the vote of the
# first model in this list wins.
TASK_MODELS = {
    TASK_THERMAL: ("densenet121", "efficientnet_v2s", "resnet50"),
    TASK_RGB: ("vgg19_bn", "efficientnet_b7", "mobilenet_v3l"),
}
MODEL_FILENAMES = {
    name: f"{name}.pt" for names in TASK_MODELS.values() for name in names
}

# Class index -> logical class name (config.yaml tasks.<task>.data.class_names).
CLASS_NAMES = {
    TASK_THERMAL: ("healthy", "sick"),
    TASK_RGB: ("healthy", "cocci", "ncd", "salmo"),
}
# On-disk folder of each logical class in the raw datasets (config.yaml data.folder_map).
FOLDER_MAP = {
    TASK_THERMAL: {"healthy": "healthy", "sick": "sick"},
    TASK_RGB: {"healthy": "pcrhealthy", "cocci": "pcrcocci", "ncd": "pcrncd", "salmo": "pcrsalmo"},
}

# src/utils/io.py::SUPPORTED_IMAGE_EXTENSIONS
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Preprocessing — src/data_processing/preprocess.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)
RGB_RESIZE = (256, 256)
# Thermal images are NOT resized: the models were trained on the camera's native frame
# (PIL size = width x height = 240 x 320; memoria §3.2).
THERMAL_NATIVE_SIZE = (240, 320)

# Training — config/config.yaml (seed, split, tasks.<task>.training) and
# src/training/trainer.py::train_model (Adam + CrossEntropy + StepLR(7, 0.1) + early stopping).
TRAIN_SEED = 42
TRAIN_SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
TRAIN_EPOCHS = 100
TRAIN_PATIENCE = 10
TRAIN_MIN_DELTA = 0.0001
TRAIN_STEP_SIZE = 7
TRAIN_GAMMA = 0.1
TRAIN_HYPERPARAMS = {
    TASK_THERMAL: {
        "learning_rate": 0.0002139312453740547,
        "weight_decay": 0.000689762040038286,
        "batch_size": 4,
    },
    TASK_RGB: {
        "learning_rate": 0.005372828091952831,
        "weight_decay": 0.0006682955795832878,
        "batch_size": 64,
    },
}

# MLflow layout of a retrained ensemble: model/<name>.pt (+ model/metadata.json).
MLFLOW_ARTIFACT_PATH = "model"
MLFLOW_METADATA_FILENAME = "metadata.json"

# Test-set metrics declared by the AI team (memoria Tablas 4, 5 y 6 / data/*/predictions/).
REPORTED_TEST_METRICS = {
    TASK_THERMAL: {
        "n_test": 40,
        "ensemble": {"accuracy": 0.8000, "precision_weighted": 0.7964,
                     "recall_weighted": 0.8000, "f1_weighted": 0.7960},
        "densenet121": {"accuracy": 0.6750, "f1_weighted": 0.6774},
        "efficientnet_v2s": {"accuracy": 0.6750, "f1_weighted": 0.6808},
        "resnet50": {"accuracy": 0.7750, "f1_weighted": 0.7604},
    },
    TASK_RGB: {
        "n_test": 192,
        "ensemble": {"accuracy": 0.8490, "precision_weighted": 0.8505,
                     "recall_weighted": 0.8490, "f1_weighted": 0.8484},
        "vgg19_bn": {"accuracy": 0.8490, "f1_weighted": 0.8493},
        "efficientnet_b7": {"accuracy": 0.8125, "f1_weighted": 0.8152},
        "mobilenet_v3l": {"accuracy": 0.8490, "f1_weighted": 0.8476},
    },
}
