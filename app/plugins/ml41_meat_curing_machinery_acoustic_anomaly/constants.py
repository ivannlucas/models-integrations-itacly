MODEL_ID = "ml41-meat-curing-machinery-acoustic-anomaly"
VERSION = "1.0.0"
ARTIFACT_FOLDER_NAME = "ml41_meat_curing_machinery_acoustic_anomaly"

FRAMEWORK = ["torch==2.4.1", "torchaudio==2.4.1", "librosa==0.11.0", "soundfile==0.13.1",
             "numpy==1.24.4", "scikit-learn==1.3.2"]

MACHINES = ("fan", "pump", "slider", "valve")
MACHINE_IDS = ("id_00", "id_02", "id_04", "id_06")
SNRS = ("-6_dB", "0_dB", "6_dB")

# Checkpoint layout: models/artifacts/vit_tiny_{machine}/{machine}/{machine_id}/{snr}/{best.pth,maha_stats.npz}
# NOTE: config_vit_tiny_{machine}.yaml (delivered by the AI team) declares
# paths.artifacts = models/artifacts/vit_tiny_hpo_{machine}, which does NOT match the
# folder actually delivered (models/artifacts/vit_tiny_{machine}, no "_hpo_"). We use the
# real on-disk path, not the one in those YAML files (see manifest.yaml known_issues).
CHECKPOINT_FILENAME = "best.pth"
MAHA_STATS_FILENAME = "maha_stats.npz"

# Audio-MAE preprocessing parameters (inbox/a41/codigo/src/data_processing/preprocess.py)
AUDIO_SR = 16000
N_MELS = 128
WIN_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 1024
TARGET_FRAMES = 1024

# Audio-MAE inference parameters (inbox/a41/codigo/config/config.yaml: predict:)
MASK_RATIO = 0.75
NUM_MASKS = 1

# Architecture hyperparameters — embed_dim is common to all machines, but
# depth/num_heads/pca_components come from a per-machine HPO study (Optuna) and DIFFER.
# The checkpoint itself does NOT self-describe these (only stores machine/machine_id/snr/
# norm_mean/norm_std) — this table MUST be used to instantiate AudioMAE before
# load_state_dict(), per machine, or the load fails on a shape mismatch.
# Source: inbox/a41/codigo/config/config_vit_tiny_{fan,pump,slider,valve}.yaml
_COMMON_ARCH = {
    "img_size": (128, 1024),
    "patch_size": (16, 16),
    "in_chans": 1,
    "embed_dim": 192,
    "decoder_embed_dim": 128,
    "decoder_depth": 4,
    "decoder_num_heads": 4,
    "norm_pix_loss": True,
}

ARCHITECTURE_BY_MACHINE: dict[str, dict] = {
    "fan":    {**_COMMON_ARCH, "depth": 9, "num_heads": 6, "pca_components": 70},
    "pump":   {**_COMMON_ARCH, "depth": 7, "num_heads": 4, "pca_components": 90},
    "slider": {**_COMMON_ARCH, "depth": 4, "num_heads": 4, "pca_components": 90},
    "valve":  {**_COMMON_ARCH, "depth": 5, "num_heads": 8, "pca_components": 30},
}

# Fine-tuning hyperparameters per machine (best Optuna HPO trial) — never invented,
# sourced from inbox/a41/codigo/config/config_vit_tiny_{machine}.yaml.
TRAINING_COMMON = {
    "epochs": 100, "patience": 20, "seed": 42, "weight_decay_default": 0.05,
    "betas": (0.9, 0.95),
}
TRAINING_HYPERPARAMS_BY_MACHINE: dict[str, dict] = {
    "fan":    {"warmup_epochs": 7,  "weight_decay": 0.0004723501245203774, "mask_ratio": 0.5574479341391712, "lr": 0.0004502789161109957, "batch_size": 32},
    "pump":   {"warmup_epochs": 9,  "weight_decay": 0.02356540010076827,   "mask_ratio": 0.7277871776757026, "lr": 0.0003962419971237532, "batch_size": 32},
    "slider": {"warmup_epochs": 11, "weight_decay": 0.000904102556502169,  "mask_ratio": 0.5934777228914295, "lr": 7.567742603471887e-05, "batch_size": 64},
    "valve":  {"warmup_epochs": 4,  "weight_decay": 0.0013619127359149519, "mask_ratio": 0.5544394608434632, "lr": 0.0002560271293015735, "batch_size": 32},
}

# How many (machine, machine_id, snr) checkpoints to keep resident in memory at once.
# Loading all 48 eagerly would need ~2GB RAM; lazy load + small LRU cache is enough
# since a deployment typically monitors a handful of physical machines.
CHECKPOINT_CACHE_SIZE = 8

# metrics_reported from manifest.yaml (source: metrics.json packaged with each
# checkpoint + memoria sección 5.4/6) — used verbatim in stats().
METRICS_REPORTED = {
    "auc_maha_mean": 0.7720,
    "fnr_mean": 0.0909,
    "fpr_mean": 0.4684,
    "pct_cumple_fnr_objetivo": 1.0,
    "fnr_objetivo_negocio": 0.10,
    "auc_maha_range_por_maquina": {
        "fan": [0.68, 0.97], "pump": [0.58, 0.99],
        "slider": [0.58, 0.99], "valve": [0.45, 0.61],
    },
}
