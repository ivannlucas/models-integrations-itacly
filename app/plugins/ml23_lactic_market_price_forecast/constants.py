"""Constants for ml23 — lactic market price forecast (GRU, PyTorch)."""

MODEL_ID = "ml23-lactic-market-price-forecast"
ARTIFACT_FOLDER_NAME = "ml23_lactic_market_price_forecast"
VERSION = "1.0.0"
FRAMEWORK = "pytorch"

MODEL_FILENAME = "gru_model.pt"
SCALER_FILENAME = "rnn_scaler.npz"
MANIFEST_FILENAME = "manifest.json"

# ── Training (fine-tuning) ───────────────────────────────────────────────────
# Mirrors inbox/a23/codigo/config/config.yaml::training + src/training/compare_models.py
# defaults — the same hyperparameters the AI team used to produce the shipped gru_model.pt
# (confirmed against artifacts/ml23_lactic_market_price_forecast/manifest.json: selected_model
# ="GRU", horizon=6, seq_len=6, hidden_size=64). train() only refits THIS architecture (the
# one already selected and deployed) — it does not re-run the full model search/comparison
# (Naive/Drift/XGBoost/LSTM/GRU across CV folds) that produced that selection, the same
# simplification already applied to ml30/ml9's neuroevolution search in this repo.
TRAIN_HORIZON = 6
TRAIN_SEQ_LEN = 6
TRAIN_HIDDEN_SIZE = 64
TRAIN_EPOCHS = 150
TRAIN_BATCH_SIZE = 32
TRAIN_N_SEEDS = 3              # compare_models.py::N_SEEDS
TRAIN_SEED_BASE = 42           # compare_models.py::train_rnn_multi_seed range(42, 42+n_seeds)
TRAIN_TEST_RATIO = 0.1
TRAIN_VAL_RATIO = 0.2
TRAIN_CV_FOLDS = 5             # only used to size the refit validation window, see training.py
TRAIN_CV_VAL_SIZE = 6
TRAIN_PATIENCE = 20
TRAIN_LR = 1e-3
