"""Static configuration for the cereal residue-optimizer plugin."""

MODEL_ID = "ml31-cereals-residue-optimizer"
ARTIFACT_FOLDER_NAME = "ml31_cereals_residue_optimizer"
MODEL_FILENAME = "surrogate_residue.pkl"

FRAMEWORK = "scikit-learn"
VERSION = "1.0.0"

REQUIRED_COLS = [
    "Sup_Secano_ha", "Sup_Regadio_ha", "Lluvia_Primavera_mm", "Sequia_Primavera", "Cultivo",
]
NUMERIC_FEATURES = ["Sup_Secano_ha", "Sup_Regadio_ha", "Lluvia_Primavera_mm", "Sequia_Primavera"]
CATEGORICAL_FEATURES = ["Cultivo"]
TRAIN_TARGET_COL = "Residuo_Disponible_Suelo_t"
