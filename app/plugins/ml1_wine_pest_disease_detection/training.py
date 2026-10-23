"""Retraining logic for ml1 — faithful port of src/training/train_tabular.py::run_tabular_training
restricted to the RandomForest tabular experts (see manifest.training.scope_note for why the CNN
Deep Ensemble is out of scope for an on-demand retraining endpoint).

The original always persists a master 80/20 split to data/splits/train_rf.csv|test_rf.csv and
reuses it across retraining runs; this plugin's train() always starts from the client's full
CSV (it has no server-side master split to reuse — a fresh model per request, not an
incremental one), but keeps the exact same split call (train_test_split(..., test_size=0.2,
random_state=42), no stratify — the original has none either) and the exact same RF
hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from app.plugins.ml1_wine_pest_disease_detection.constants import (
    CLASSES,
    SENSOR_COLS,
    TRAIN_RANDOM_STATE,
    TRAIN_RF_CLASE_PARAMS,
    TRAIN_RF_RIESGO_PARAMS,
    TRAIN_TEST_SIZE,
)

REQUIRED_COLUMNS = (*SENSOR_COLS, "Etiqueta", "Indice_Riesgo_Cientifico")


@dataclass
class TabularTrainResult:
    """Everything train() needs to build TrainResponse and upload to MLflow."""

    rf_clase: RandomForestClassifier
    rf_riesgo: RandomForestRegressor
    accuracy: float
    f1_macro: float
    f1_weighted: float
    mae_riesgo: float
    n_train: int
    n_test: int


def train_tabular_experts(df: pd.DataFrame) -> TabularTrainResult:
    """Validate the input CSV and fit fresh RandomForest classifier/regressor instances.

    Never mutates any already-loaded model — always builds brand-new sklearn estimators, so a
    concurrent /predict against the base artifact is unaffected while this runs.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV de entrenamiento: {missing}")

    unknown_labels = sorted(set(df["Etiqueta"].unique()) - set(CLASSES))
    if unknown_labels:
        raise ValueError(
            f"El CSV contiene etiquetas no reconocidas (no están en las 15 clases del modelo): "
            f"{unknown_labels}"
        )
    if len(df) < 10:
        raise ValueError(
            f"Se necesitan al menos 10 filas para separar un holdout de test del "
            f"{TRAIN_TEST_SIZE:.0%} (recibidas: {len(df)})."
        )

    name_to_id = {name: i for i, name in enumerate(CLASSES)}
    x_full = df[list(SENSOR_COLS)].copy()
    y_clase = df["Etiqueta"].map(name_to_id).values.astype(int)
    y_riesgo = df["Indice_Riesgo_Cientifico"].values.astype(float)

    x_train, x_test, y_clase_train, y_clase_test, y_riesgo_train, y_riesgo_test = train_test_split(
        x_full, y_clase, y_riesgo, test_size=TRAIN_TEST_SIZE, random_state=TRAIN_RANDOM_STATE,
    )

    rf_clase = RandomForestClassifier(
        **TRAIN_RF_CLASE_PARAMS, n_jobs=-1, random_state=TRAIN_RANDOM_STATE,
    )
    rf_clase.fit(x_train, y_clase_train)

    rf_riesgo = RandomForestRegressor(
        **TRAIN_RF_RIESGO_PARAMS, n_jobs=-1, random_state=TRAIN_RANDOM_STATE,
    )
    rf_riesgo.fit(x_train, y_riesgo_train)

    pred_clase_test = rf_clase.predict(x_test)
    pred_riesgo_test = rf_riesgo.predict(x_test)

    return TabularTrainResult(
        rf_clase=rf_clase,
        rf_riesgo=rf_riesgo,
        accuracy=float(accuracy_score(y_clase_test, pred_clase_test)),
        f1_macro=float(f1_score(y_clase_test, pred_clase_test, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_clase_test, pred_clase_test, average="weighted", zero_division=0)),
        mae_riesgo=float(mean_absolute_error(y_riesgo_test, pred_riesgo_test)),
        n_train=len(x_train),
        n_test=len(x_test),
    )
