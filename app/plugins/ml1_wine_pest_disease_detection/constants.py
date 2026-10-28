"""Constants for ml1 — vineyard pest/disease image detection (Deep Ensemble, VID-Deep_Ensemble).

See inbox/a01/manifest.yaml for the full contract, the two-delivery provenance of the artifacts
(original code ZIP + "modelo 1.zip" with the RF experts and the audited metrics/platform
examples) and the known issues — especially the memoria-vs-code discrepancy on the no-image
fallback and the finding that the delivered ejemplo_inferencia.py/resultado_inferencia_batch.csv
do NOT reflect the real Deep Ensemble's accuracy (RF-only, ~45%) vs. the real, audited
94.69% (report_ensemble_final_no_gases_desacoplado.csv).

Production variant served: Desacoplado + Sin Gases (config.yaml defaults: use_fusion_layer=false,
exclude_gases=true) — the one the memoria's own "Conclusión final y selección del modelo de
despliegue" (sección 6.4) designates for field deployment.
"""

MODEL_ID = "ml1-wine-pest-disease-detection"
ARTIFACT_FOLDER_NAME = "ml1_wine_pest_disease_detection"
VERSION = "1.0.0"
FRAMEWORK = "keras3(backend=torch) + ultralytics(YOLOv8) + scikit-learn(RandomForest)"

# --- Artefactos (todos planos en artifacts/ml1_wine_pest_disease_detection/, variante _no_gases_desacoplado) ---
YOLO_WEIGHTS_FILENAME = "best.pt"
CLASS_MAPPING_FILENAME = "class_mapping_no_gases.json"
ENSEMBLE_WEIGHTS_FILENAMES = {
    "mobilenet": "ensemble_member_mobilenet_no_gases_desacoplado.weights.h5",
    "efficientnet": "ensemble_member_efficientnet_no_gases_desacoplado.weights.h5",
    "densenet": "ensemble_member_densenet_no_gases_desacoplado.weights.h5",
}
# Solo el RF de riesgo (fallback cuando no hay imagen). El RF de clase no se carga: en modo
# desacoplado no se usa nunca (ver inference.run_inference_tabular_only).
RF_RIESGO_FILENAME = "rf_riesgo_expert_no_gases.joblib"

# --- Clases (orden vinculante -- config.yaml::model_params.classes, coincide con class_mapping_no_gases.json) ---
CLASSES = (
    "ALTICA", "BLACK_ROT", "BOTRYTIS", "EMPOASCA", "ERINOSIS", "ESCA_HOJA", "ESCA_RACIMO",
    "HEALTHY_HOJA", "HEALTHY_RACIMO", "ISARIOPSIS", "LOBESIA", "MILDIU", "OIDIO",
    "OIDIO_RACIMO", "RED_MITE",
)
HEALTHY_CLASSES = ("HEALTHY_HOJA", "HEALTHY_RACIMO")
NO_IMAGE_LABEL = "IMAGEN_NO_DISPONIBLE"  # predictor.py::run_inference_tabular_only, modo desacoplado

IMAGE_SIZE = (224, 224)
# keras/src/applications/efficientnet.py::IMAGENET_STDDEV_RGB -- copiado aqui (modulo privado de
# Keras) para reconstruir la capa Rescaling que EfficientNetB0 lleva con weights='imagenet'.
IMAGENET_STDDEV_RGB = (0.229, 0.224, 0.225)
YOLO_CONF_THRESHOLD = 0.3

# Sensores de produccion (exclude_gases=true -> 9, sin CO2_ppm/VOC_ppb). Orden vinculante: el
# que usa predictor.py::_align_sensors para construir el vector de entrada de la rama tabular.
SENSOR_COLS = (
    "Mes", "Temp_Max_Media_5d_C", "Temp_Min_Media_5d_C", "Hum_Rel_Max_5d_Pct",
    "Hum_Rel_Min_5d_Pct", "Hum_Suelo_Pct", "Lluvia_Acum_5d_mm", "Horas_Humedad_Foliar",
    "GDD_Acumulado",
)
# Campos que el cliente puede enviar pero que se ignoran en la configuracion de produccion
# (exclude_gases=true) -- no son error si llegan, pero no participan en la inferencia.
IGNORED_GAS_COLS = ("CO2_ppm", "VOC_ppb")

# Regla de negocio de predictor.py::run_inference: si la clase ganadora es sana con alta
# confianza, el riesgo se atenua (no se re-sube nunca).
HEALTHY_CONFIDENCE_THRESHOLD = 0.80
HEALTHY_RISK_DAMPING = 0.6

# --- Hiperparametros de entrenamiento (unicamente los RF tabulares -- ver manifest.training.scope_note) ---
# src/training/train_tabular.py:117-137 -- identicos a get_params() de los RF entregados.
TRAIN_RF_CLASE_PARAMS = {
    "n_estimators": 200, "max_depth": 15, "min_samples_leaf": 20, "max_features": "log2",
    "bootstrap": True,
}
TRAIN_RF_RIESGO_PARAMS = {
    "n_estimators": 200, "max_depth": 30, "min_samples_leaf": 10, "max_features": "log2",
    "bootstrap": True,
}
TRAIN_RANDOM_STATE = 42
TRAIN_TEST_SIZE = 0.2  # src/training/train_tabular.py -- holdout 80/20

METRICS_REPORTED = {
    "clasificacion_deep_ensemble_produccion": {
        "accuracy": 0.9469, "f1_macro": 0.8807, "f1_weighted": 0.9463,
        "fuente": "models/metrics/classification/report_ensemble_final_no_gases_desacoplado.csv (auditado, support=1470)",
    },
    "clasificacion_rf_tabular_sin_gases": {
        "accuracy": 0.11593, "f1_macro": 0.1180,
        "fuente": "models/metrics/classification/report_rf_tabular_no_gases.csv (auditado, support=100000)",
        "nota": "No se usa para clasificar en produccion (modo desacoplado) -- ver known_issues.",
    },
    "regresion_riesgo_deep_ensemble_desacoplado_sin_gases": {
        "mae": 0.189, "mae_con_10pct_ruido_sensores": 0.188,
        "fuente": "memoria v5.0, Tabla 7 (sin CSV auditado propio, solo graficas PNG)",
    },
    "deteccion_yolo": {
        "precision": 0.995, "recall": 1.0, "map50": 0.995, "map50_95": 0.868,
        "fuente": "models/metrics/detection/YOLO/results.csv (auditado, epoca final 50/50)",
    },
    "kpi_contractual": "Accuracy >= 0.80 (memoria 6.1) -- CUMPLE (0.9469). F1-macro >= 0.85 -- CUMPLE por escaso margen (0.8807). Varias clases minoritarias no llegan a Recall >= 0.85 individualmente (EMPOASCA 0.588, LOBESIA 0.60, OIDIO_RACIMO 0.615).",
}
