"""Static configuration for the ml3 wine disease/pest Deep Ensemble plugin."""

# Los textos de tratamiento son literales de config.yaml (verbatim), no se pueden partir.
# pylint: disable=line-too-long

MODEL_ID = "ml3-wine-disease-pest-forecast"
ARTIFACT_FOLDER_NAME = "ml3_wine_disease_pest_forecast"

MODEL_FILENAMES = ["M1_LSTM.keras", "M2_CNN.keras", "M3_BiGRU.keras"]
SCALER_FILENAME = "scaler.pkl"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"

# User-retrained artifacts are persisted under these names locally and uploaded to
# MLflow with the canonical names — the fixed S3 artifacts are never overwritten.
USER_MODEL_FILENAMES = ["user_M1_LSTM.keras", "user_M2_CNN.keras", "user_M3_BiGRU.keras"]
USER_SCALER_FILENAME = "user_scaler.pkl"
USER_LABEL_ENCODER_FILENAME = "user_label_encoder.pkl"

FRAMEWORK = "tensorflow/keras/pandas/numpy/scikit-learn"
VERSION = "1.0.0"

# manifest.yaml -> inputs.window
WINDOW_SIZE = 168

# manifest.yaml -> inputs.fixed (raw sensor columns, in config.yaml -> raw_features order)
DATE_COLUMN = "Fecha"
RAW_FIXED_COLUMNS = [
    "Temp_Amb_C",
    "Hum_Rel_Pct",
    "Lluvia_mm",
    "Viento_kmh",
    "CO2_ppm",
    "VOC_ppb",
    "Hum_Suelo_Pct",
    "pH_Suelo",
]
SERIES_COLUMN = "ID_Serie"

# manifest.yaml -> inputs.derived_if_missing + config.yaml -> model_features (input final a la IA)
MODEL_FEATURES = [
    "Temp_Amb_C",
    "Hum_Rel_Pct",
    "Lluvia_mm",
    "Viento_kmh",
    "Horas_Humedad_Foliar",
    "GDD_Acumulado",
    "Hum_Suelo_Pct",
    "pH_Suelo",
    "CO2_ppm",
    "VOC_ppb",
    "Hora_Sin",
    "Hora_Cos",
]

# manifest.yaml -> outputs.predict_inline values (label_encoder.pkl order, clase fija)
CLASS_NAMES = [
    "ALTICA", "BLACK_ROT", "BOTRYTIS", "EMPOASCA", "ERINOSIS", "ESCA",
    "HEALTHY", "LOBESIA", "MILDIU", "OIDIO", "RED_MITE",
]

# manifest.yaml -> training.target_column / required_columns
TARGET_CLASS_COLUMN = "Clase_Entrenamiento"
TARGET_CLASS_LABEL = "Etiqueta_Num"
TARGET_REG_COLUMN = "Grado_Infeccion"
STRATIFY_COLUMN = "Etiqueta_Clase"

TRAIN_HARD_REQUIRED_COLUMNS = [DATE_COLUMN, SERIES_COLUMN, TARGET_CLASS_COLUMN, TARGET_REG_COLUMN] + RAW_FIXED_COLUMNS

# manifest.yaml -> training.hyperparams (config.yaml -> split)
TRAIN_TEST_SIZE = 0.30
TRAIN_VAL_TEST_RATIO = 0.50
TRAIN_RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 512
EARLY_STOPPING_PATIENCE = 2
LEARNING_RATES = {"lstm": 0.001, "cnn": 0.001, "bigru": 0.001}
ARCHITECTURES = {
    "lstm": {"lstm_1": 8, "dropout": 0.2, "lstm_2": 32},
    "cnn": {"filters_1": 16, "kernel_1": 3, "filters_2": 32, "kernel_2": 3, "dropout": 0.2},
    "bigru": {"gru_units": 16, "dropout": 0.2, "dense_units": 8},
}

# config.yaml -> base_conocimiento_tratamientos (la recomendación es editable). El tratamiento
# de cada clase se serializa igual que el run_inference original: " | ".join(f"{k}: {v}").
TREATMENT_KNOWLEDGE_BASE = {
    "HEALTHY": {
        "Diagnóstico": "Planta sana. No se detectan patrones de estrés biológico.",
        "Recomendación": "Mantener monitorización habitual. No requiere intervención.",
    },
    "LOBESIA": {
        "Químico": "Impregnar bien racimos. Bacillus thuringiensis (inicio eclosión), clorpirifos, fenoxicarb, flufenoxuron, metoxifenocide, tebufenocide.",
        "Biológico": "Crysopa carnea, Dybrachys spp., Coccinélidos (poca trascendencia en campaña). Mejor acción parasitaria en invierno sobre crisálidas.",
        "Biotecnológico": "Confusión sexual (Isonet L / Quant Lb) a 500/350 difusores/ha. Alternativa: Puffers (4-8 aerosoles/ha) sincronizados con vuelo.",
        "Cultural": "Poda en verde para ventilar racimos y exponerlos a insecticidas.",
    },
    "EMPOASCA": {
        "Químico": "Tratar focos de larvas/ninfas. Acrinatrin (uva de mesa), clorpirifos, flufenoxuron, imidacloprid.",
        "Biológico": "Anagrus atomus (baja eficiencia en vid).",
        "Biotecnológico": "Placas amarillas engomadas en perímetros e interior de parcela.",
        "Cultural": "Eliminar malas hierbas en invierno (hospedantes). Controlar el vigor para evitar brotación excesivamente tierna.",
    },
    "ALTICA": {
        "Químico": "Generalmente controlado por tratamientos para Lobesia. Específicos: Lambda cihalotrin y clorpirifos.",
        "Biológico": "Depredador Zicrona coerulea (chinche azul) y polífagos.",
        "Biotecnológico": "No establecido.",
        "Cultural": "Refugios trampa al final del verano (sacos/paja en el tronco) para capturar y destruir adultos invernantes. Mantener terreno limpio.",
    },
    "RED_MITE": {
        "Químico": "Mojar brotes/madera en invierno. En otoño, mojar haz de las hojas. Aceite mineral (ovicida), acrinatrin, dicofol, fenbutestan, piridaben.",
        "Biológico": "Fitoséidos espontáneos (ej. Phytoseiulus persimilis) si no se usan químicos agresivos.",
        "Biotecnológico": "No establecido.",
        "Cultural": "Eliminar puestas de invierno mediante poda. Controlar vigor. Evitar frutales aislados cercanos a la parcela.",
    },
    "ERINOSIS": {
        "Químico": "Raramente necesario. Azufre (mismo que Oidio) suele bastar. Dicofol en casos severos de raza de yemas (estados fenológicos C/D y G/H).",
        "Biológico": "Fitoséidos depredadores (Typhlodromus pyri, T. phialatus).",
        "Biotecnológico": "No establecido.",
        "Cultural": "Quemar restos de poda infectados. No usar yemas de plantas enfermas para reproducción.",
    },
    "ESCA": {
        "Químico": "Solo preventivo post-poda en heridas (cubiet, quinosol, tebuconazol+resinas). Inútil si la planta ya está contaminada.",
        "Biológico": "No establecido.",
        "Biotecnológico": "No establecido.",
        "Cultural": "Poda hasta madera sana y aplicar mastic. Podar cepas enfermas al final. Desinfectar herramientas. Marcar cepas en vegetativo. Técnica de abrir la cruz con cuña (aerobiosis).",
    },
    "OIDIO": {
        "Químico": "Mojar adecuadamente racimos. Poda/deshojado previo. Azufre y fungicidas antioídio específicos.",
        "Biológico": "No establecido.",
        "Biotecnológico": "No establecido.",
        "Cultural": "Deshojado basal del sarmiento (2-3 hojas). Eliminar brotes secundarios sin fruto. Control del vigor vegetativo excesivo.",
    },
    "MILDIU": {
        "Químico": "Hasta tamaño guisante: Sistémicos/penetrantes. Desde envero: De contacto. Preventivo al inicio de floración.",
        "Biológico": "No establecido.",
        "Biotecnológico": "No establecido.",
        "Cultural": "Buena aireación (deshojados y podas en verde). Controlar microclima en plantaciones bajo plástico.",
    },
    "BOTRYTIS": {
        "Químico": "Preventivos fijos o Método 15-15 (humedad/temp). Boscalida, ciprodinil, fenhexamida, fludioxonil, pirimetanil.",
        "Biológico": "No establecido.",
        "Biotecnológico": "No establecido.",
        "Cultural": "Deshojado manual/mecánico de racimos. Control estricto de polilla/oídio (puertas de entrada). Evitar exceso de nitrógeno.",
    },
    "BLACK_ROT": {
        "Químico": "Preventivo desde brotación a tamaño guisante. Alternar contacto (Mancozeb, folpet) con sistémicos (Miclobutanil, triazoles).",
        "Biológico": "No establecido.",
        "Biotecnológico": "Sistemas IoT / Modelos predictivos de humedad foliar (Spotts).",
        "Cultural": "Eliminación obligatoria y quema de 'momias' (uvas secas del año anterior). Deshojado/espergurado para aireación.",
    },
}

# manifest.yaml -> metrics_reported (ensemble DEL sobre test hold-out; usadas en /stats)
REPORTED_METRICS = {
    "dataset": "test_hold_out",
    "split": "70/15/15_por_serie",
    "random_seed": 42,
    "n_series_test": 149,
    "n_samples_test_windows": 224374,
    "classification": {
        "accuracy": 0.7052,
        "precision_macro": 0.7494,
        "recall_macro": 0.7198,
        "f1_macro": 0.7259,
        "f1_weighted": 0.7055,
    },
    "regression": {"mae": 0.0610, "rmse": 0.0936, "r2": 0.8903},
    "source": "models/metrics/DEL/metrics_DEL_test.csv + memoria 7.3 (Tablas 11 y 12)",
}
