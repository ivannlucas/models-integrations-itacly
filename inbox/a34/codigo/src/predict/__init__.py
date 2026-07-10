from src.predict.inference import predict_with_model, predict_batch, save_predictions
from src.predict.optimization import (
    setup_ga_toolbox,
    fitness_consumo_especifico,
    run_ga_single,
)
from src.predict.lookup import query_setpoints
