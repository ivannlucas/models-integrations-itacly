import numpy as np
import torch

from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.constants import (
    COMPONENT_NAMES,
    STATE_LABELS,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.preprocessing import pad_or_truncate


def run_inference(model, scaler, feature_cols, x_df, device) -> dict | None:
    try:
        df_to_scale = x_df[feature_cols]
    except KeyError:
        return None

    X_scaled = scaler.transform(df_to_scale)
    X_ready = pad_or_truncate(X_scaled)
    input_tensor = (
        torch.tensor(X_ready, dtype=torch.float32)
        .unsqueeze(0)
        .permute(0, 2, 1)
        .to(device)
    )

    with torch.no_grad():
        p_foul, p_valv, p_bomb, p_acum = model(input_tensor)
        probs = [torch.softmax(p, dim=1) for p in [p_foul, p_valv, p_bomb, p_acum]]
        preds = [p.argmax(1).item() for p in probs]
        confidences = [p.max(1).values.item() for p in probs]

    return {"predicciones": preds, "confianzas": confidences}


def format_inline_response(model_id: str, resultados: dict) -> dict:
    preds = resultados["predicciones"]
    confs = resultados["confianzas"]
    return {
        "model_id": model_id,
        COMPONENT_NAMES[0]: preds[0],
        COMPONENT_NAMES[1]: preds[1],
        COMPONENT_NAMES[2]: preds[2],
        COMPONENT_NAMES[3]: preds[3],
        "Confianza_Fouling": confs[0],
        "Confianza_Valvula": confs[1],
        "Confianza_Bomba": confs[2],
        "Confianza_Acumulador": confs[3],
        "model_name": model_id,
    }


def format_batch_row(resultados: dict, cycle_id) -> dict:
    preds = resultados["predicciones"]
    confs = resultados["confianzas"]
    row: dict = {}
    if cycle_id is not None:
        row["Cycle_ID"] = int(cycle_id) if isinstance(cycle_id, np.integer) else cycle_id
    for name, pred in zip(COMPONENT_NAMES, preds):
        row[name] = int(pred)
        row[f"{name}_Texto"] = STATE_LABELS[int(pred)]
    row["Confianza_Fouling"] = confs[0]
    row["Confianza_Valvula"] = confs[1]
    row["Confianza_Bomba"] = confs[2]
    row["Confianza_Acumulador"] = confs[3]
    row["model_name"] = "m47-dnsl-fallas-maquinaria-pasteurizado"
    return row
