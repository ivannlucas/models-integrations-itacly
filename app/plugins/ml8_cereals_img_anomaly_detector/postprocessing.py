import torch
import torch.nn.functional as F


def build_inline_response(
    logits_categoria: torch.Tensor,
    logits_cereal: torch.Tensor,
    idx_to_class: dict,
    idx_to_cereal: dict,
    model_id: str,
) -> dict:
    prob_cat = F.softmax(logits_categoria.detach(), dim=-1).squeeze(0)
    prob_cer = F.softmax(logits_cereal.detach(), dim=-1).squeeze(0)

    idx_cat = int(prob_cat.argmax().item())
    idx_cer = int(prob_cer.argmax().item())

    return {
        "model_id": model_id,
        "categoria": idx_to_class[idx_cat],
        "cereal": idx_to_cereal[idx_cer],
        "confianza_categoria": round(float(prob_cat[idx_cat].item()), 6),
        "confianza_cereal": round(float(prob_cer[idx_cer].item()), 6),
        "probabilidades_categoria": {
            idx_to_class[i]: round(float(prob_cat[i].item()), 6)
            for i in range(len(prob_cat))
        },
        "probabilidades_cereal": {
            idx_to_cereal[i]: round(float(prob_cer[i].item()), 6)
            for i in range(len(prob_cer))
        },
    }


def build_batch_response(predictions: list[dict], model_id: str) -> dict:
    return {
        "model_id": model_id,
        "predictions": predictions,
        "output_path": None,
    }
