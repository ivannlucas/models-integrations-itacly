import base64
import io

import torch
import torch.nn.functional as F
from PIL import Image

# Same sizing/quality as ml7_cereals_grain_pest_detection's annotated_image, for a
# consistent per-image payload size across the batch grids in the platform UI.
MAX_THUMBNAIL_SIZE = 640
JPEG_QUALITY = 70


def encode_image_base64(image: Image.Image) -> str:
    """Resize (if needed) and JPEG-encode a decoded image as base64.

    ml8 is a pure classifier with no bounding boxes to draw, so — unlike ml7's
    annotated_image — this just re-encodes the already-decoded image instead of
    reopening it from disk, so the platform's batch results grid can show a
    per-row thumbnail the same way it does for grain_pest_detection.
    """
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_THUMBNAIL_SIZE:
        scale = MAX_THUMBNAIL_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


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
