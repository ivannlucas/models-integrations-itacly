"""«desconocido» for ml8: images unlike the ones the model was trained on.

The network has no "none of these" class: softmax always spreads 100 % over its 4 categories and
4 cereals, and on images from another domain (vine leaves, insects, grain photos…) it answers
«insectos» with trigo/maíz at 0.9+ confidence, so a confidence threshold cannot catch them
(measured: it flags 1-7 % of vine leaves, ticks or insects at 95 % of real images kept).

The gate works on the network's own 960-d pooled MobileNetV3 features, the input of both heads:

* ``knn``: cosine distance to the nearest training image (deep k-NN, Sun et al., ICML 2022);
* ``logreg``: a logistic regression (outlier exposure, Hendrycks et al., ICLR 2019) trained on
  those features + the k-NN distance, real training images vs images of other domains
  (``ood_outliers.zip``: vine leaves, insects, ticks, poultry, thermal, cattle, grain pests).
  Evaluated leaving each outlier domain out of training, it catches 2-3x more of that unseen
  domain than k-NN alone at the same share of real images kept (outputs/a08/README_ood.md).

The generic gate lives in app/domain/services/feature_ood.py (shared with ml10). The threshold
keeps a fixed share (95 %) of real images that were not used to fit the gate.
Above it the image is reported as «desconocido» and the raw network labels are kept apart.
The classifier itself is untouched: same weights, same predictions for in-distribution images.
"""
from __future__ import annotations

import torch

from app.domain.services.feature_ood import (  # re-exported for the plugin and the script
    DEFAULT_KEEP,
    OOD_REFERENCE_FILENAME,
    UNKNOWN_LABEL,
    OodReference,
    build_reference,
    fit_reference,
)
from app.domain.services.feature_ood import load_reference as _load_reference

__all__ = [
    "DEFAULT_KEEP", "OOD_OUTLIERS_FILENAME", "OOD_REFERENCE_FILENAME", "UNKNOWN_LABEL",
    "OodReference", "apply_gate", "build_reference", "fit_reference", "forward_with_features",
    "load_reference",
]

OOD_OUTLIERS_FILENAME = "ood_outliers.zip"


def load_reference(folder: str) -> OodReference | None:
    """The ml8 OodReference stored in *folder*, or None (gate disabled) if it is not there."""
    return _load_reference(folder, "ml8")


def forward_with_features(model: torch.nn.Module, x: torch.Tensor):
    """(logits_categoria, logits_cereal, pooled features) in one pass.

    Same computation as MultiTaskMobileNetV3Large.forward, also returning the pooled features
    the gate scores.
    """
    feats = torch.flatten(model.avgpool(model.features(x)), 1)
    hidden = model.neck(feats)
    return model.classifier_categoria(hidden), model.classifier_cereal(hidden), feats


def apply_gate(result: dict, reference: OodReference | None, features: torch.Tensor) -> dict:
    """Add the «desconocido» fields to a prediction dict (in place) and return it.

    ``categoria``/``cereal`` become «desconocido» above the threshold; the raw network labels
    move to ``categoria_modelo``/``cereal_modelo``. Confidences and probabilities stay the raw
    network outputs. Without a reference (e.g. a retrained run from before this gate) the
    prediction is returned as before, with ``es_desconocido`` False and null score.
    """
    result["categoria_modelo"] = result["categoria"]
    result["cereal_modelo"] = result["cereal"]
    if reference is None:
        result.update(es_desconocido=False, puntuacion_ood=None, umbral_ood=None)
        return result
    score = float(reference.score(features.detach().cpu().numpy())[0])
    unknown = score > reference.threshold
    result.update(es_desconocido=unknown, puntuacion_ood=round(score, 6),
                  umbral_ood=round(reference.threshold, 6))
    if unknown:
        result["categoria"] = UNKNOWN_LABEL
        result["cereal"] = UNKNOWN_LABEL
    return result
