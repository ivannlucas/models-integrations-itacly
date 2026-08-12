"""Vendored verbatim from inbox/a45/codigo/src/xai/utils.py."""
from __future__ import annotations

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple


def safe_float(x: Any, default: float = 0.0) -> float:
    """Convierte a float de manera segura."""
    try:
        return float(x)
    except Exception:
        return float(default)


def sigmoid(x: float) -> float:
    """Aplica sigmoid."""
    return 1.0 / (1.0 + np.exp(-float(x)))


def split_feature_base_and_stat(
    feature_name: str,
    stats: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Separa nombre base y estadístico derivado.
    Acepta tanto mean_x como x_mean.
    """
    derived_stats = tuple(
        str(s) for s in (stats or ("mean", "std", "range", "diff", "slope"))
    )
    name = str(feature_name)
    for stat in derived_stats:
        pref = f"{stat}_"
        if name.startswith(pref):
            return name[len(pref):], stat
    for stat in derived_stats:
        suf = f"_{stat}"
        if name.endswith(suf):
            return name[:-len(suf)], stat
    return name, "raw_or_unknown"


def get_semantic_mf_name_for_feature(
    mf_semantics: Dict[str, Any],
    feature_name: str,
    mf_idx: int,
) -> str:
    """Obtiene nombre semántico de una función de pertenencia."""
    lookup = mf_semantics.get("semantic_name_lookup", {})
    return lookup.get(str(feature_name), {}).get(int(mf_idx), f"mf_{mf_idx}")


def infer_semantic_mf_names_from_centers(
    model,
    feature_names: Optional[List[str]] = None,
    semantic_names_map: Optional[Dict[int, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Asigna nombres semánticos relativos a las MFs de cada feature
    ordenándolas por centro ascendente.
    """
    if not hasattr(model, "fuzzy_layer"):
        raise AttributeError("El modelo no tiene atributo 'fuzzy_layer'.")

    fl = model.fuzzy_layer
    if not hasattr(fl, "centers"):
        raise AttributeError("model.fuzzy_layer no tiene atributo 'centers'.")

    centers = fl.centers
    sigmas = getattr(fl, "sigmas", None)

    with torch.no_grad():
        centers = centers.detach().cpu().numpy() if torch.is_tensor(centers) else np.asarray(centers)
        sigmas = (
            sigmas.detach().cpu().numpy() if (sigmas is not None and torch.is_tensor(sigmas))
            else (np.asarray(sigmas) if sigmas is not None else None)
        )

    if centers.ndim != 2:
        raise ValueError(f"Se esperaban centros con forma [F, M]. Recibido: {centers.shape}")

    F_dim, M = centers.shape

    if feature_names is None or len(feature_names) != F_dim:
        feature_names = [f"feature_{i}" for i in range(F_dim)]

    if semantic_names_map is None:
        semantic_names_map = {
            3: ["bajo", "medio", "alto"],
            5: ["muy_bajo", "bajo", "medio", "alto", "muy_alto"],
        }

    semantic_labels = semantic_names_map.get(M, [f"nivel_{i}" for i in range(M)])

    global_mf_names_by_feature = {}
    semantic_name_lookup = {}

    for f_idx, feat_name in enumerate(feature_names):
        order = np.argsort(centers[f_idx])
        feature_info = []
        lookup = {}

        for pos, mf_idx in enumerate(order):
            sem_name = semantic_labels[pos]
            item = {
                "mf_idx": int(mf_idx),
                "semantic_name": sem_name,
                "center": float(centers[f_idx, mf_idx]),
                "ordered_position": int(pos),
            }
            if sigmas is not None:
                item["sigma"] = float(sigmas[f_idx, mf_idx])

            feature_info.append(item)
            lookup[int(mf_idx)] = sem_name

        global_mf_names_by_feature[str(feat_name)] = feature_info
        semantic_name_lookup[str(feat_name)] = lookup

    return {
        "global_mf_names_by_feature": global_mf_names_by_feature,
        "semantic_name_lookup": semantic_name_lookup,
        "centers": centers,
        "sigmas": sigmas,
    }
