"""Genera la referencia de «desconocido» de ml8 (ood_reference.npz + ood_outliers.zip).

ml8 no tiene clase «ninguna de estas»: con imágenes de otro dominio (hojas de vid, insectos,
grano…) responde «insectos» con trigo/maíz y >0,9 de confianza. El filtro de
app/plugins/ml8_cereals_img_anomaly_detector/ood.py decide «desconocido» con las características
internas de la red (960-d, entrada de las dos cabezas). Este script lo calibra para el modelo
base:

1. Imágenes reales: la única copia accesible del dataset, s3://<STORAGE_BUCKET>/datasets/modelo-8/
   ({cereal}/train/{categoria}/, 50 por combinación, 600). O un directorio local con --id-dir.
2. Imágenes de otros dominios (outliers), de los datos reales de otros modelos del repo
   (inbox/aNN/codigo/, no commiteado, y outputs/aNN/datos_prueba/): vid (a01, a02), heces de ave
   (a06), moscas/mosquitos, garrapatas e insectos sobre ganado (a10), vacas (a05), térmicas de
   ubre (a04) y plagas del grano (a07). Hasta --per-domain por dominio, reescaladas a 256 px, en
   ood_outliers.zip: el reentrenamiento las usa para recalibrar el filtro de su modelo.
3. Regresión logística (imágenes reales vs outliers) sobre esas características + distancia
   k-NN. Umbral: el cuantil --keep (95 %) de la puntuación de imágenes reales que no se usaron
   para ajustar, por validación cruzada de 5 particiones.
4. --evaluate: además, deja cada dominio fuera del ajuste y mide cuánto detecta de él (dominios
   no vistos) frente a la distancia k-NN sola, y la tasa de «desconocido» en imágenes reales de
   sorgo no usadas (Sorghum Disease Image Dataset, una de las fuentes del dataset de ml8).

Salida en artifacts/ml8_cereals_img_anomaly_detector/ (no se commitea). Subir los dos ficheros a
s3://<STORAGE_BUCKET>/artifacts/fixed/ml8_cereals_img_anomaly_detector/ junto al checkpoint (no
se sobrescribe nada: son ficheros nuevos). Sin ellos el plugin sirve como antes.

Uso: .venv/bin/python outputs/a08/generar_referencia_ood.py [--id-dir DIR] [--evaluate]
"""
from __future__ import annotations

import argparse
import io
import os
import random
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
from app.plugins.ml8_cereals_img_anomaly_detector.constants import (  # noqa: E402
    ARTIFACT_FOLDER_NAME,
    IMAGE_EXTENSIONS,
)
from app.plugins.ml8_cereals_img_anomaly_detector.model_loader import load_model_bundle  # noqa: E402
from app.plugins.ml8_cereals_img_anomaly_detector.ood import (  # noqa: E402
    DEFAULT_KEEP,
    OOD_OUTLIERS_FILENAME,
    OOD_REFERENCE_FILENAME,
    fit_reference,
)
from app.plugins.ml8_cereals_img_anomaly_detector.preprocessing import (  # noqa: E402
    image_path_to_tensor_and_image,
)
from app.plugins.ml8_cereals_img_anomaly_detector.ood import forward_with_features  # noqa: E402

A10 = "inbox/a10/codigo/a10-del-dairy-deteccion-vectores-enfermedades-ganado-vacuno/data"
OUTLIER_DOMAINS = {
    "vid_hojas_a01": "inbox/a01/codigo/data/raw/IMAGENES",
    "vid_hojas_a02": "inbox/a02/codigo/data/raw/datasets",
    "aves_heces_a06": "inbox/a06/codigo/data/fecal/raw",
    "moscas_mosquitos_a10": f"{A10}/classification/raw/fly_mos/FLY_MOS_Dataset",
    "garrapatas_a10": f"{A10}/classification/raw/ticks",
    "insectos_ganado_a10": f"{A10}/detection/raw/insects",
    "vacas_a05": "inbox/a05/codigo/data/images/raw_frames",
    "termica_ubre_a04": "outputs/a04/datos_prueba/predict_batch_golden.zip",
    "plagas_grano_a07": "outputs/a07/datos_prueba/predict_batch_golden.zip",
}
SORGHUM_ID = f"{A10}/classification/raw/fly_mos/Sorghum Disease Image Dataset"
OUTLIER_MAX_SIDE = 256


def _env() -> dict:
    env = dict(os.environ)
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if m and m.group(1) not in env:
            env[m.group(1)] = m.group(2).strip().strip('"')
    return env


def download_id_images(dest: Path) -> Path:
    """s3://<bucket>/datasets/modelo-8/ -> dest (skips files already there)."""
    import boto3  # pylint: disable=import-outside-toplevel

    env = _env()
    s3 = boto3.client("s3", endpoint_url=env["CUSTOM_S3_ENDPOINT"], region_name=env.get("CUSTOM_REGION"),
                      aws_access_key_id=env.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY"))
    prefix = "datasets/modelo-8/"
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=env["STORAGE_BUCKET"], Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if "/" not in rel or rel.endswith("/") or obj["Size"] == 0:
                continue
            target = dest / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(env["STORAGE_BUCKET"], obj["Key"], str(target))
    return dest


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _domain_images(rel: str, per_domain: int, rng: random.Random, tmp: Path) -> list[Path]:
    src = REPO / rel
    if not src.exists():
        print(f"  (no disponible: {rel})")
        return []
    if src.suffix == ".zip":
        out = tmp / src.parent.parent.name
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as zf:
            for m in zf.namelist():
                if Path(m).suffix.lower() in IMAGE_EXTENSIONS and "__MACOSX" not in m:
                    (out / Path(m).name).write_bytes(zf.read(m))
        files = _images(out)
    else:
        files = _images(src)
    return rng.sample(files, min(per_domain, len(files)))


def _resized_jpeg(path: Path) -> bytes:
    img = Image.open(path).convert("RGB")
    img.thumbnail((OUTLIER_MAX_SIDE, OUTLIER_MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_outliers_zip(zip_path: Path, per_domain: int) -> dict[str, list[str]]:
    """Write ood_outliers.zip (<dominio>/<n>.jpg) and return its member names per domain."""
    rng = random.Random(0)
    members: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for domain, rel in OUTLIER_DOMAINS.items():
            names = []
            for i, path in enumerate(_domain_images(rel, per_domain, rng, Path(tmp))):
                try:
                    data = _resized_jpeg(path)
                except Exception:  # pylint: disable=broad-exception-caught  # unreadable: skip
                    continue
                name = f"{domain}/{i:04d}.jpg"
                zf.writestr(name, data)
                names.append(name)
            members[domain] = names
            print(f"  {domain}: {len(names)}")
    return members


def features(model, device, paths) -> np.ndarray:
    """Pooled features with the plugin's inference preprocessing (same as /predict)."""
    out = []
    with torch.no_grad():
        for p in paths:
            tensor, _ = image_path_to_tensor_and_image(p)
            out.append(forward_with_features(model, tensor.to(device))[2].cpu().numpy()[0])
    return np.array(out, dtype=np.float32)


def cv_threshold_scores(id_feats, id_labels, outliers, keep) -> np.ndarray:
    """Scores of each real image by a gate fitted without it (5-fold, stratified)."""
    scores = np.zeros(len(id_feats))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(id_feats, id_labels):
        scores[te] = fit_reference(id_feats[tr], outliers, keep).score(id_feats[te])
    return scores


def evaluate(id_feats, id_labels, by_domain, sorghum, keep) -> list[str]:
    """Leave-one-domain-out detection (logreg vs k-NN) and false «desconocido» on sorghum."""
    domains = [d for d in by_domain if len(by_domain[d])]
    lines = ["| Dominio dejado fuera | k-NN solo | Filtro (logreg + k-NN) |", "|---|---|---|"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(id_feats, id_labels))
    sorghum_rates = []
    for held in domains:
        others = np.concatenate([by_domain[d] for d in domains if d != held])
        knn_rates, lr_rates = [], []
        for tr, te in folds:
            for fit_outliers, rates in ((None, knn_rates), (others, lr_rates)):
                ref = fit_reference(id_feats[tr], fit_outliers, keep)
                ref = ref.with_threshold(ref.score(id_feats[te]))
                rates.append(float((ref.score(by_domain[held]) > ref.threshold).mean()))
                if fit_outliers is not None and held == domains[0] and len(sorghum):
                    sorghum_rates.append(float((ref.score(sorghum) > ref.threshold).mean()))
        lines.append(f"| {held} | {np.mean(knn_rates):.0%} | {np.mean(lr_rates):.0%} |")
    lines.append("")
    lines.append(f"Imágenes reales de sorgo no usadas marcadas como «desconocido»: "
                 f"{np.mean(sorghum_rates):.1%} ({len(sorghum)} imágenes).")
    return lines


def main() -> int:
    """Build ood_outliers.zip and ood_reference.npz for the base ml8 checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id-dir", type=Path, help="Imágenes reales {cereal}/train/{categoria}/ (por defecto, S3)")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / ARTIFACT_FOLDER_NAME)
    ap.add_argument("--per-domain", type=int, default=200)
    ap.add_argument("--keep", type=float, default=DEFAULT_KEEP)
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    bundle = load_model_bundle()
    model, device = bundle["model"], bundle["device"]
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        id_dir = args.id_dir or download_id_images(Path(tmp) / "id")
        id_paths = _images(id_dir)
        id_labels = np.array(["/".join(p.relative_to(id_dir).parts[::2][:2]) for p in id_paths])
        print(f"Imágenes reales: {len(id_paths)}")
        id_feats = features(model, device, id_paths)

        zip_path = args.out / OOD_OUTLIERS_FILENAME
        print(f"Outliers -> {zip_path}")
        members = build_outliers_zip(zip_path, args.per_domain)
        ext = Path(tmp) / "outliers"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ext)
        by_domain = {d: features(model, device, [ext / n for n in names]) for d, names in members.items()}
        outliers = np.concatenate([f for f in by_domain.values() if len(f)])

        held_out = cv_threshold_scores(id_feats, id_labels, outliers, args.keep)
        reference = fit_reference(id_feats, outliers, args.keep).with_threshold(held_out)
        reference.save(str(args.out / OOD_REFERENCE_FILENAME))
        print(f"{OOD_REFERENCE_FILENAME}: método={reference.method} umbral={reference.threshold:.4f} "
              f"(conserva el {args.keep:.0%} de las imágenes reales)")
        print(f"Outliers del ajuste por encima del umbral: {(reference.score(outliers) > reference.threshold).mean():.1%}")

        if args.evaluate:
            rng = random.Random(1)
            sorghum_paths = rng.sample(_images(REPO / SORGHUM_ID), 200) if (REPO / SORGHUM_ID).exists() else []
            sorghum = features(model, device, sorghum_paths) if sorghum_paths else np.zeros((0, id_feats.shape[1]))
            print("\n".join(evaluate(id_feats, id_labels, by_domain, sorghum, args.keep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
