"""Genera la referencia de «desconocido» de ml10 (ood_reference.npz + ood_crops.zip).

ml10 detecta → recorta → clasifica, y ninguna etapa tiene clase «ninguna de estas»: con fotos de
otro dominio (hojas de vid, heces de ave, vacas, grano…) el detector dispara y el clasificador
etiqueta el recorte como fly/mos/tick con ~0,9 de confianza. El filtro de
app/plugins/ml10_dairy_disease_vector_detection/ood.py decide «desconocido» por recorte con las
características internas del clasificador (960-d). Este script lo calibra para el modelo base:

1. Recortes reales del detector: el pipeline del plugin (det_conf 0,2, cls_conf 0,5) quedándose
   con los que coinciden con una caja anotada (IoU >= 0,5), sobre
   a. las imágenes de detección de a10 (data/detection/splits/{train,val,test}) →
      ood_crops.zip id_train/deteccion/, id_val/deteccion/ (test solo en --evaluate);
   b. las fotos originales de clasificación (data/classification/raw/{ticks,fly_mos}), según el
      split en el que acabaron sus recortes procesados (<foto>_cN.jpg, tick_<foto>_cN.jpg) →
      id_train/fotos/, id_val/fotos/; las de test y las que no llegaron a ningún split, solo en
      --evaluate. Sin ellas, el 8 % de las fotos reales no usadas salía «desconocido»: el
      recorte del detector no es el recorte procesado con margen con el que se entrenó.
2. Recortes de otros dominios (outliers): el mismo pipeline sobre imágenes de otros modelos del
   repo (inbox/aNN/codigo/, no commiteado, y outputs/aNN/datos_prueba/), en
   ood_crops.zip (outliers/<dominio>/).
3. Referencia = ood.build_crops_reference, la misma función que usa /train: banco = recortes de
   clasificación de train + id_train; umbral = cuantil 95 % de la puntuación de los de validación +
   id_val; regresión logística banco vs outliers.
4. --evaluate: tasa de «desconocido» en recortes reales de test (clasificación y detector), en
   falsos positivos del detector, en cada dominio dejándolo fuera del ajuste (dominios no
   vistos) frente a la distancia k-NN sola, y en gráficos/capturas (nunca usados en el ajuste).

Los recortes se guardan con el lado corto <= 256 px: la preprocesado de inferencia
(Resize(256) + CenterCrop(224)) da prácticamente lo mismo.

Salida en artifacts/ml10_dairy_disease_vector_detection/ (no se commitea). Subir los dos
ficheros a s3://<STORAGE_BUCKET>/artifacts/fixed/ml10_dairy_disease_vector_detection/ junto al
clasificador (no se sobrescribe nada: son ficheros nuevos). Sin ellos el plugin sirve como antes.

Uso: .venv/bin/python outputs/a10/generar_referencia_ood.py [--evaluate] [--work-dir DIR]
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
from app.domain.services.feature_ood import (  # noqa: E402
    DEFAULT_KEEP,
    OOD_REFERENCE_FILENAME,
    build_reference,
)
from app.plugins.ml10_dairy_disease_vector_detection.constants import (  # noqa: E402
    ARTIFACT_FOLDER_NAME,
    DEFAULT_CLS_CONF,
    DEFAULT_DET_CONF,
    SUPPORTED_EXTENSIONS,
)
from app.plugins.ml10_dairy_disease_vector_detection.ood import (  # noqa: E402
    ID_TRAIN_DIR,
    ID_VAL_DIR,
    OOD_CROPS_FILENAME,
    OUTLIERS_DIR,
    build_crops_reference,
    crop_features,
    image_files,
)
from app.plugins.ml10_dairy_disease_vector_detection.plugin import (  # noqa: E402
    Ml10DairyDiseaseVectorDetectionPlugin,
    _Classifier,
)

A10 = REPO / "inbox/a10/codigo/a10-del-dairy-deteccion-vectores-enfermedades-ganado-vacuno/data"
CLS_SPLITS = A10 / "classification/splits"
DET_SPLITS = A10 / "detection/splits"
# Original classification photos: (folder, YOLO classes kept). fly_mos labels: 0 fly, 1 mos (the
# rest are not vectors, as in a10's preprocess.py); tick labels are polygons, all kept.
CLS_PHOTOS = (
    (A10 / "classification/raw/ticks", None),
    (A10 / "classification/raw/fly_mos/FLY_MOS_Dataset", {"0", "1"}),
)
PHOTO_SPLITS = {"train": f"{ID_TRAIN_DIR}/fotos", "val": f"{ID_VAL_DIR}/fotos",
                "test": "fotos_test", "sin_usar": "fotos_sin_usar"}
# Other domains used to fit the regression (and, one at a time, left out in --evaluate).
OUTLIER_DOMAINS = {
    "vid_hojas_a01": REPO / "inbox/a01/codigo/data/raw/IMAGENES",
    "vid_hojas_a02": REPO / "inbox/a02/codigo/data/raw/datasets",
    "aves_heces_a06": REPO / "inbox/a06/codigo/data/fecal/raw",
    "vacas_a05": REPO / "inbox/a05/codigo/data/images/raw_frames",
    # Delivered by the a10 team inside fly_mos/ but never used to train ml10.
    "sorgo_hojas": A10 / "classification/raw/fly_mos/Sorghum Disease Image Dataset",
    "termica_ubre_a04": REPO / "outputs/a04/datos_prueba/predict_batch_golden.zip",
    "plagas_grano_a07": REPO / "outputs/a07/datos_prueba/predict_batch_golden.zip",
    "cereales_a08": REPO / "outputs/a08/datos_prueba/predict_batch_12.zip",
}
# Evaluation only (never in the fit): charts and screenshots of other deliveries.
UNSEEN_DOMAINS = {
    "graficos": [REPO / f"inbox/{a}/codigo" for a in ("a17", "a19", "a21", "a26", "a28", "a41")],
}
CROP_MIN_SIDE = 256
IOU_MATCH = 0.5


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _open(source) -> Image.Image | None:
    try:
        return Image.open(source).convert("RGB")
    except Exception:  # pylint: disable=broad-exception-caught  # unreadable: skip
        return None


def _domain_images(src: Path, per_domain: int, rng: random.Random):
    """Up to *per_domain* PIL images of a folder or ZIP (seeded sample)."""
    if not src.exists():
        print(f"  (no disponible: {src.relative_to(REPO)})")
        return []
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            names = sorted(m for m in zf.namelist()
                           if Path(m).suffix.lower() in SUPPORTED_EXTENSIONS and "__MACOSX" not in m)
            names = rng.sample(names, min(per_domain, len(names)))
            images = [_open(io.BytesIO(zf.read(m))) for m in names]
    else:
        files = _images(src)
        images = [_open(p) for p in rng.sample(files, min(per_domain, len(files)))]
    return [im for im in images if im is not None]


def _crop_jpeg(image: Image.Image, bbox: dict) -> bytes:
    crop = image.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
    short = min(crop.size)
    if short > CROP_MIN_SIDE:
        scale = CROP_MIN_SIDE / short
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                           Image.BILINEAR)  # pylint: disable=no-member
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _iou(a: dict, b: list[float]) -> float:
    ix = max(0.0, min(a["x2"], b[2]) - max(a["x1"], b[0]))
    iy = max(0.0, min(a["y2"], b[3]) - max(a["y1"], b[1]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _gt_boxes(image_path: Path, size: tuple[int, int]) -> list[list[float]]:
    """Labelled boxes (YOLO txt: class cx cy w h, normalised) in pixels."""
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label.exists():
        return []
    w, h = size
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) >= 5:
            cx, cy, bw, bh = (float(x) for x in v[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def _photo_split() -> dict[str, set[str]]:
    """Splits where the processed crops of each original photo ended (key: photo stem)."""
    splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        for crop in _images(CLS_SPLITS / split):
            stem = crop.stem.rsplit("_c", 1)[0].removeprefix("tick_")
            splits.setdefault(stem, set()).add(split)
    return splits


def _photo_boxes(photo: Path, size: tuple[int, int], classes: set[str] | None) -> list[list[float]]:
    """Labelled vectors of an original photo in pixels (YOLO bbox, or polygon → its box)."""
    for label in (photo.with_suffix(".txt"),
                  Path(str(photo.parent).replace("images", "labels")) / f"{photo.stem}.txt"):
        if label.exists():
            break
    else:
        return []
    w, h = size
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) < 5 or (classes is not None and v[0] not in classes):
            continue
        c = [float(x) for x in v[1:]]
        if len(c) == 4:
            cx, cy, bw, bh = c
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        else:
            xs, ys = c[0::2], c[1::2]
            boxes.append([min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h])
    return boxes


class CropMaker:
    """The plugin's own pipeline (no gate) cutting the crops the gate will see at inference."""

    def __init__(self) -> None:
        self.plugin = Ml10DairyDiseaseVectorDetectionPlugin()
        self.plugin.load()
        self.classifier = self.plugin._classifier  # pylint: disable=protected-access
        self.device = self.plugin._device  # pylint: disable=protected-access
        self._clf = _Classifier(self.classifier, self.plugin._class_names, None)  # pylint: disable=protected-access

    def detections(self, image: Image.Image) -> list[dict]:
        # pylint: disable-next=protected-access
        return self.plugin._run_pipeline(image, DEFAULT_DET_CONF, DEFAULT_CLS_CONF, self._clf)

    def detector_crops(self, split: str) -> tuple[list[bytes], list[bytes]]:
        """(crops matching a labelled box, crops that do not) of a detection split."""
        matched, unmatched = [], []
        paths = _images(DET_SPLITS / split / "images")
        for i, path in enumerate(paths):
            if i % 500 == 0:
                print(f"    {split}: {i}/{len(paths)} imágenes", flush=True)
            image = _open(path)
            if image is None:
                continue
            gt = _gt_boxes(path, image.size)
            for det in self.detections(image):
                hit = any(_iou(det["bbox"], g) >= IOU_MATCH for g in gt)
                (matched if hit else unmatched).append(_crop_jpeg(image, det["bbox"]))
        return matched, unmatched

    def photo_crops(self) -> dict[str, list[bytes]]:
        """Crops matching a labelled vector in the original classification photos, by split.

        A photo whose crops ended in more than one split is skipped (none in practice).
        """
        split_of = _photo_split()
        crops: dict[str, list[bytes]] = {split: [] for split in PHOTO_SPLITS}
        for root, classes in CLS_PHOTOS:
            paths = _images(root)
            for i, path in enumerate(paths):
                if i % 2000 == 0:
                    print(f"    fotos {root.name}: {i}/{len(paths)}", flush=True)
                splits = split_of.get(path.stem, set())
                if len(splits) > 1:
                    continue
                image = _open(path)
                if image is None:
                    continue
                gt = _photo_boxes(path, image.size, classes)
                for det in self.detections(image):
                    if any(_iou(det["bbox"], g) >= IOU_MATCH for g in gt):
                        crops[next(iter(splits), "sin_usar")].append(_crop_jpeg(image, det["bbox"]))
        return crops

    def domain_crops(self, images, max_crops: int, rng: random.Random) -> list[bytes]:
        crops = [_crop_jpeg(im, det["bbox"]) for im in images for det in self.detections(im)]
        return rng.sample(crops, min(max_crops, len(crops)))


def _write(folder: Path, crops: list[bytes]) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, data in enumerate(crops):
        path = folder / f"{i:05d}.jpg"
        path.write_bytes(data)
        paths.append(path)
    return paths


def _stage(folder: Path, make) -> int:
    """Write the crops of ``make()`` to *folder* once; a rerun with --work-dir reuses them."""
    done = folder / ".done"
    if not done.exists():
        _write(folder, make())
        done.touch()
    return len(image_files(folder))


def _rate(ref, feats: np.ndarray) -> str:
    return f"{(ref.score(feats) > ref.threshold).mean():.1%}" if len(feats) else "—"


def evaluate(maker: CropMaker, crops_dir: Path, eval_dir: Path, keep: float) -> list[str]:
    """Markdown table: share flagged «desconocido» per set (k-NN only / domain left out / final)."""
    feats = lambda paths: crop_features(maker.classifier, paths, maker.device)  # noqa: E731
    bank = np.r_[feats(_images(CLS_SPLITS / "train")), feats(image_files(crops_dir / ID_TRAIN_DIR))]
    held = np.r_[feats(_images(CLS_SPLITS / "val")), feats(image_files(crops_dir / ID_VAL_DIR))]
    domains = {d.name: feats(image_files(d)) for d in sorted((crops_dir / OUTLIERS_DIR).iterdir())}
    knn = build_reference(bank, held, keep=keep)
    final = build_reference(bank, held, np.concatenate(list(domains.values())), keep)

    lines = [f"Banco {len(bank)} recortes, validación {len(held)}, outliers "
             f"{sum(len(f) for f in domains.values())}; umbral final {final.threshold:.4f}.", "",
             "| Conjunto | Recortes | Solo k-NN | Dominio fuera del ajuste | Filtro final |",
             "|---|---|---|---|---|"]
    real = {
        "Reales: clasificación test": feats(_images(CLS_SPLITS / "test")),
        "Reales: detector test (caja anotada)": feats(image_files(eval_dir / "det_test_tp")),
        "Reales: fotos de clasificación de test": feats(image_files(eval_dir / "fotos_test")),
        "Reales: fotos de clasificación sin usar en ningún split":
            feats(image_files(eval_dir / "fotos_sin_usar")),
        "Detector test sin caja anotada (falsos positivos)": feats(image_files(eval_dir / "det_test_fp")),
    }
    for name, f in real.items():
        lines.append(f"| {name} | {len(f)} | {_rate(knn, f)} | — | {_rate(final, f)} |")
    for name, f in domains.items():
        others = np.concatenate([g for d, g in domains.items() if d != name])
        loo = build_reference(bank, held, others, keep)
        lines.append(f"| {name} | {len(f)} | {_rate(knn, f)} | {_rate(loo, f)} | {_rate(final, f)} |")
    for name in UNSEEN_DOMAINS:
        f = feats(image_files(eval_dir / name))
        lines.append(f"| {name} (nunca en el ajuste) | {len(f)} | {_rate(knn, f)} | — | {_rate(final, f)} |")
    return lines


def main() -> int:
    """Build ood_crops.zip and ood_reference.npz for the base ml10 classifier."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / ARTIFACT_FOLDER_NAME)
    ap.add_argument("--per-domain", type=int, default=400, help="imágenes por dominio")
    ap.add_argument("--max-crops", type=int, default=400, help="recortes por dominio")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--work-dir", type=Path,
                    help="carpeta para los recortes intermedios; al relanzar se reutilizan "
                         "(por defecto, una temporal)")
    args = ap.parse_args()

    maker = CropMaker()
    with tempfile.TemporaryDirectory() as tmp:
        work = args.work_dir or Path(tmp)
        crops_dir, eval_dir = work / "crops", work / "eval"
        for split, folder in (("train", ID_TRAIN_DIR), ("val", ID_VAL_DIR)):
            n = _stage(crops_dir / folder / "deteccion",
                       lambda split=split: maker.detector_crops(split)[0])
            print(f"  detector {split}: {n} recortes", flush=True)
        # One pass over the photos gives every split; test/sin_usar are only for --evaluate.
        photo_dirs = {split: (crops_dir if split in ("train", "val") else eval_dir) / sub
                      for split, sub in PHOTO_SPLITS.items()}
        photos: dict[str, list[bytes]] = {}

        def photo_crops(split: str) -> list[bytes]:
            if not photos:
                photos.update(maker.photo_crops())
            return photos[split]

        for split, folder in photo_dirs.items():
            n = _stage(folder, lambda split=split: photo_crops(split))
            print(f"  fotos de clasificación {split}: {n} recortes", flush=True)
        for domain, src in OUTLIER_DOMAINS.items():
            rng = random.Random(domain)
            n = _stage(crops_dir / OUTLIERS_DIR / domain, lambda src=src, rng=rng: maker.domain_crops(
                _domain_images(src, args.per_domain, rng), args.max_crops, rng))
            print(f"  {domain}: {n} recortes", flush=True)

        zip_path = args.out / OOD_CROPS_FILENAME
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for path in sorted(crops_dir.rglob("*.jpg")):
                zf.write(path, path.relative_to(crops_dir).as_posix())
        print(f"{OOD_CROPS_FILENAME}: {zip_path.stat().st_size / 1e6:.1f} MB")

        reference = build_crops_reference(maker.classifier, maker.device,
                                          _images(CLS_SPLITS / "train"),
                                          _images(CLS_SPLITS / "val"), crops_dir)
        reference.save(str(args.out / OOD_REFERENCE_FILENAME))
        print(f"{OOD_REFERENCE_FILENAME}: método={reference.method} "
              f"umbral={reference.threshold:.4f} (conserva el {DEFAULT_KEEP:.0%} de validación)")

        if args.evaluate:
            if not (eval_dir / "det_test_fp" / ".done").exists():
                matched, unmatched = maker.detector_crops("test")
                _stage(eval_dir / "det_test_tp", lambda: matched)
                _stage(eval_dir / "det_test_fp", lambda: unmatched)
            for name, roots in UNSEEN_DOMAINS.items():
                rng = random.Random(name)
                _stage(eval_dir / name, lambda roots=roots, rng=rng: maker.domain_crops(
                    [im for root in roots for im in _domain_images(root, args.per_domain, rng)],
                    args.max_crops, rng))
            print("\n".join(evaluate(maker, crops_dir, eval_dir, DEFAULT_KEEP)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
