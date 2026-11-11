"""Genera los datos de prueba de ml5-meat-cow-behaviour.

Fuentes (dataset real entregado, sin valores inventados):
- inbox/a05/golden_cases_real.json: 25 clips (clip_name, track_id, 32 frames con su bbox COCO) del
  split de test. El ROI se recorta como utils/data_utils.py::BehaviorClassificationDataset
  (bbox entero, recortado a la imagen, resize 224×224; relleno duplicando el último frame si faltan).
- inbox/a05/codigo/data/images/annotations/<clip>/annotations/instances_default.json (export CVAT por
  clip) + inbox/a05/codigo/data/{train,val,test}/<clip>_img_NNNNN.jpg (frames 1920×1080).

Uso: .venv/bin/python -I outputs/a05/datos_prueba/generar_datos_prueba.py "$PWD"
El batch se compara con su anotación con comprobar_batch_video.py.
"""
import base64
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2

CLIP_LENGTH = 32
N_FRAMES_TRAIN = 32      # frames por clip en el ZIP de train (los primeros del clip)
N_FRAMES_VIDEO = 64      # frames del vídeo de batch
VIDEO_FPS = 30
KNOWN_FAILURE_CLIP = "1349_"  # resting-standing predicho como hidden (verification_report.md)
# Clip del vídeo de batch: split de test, fuera de golden y del ZIP de train; sus 8 vacas están en los
# 64 frames con un comportamiento constante, 5 distintos y solo 1 grazing (la clase de relleno de los
# primeros 31 frames, así que un clip de casi todo grazing no distingue relleno de predicción).
BATCH_CLIP = "1333_arm01_gopro4_20200323_035752_beh4_ani2_ins1_cut_00058"


def _frame_path(data: Path, flat_name: str) -> Path:
    for split in ("test", "val", "train"):
        p = data / split / flat_name
        if p.exists():
            return p
    raise FileNotFoundError(flat_name)


def _roi(img, bbox):
    """Recorte de data_utils.py::BehaviorClassificationDataset.__getitem__ (img en BGR)."""
    img_h, img_w = img.shape[:2]
    x, y, w, h = (int(v) for v in bbox)
    x, y = max(0, min(x, img_w - 1)), max(0, min(y, img_h - 1))
    x2, y2 = max(0, min(x + w, img_w)), max(0, min(y + h, img_h))
    return cv2.resize(img[y:y2, x:x2], (224, 224))


def _cvat(data: Path, clip: str) -> dict:
    return json.loads((data / "images/annotations" / clip / "annotations/instances_default.json")
                      .read_text(encoding="utf-8"))


def golden(repo: Path, out: Path, data: Path) -> None:
    cases = json.loads((repo / "inbox/a05/golden_cases_real.json").read_text(encoding="utf-8"))
    rows = []
    for c in cases:
        frames = c["frames"]
        frames = json.loads(frames.replace("'", '"')) if isinstance(frames, str) else frames
        rois = [_roi(cv2.imread(str(_frame_path(data, f["file_name"]))), f["bbox"]) for f in frames]
        while len(rois) < CLIP_LENGTH:                     # relleno del original
            rois.append(rois[-1])
        frames_b64 = [base64.b64encode(cv2.imencode(".jpg", r)[1].tobytes()).decode("ascii")
                      for r in rois]
        body = {"mode": "inline", "frames_base64": frames_b64}
        (out / f"predict_inline_{c['id']}.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
        rows.append({"id": c["id"], "clip_name": c["clip_name"], "track_id": c["track_id"],
                     "n_real_frames": c["n_real_frames"], "padded_to_32": c["padded_to_32"],
                     "esperado_prediction": c["behavior_ground_truth"],
                     "nota": ("fallo documentado del modelo: lo predice como hidden "
                              "(verification_report.md)") if c["clip_name"].startswith(
                                  KNOWN_FAILURE_CLIP) else ""})
    with open(out / "expected_golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return cases


def batch_video(out: Path, data: Path, clip: str) -> None:
    cvat = _cvat(data, clip)
    images = sorted(cvat["images"], key=lambda i: i["id"])[:N_FRAMES_VIDEO]
    writer = None
    by_image = {}
    for a in cvat["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    rows = []
    for idx, img in enumerate(images):
        name = Path(img["file_name"]).name
        frame = cv2.imread(str(_frame_path(data, f"{clip}_{name}")))
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out / "predict_batch_video.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
        writer.write(frame)
        for a in by_image.get(img["id"], []):
            rows.append({"frame": idx, "track_id_anotado": a["attributes"]["track_id"],
                         "behavior_anotado": a["attributes"]["behavior"],
                         "bbox_xywh": [round(v, 2) for v in a["bbox"]]})
    writer.release()
    with open(out / "batch_video_anotaciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def train_zip(out: Path, data: Path, excluded: set[str]) -> list[str]:
    clips = sorted(p.name for p in (data / "images/annotations").iterdir() if p.is_dir())
    chosen, seen = [], set()
    for clip in clips:                                     # primer clip de cada código behN
        code = re.search(r"_beh(\d+)_", clip).group(1)
        if code not in seen and clip not in excluded:
            seen.add(code)
            chosen.append(clip)
    with zipfile.ZipFile(out / "train_cvat_9clips.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for clip in chosen:
            cvat = _cvat(data, clip)
            images = sorted(cvat["images"], key=lambda i: i["id"])[:N_FRAMES_TRAIN]
            keep = {i["id"] for i in images}
            cvat["images"] = images
            cvat["annotations"] = [a for a in cvat["annotations"] if a["image_id"] in keep]
            zf.writestr(f"annotations/{clip}/annotations/instances_default.json", json.dumps(cvat))
            for img in images:
                name = Path(img["file_name"]).name
                zf.write(_frame_path(data, f"{clip}_{name}"), f"raw_frames/{clip}/{name}")
    return chosen


def main(repo: Path) -> None:
    out = repo / "outputs/a05/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    data = repo / "inbox/a05/codigo/data"
    cases = golden(repo, out, data)
    batch_video(out, data, BATCH_CLIP)
    chosen = train_zip(out, data, {c["clip_name"] for c in cases} | {BATCH_CLIP})
    print("OK", len(cases), "golden; clips de train:", chosen)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
