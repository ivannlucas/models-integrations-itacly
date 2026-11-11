"""Genera los datos de prueba de ml7-cereals-grain-pest-detection.

inbox/a07/codigo llega vacío. Fuentes (dataset real Pestdataset, sin valores inventados):
- imágenes: s3://<STORAGE_BUCKET>/datasets/modelo-7/raw/images/<ID>.jpg (las 18.069 del dataset);
- anotaciones: annos.zip (XML VOC) de la carpeta pública de Google Drive que enlaza la memoria
  (https://drive.google.com/drive/folders/1YykdhH_dwZkFsAh8EEgRVTAPk1clyGoy);
- golden_cases de inbox/a07/manifest.yaml (23 imágenes del split de test y su salida esperada).

La conversión VOC → YOLO replica preprocess.py::convert_xmls_to_yolo_labels con el orden de clases de
config/config.yaml::valid_classes [cf, sz, rd, tc, os] — el que aprendió el detector (ver
constants.TRAINING_CLASS_ID_TO_CODE del plugin). Como el ETL original, descarta las imágenes con algún
objeto fuera de esas clases (unknown, ct).

Uso (necesita las credenciales S3 del .env):
  .venv/bin/python -I outputs/a07/datos_prueba/generar_datos_prueba.py "$PWD"
"""
import base64
import csv
import io
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import boto3
import requests
import yaml

ANNOS_URL = ("https://drive.usercontent.google.com/download?id=1ishYW98NOugMY0Iq2C-XFnjO4du6MLeM"
             "&export=download&confirm=t")
VALID_CLASSES = ["cf", "sz", "rd", "tc", "os"]          # config/config.yaml::valid_classes
PER_CLASS = {"train": 8, "val": 2, "test": 2}           # imágenes por especie dominante


def _env(repo: Path) -> dict:
    env = dict(os.environ)
    for line in (repo / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if m and m.group(1) not in env:
            env[m.group(1)] = m.group(2).strip().strip('"')
    return env


def _yolo_lines(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in VALID_CLASSES:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(f"{VALID_CLASSES.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def _has_invalid_class(xml_bytes: bytes) -> bool:
    return any(obj.find("name").text not in VALID_CLASSES
               for obj in ET.fromstring(xml_bytes).findall("object"))


def main(repo: Path) -> None:
    out = repo / "outputs/a07/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    env = _env(repo)
    s3 = boto3.client("s3", endpoint_url=env["CUSTOM_S3_ENDPOINT"],
                      region_name=env.get("CUSTOM_REGION"),
                      aws_access_key_id=env.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY"))
    bucket = env["STORAGE_BUCKET"]

    def image(img_id: str) -> bytes:
        return s3.get_object(Bucket=bucket, Key=f"datasets/modelo-7/raw/images/{img_id}.jpg")[
            "Body"].read()

    manifest = yaml.safe_load((repo / "inbox/a07/manifest.yaml").read_text(encoding="utf-8"))
    cases = manifest["golden_cases"]["cases"]

    # ── golden ──
    rows = []
    with zipfile.ZipFile(out / "predict_batch_golden.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for c in cases:
            name = Path(c["input"]["image"]).name
            data = image(Path(name).stem)
            zf.writestr(name, data)
            body = {"mode": "inline", "image_base64": base64.b64encode(data).decode("ascii")}
            (out / f"predict_inline_{c['id']}.json").write_text(json.dumps(body) + "\n",
                                                                encoding="utf-8")
            exp = c["expected"]
            rows.append({"id": c["id"], "filename": name,
                         "ground_truth_species": c["ground_truth_species"],
                         "esperado_prediction": exp["prediction"],
                         "esperado_confidence": exp["confidence"],
                         "esperado_total_detections": exp["total_detections"],
                         "esperado_species_counts": json.dumps(exp["species_counts"]),
                         "nota": c.get("note", c.get("notes", ""))})
    with open(out / "expected_golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # ── train: dataset YOLO en ZIP (train_yolo/ dentro) ──
    resp = requests.get(ANNOS_URL, timeout=300)
    resp.raise_for_status()
    annos = zipfile.ZipFile(io.BytesIO(resp.content))
    golden_ids = {Path(c["input"]["image"]).stem for c in cases}
    labels = {}
    for member in annos.namelist():
        if member.endswith(".xml"):
            img_id = Path(member).stem
            xml = annos.read(member)
            # Como preprocess.py: relabel_invalid_classes + remove_images_with_class("unknown")
            # descartan la imagen entera si algún objeto no es de valid_classes, no solo esa caja.
            if img_id in golden_ids or _has_invalid_class(xml):
                continue
            lines = _yolo_lines(xml)
            if lines:
                labels[img_id] = lines

    # Verificación de la conversión: las especies de cada golden deben ser las reales
    # (ground_truth_species del manifest: "cf" o "multi (sz+tc)").
    for c in cases:
        img_id = Path(c["input"]["image"]).stem
        lines = _yolo_lines(annos.read(f"annos2/{img_id}.xml"))
        found = {VALID_CLASSES[int(line.split()[0])] for line in lines}
        gt = c["ground_truth_species"]
        m = re.match(r"multi \((.*)\)", gt)
        if found != (set(m.group(1).split("+")) if m else {gt}):
            raise RuntimeError(f"Conversión VOC→YOLO incoherente en {img_id}: {found} vs {gt}")

    shutil.rmtree(out / "train_yolo", ignore_errors=True)  # formato anterior (carpeta)
    used = {s: Counter() for s in PER_CLASS}
    with zipfile.ZipFile(out / "train_yolo.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for img_id in sorted(labels, key=int):
            dominant = VALID_CLASSES[Counter(int(line.split()[0]) for line in labels[img_id])
                                     .most_common(1)[0][0]]
            split = next((s for s, n in PER_CLASS.items() if used[s][dominant] < n), None)
            if split is None:
                continue
            used[split][dominant] += 1
            zf.writestr(f"train_yolo/images/{split}/{img_id}.jpg", image(img_id))
            zf.writestr(f"train_yolo/labels/{split}/{img_id}.txt", "\n".join(labels[img_id]) + "\n")
            if all(sum(used[s].values()) == n * len(VALID_CLASSES) for s, n in PER_CLASS.items()):
                break
        zf.writestr("train_yolo/dataset.yaml", yaml.safe_dump({
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": len(VALID_CLASSES), "names": VALID_CLASSES}, sort_keys=False))

    print("OK", len(cases), "golden; train_yolo.zip:", {s: sum(c.values()) for s, c in used.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
