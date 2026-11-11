"""Genera los datos de prueba de ml10-dairy-disease-vector-detection.

Fuentes (dataset real entregado, sin valores inventados):
- golden_cases de inbox/a10/manifest.yaml (25 imágenes: 18 crops del split de test de clasificación,
  6 imágenes completas sin recortar y data/test.jpg) con su salida esperada;
- inbox/a10/codigo/.../data/classification/splits/train/{fly,mos,tick}/ para el ZIP de /train
  (primeras N imágenes de cada especie por nombre de fichero).

Uso: .venv/bin/python -I outputs/a10/datos_prueba/generar_datos_prueba.py "$PWD"
"""
import base64
import csv
import json
import sys
import zipfile
from pathlib import Path

import yaml

ROOT_REL = "inbox/a10/codigo/a10-del-dairy-deteccion-vectores-enfermedades-ganado-vacuno"
N_TRAIN_PER_CLASS = 30
SPECIES = ("fly", "mos", "tick")


def main(repo: Path) -> None:
    out = repo / "outputs/a10/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    root = repo / ROOT_REL
    cases = yaml.safe_load((repo / "inbox/a10/manifest.yaml").read_text(encoding="utf-8"))[
        "golden_cases"]

    rows = []
    with zipfile.ZipFile(out / "predict_batch_golden.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for c in cases:
            rel = c["input"]["image_path"]                 # relativa al repo o a la carpeta del código
            img = repo / rel if rel.startswith("inbox/") else root / rel
            data = img.read_bytes()
            name = f"{c['id']}{img.suffix.lower()}"     # nombre único (hay crops homónimos)
            zf.writestr(name, data)
            body = {"mode": "inline", "image_base64": base64.b64encode(data).decode("ascii"),
                    "det_conf_thresh": c["input"]["det_conf_thresh"],
                    "cls_conf_thresh": c["input"]["cls_conf_thresh"]}
            (out / f"predict_inline_{c['id']}.json").write_text(json.dumps(body) + "\n",
                                                                encoding="utf-8")
            exp = c["expected"]
            rows.append({"id": c["id"], "filename": name,
                         "imagen_original": str(img.relative_to(root)),
                         "esperado_prediction": exp["prediction"],
                         "esperado_confidence": exp["confidence"],
                         "esperado_vectors_count": exp["vectors_count"],
                         "esperado_species_summary": json.dumps(exp["species_summary"]),
                         "esperado_detections": json.dumps(exp["detections"]),
                         "tolerance_abs": c.get("tolerance_abs", "")})
    with open(out / "expected_golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    train_dir = root / "data/classification/splits/train"
    with zipfile.ZipFile(out / f"train_{N_TRAIN_PER_CLASS}_por_especie.zip", "w",
                         zipfile.ZIP_DEFLATED) as zf:
        for sp in SPECIES:
            for img in sorted((train_dir / sp).iterdir())[:N_TRAIN_PER_CLASS]:
                zf.write(img, f"{sp}/{img.name}")

    print("OK", len(cases), "golden")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
