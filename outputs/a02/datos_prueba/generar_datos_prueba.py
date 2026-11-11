"""Genera los datos de prueba de ml2-fungal-cnn-disease-detection.

Fuentes (sin valores inventados):
- golden_cases de inbox/a02/manifest.yaml (18 imágenes del split de test real, salida esperada del
  checkpoint servido);
- inbox/a02/codigo/data/splits/train.csv para el ZIP de /train (primeras N imágenes de cada clase en
  el orden del fichero).

Uso: .venv/bin/python -I outputs/a02/datos_prueba/generar_datos_prueba.py "$PWD"
"""
import base64
import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml

N_TRAIN_PER_CLASS = 30


def main(repo: Path) -> None:
    out = repo / "outputs/a02/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    codigo = repo / "inbox/a02/codigo"
    cases = yaml.safe_load((repo / "inbox/a02/manifest.yaml").read_text(encoding="utf-8"))[
        "golden_cases"]
    classes = sorted(cases[0]["expected"]["probabilities"])

    rows = []
    with zipfile.ZipFile(out / "predict_batch_golden.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for c in cases:
            img = repo / c["input"]["image_path"]
            data = img.read_bytes()
            zf.writestr(img.name, data)
            body = {"mode": "inline", "image_base64": base64.b64encode(data).decode("ascii")}
            (out / f"predict_inline_{c['id']}.json").write_text(json.dumps(body) + "\n",
                                                                encoding="utf-8")
            exp = c["expected"]
            rows.append({"id": c["id"], "filename": img.name,
                         "ground_truth_label": c["ground_truth_label"],
                         "esperado_prediction": exp["prediction"],
                         "esperado_confidence": exp["confidence"],
                         **{f"esperado_prob_{k}": exp["probabilities"][k] for k in classes}})

    with open(out / "expected_golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    by_class = defaultdict(list)
    with open(codigo / "data/splits/train.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if len(by_class[r["label"]]) < N_TRAIN_PER_CLASS:
                by_class[r["label"]].append(r["path"])
    with zipfile.ZipFile(out / f"train_{N_TRAIN_PER_CLASS}_por_clase.zip", "w",
                         zipfile.ZIP_DEFLATED) as zf:
        for label in sorted(by_class):
            for rel in by_class[label]:
                zf.write(codigo / rel, f"{label}/{Path(rel).name}")

    print("OK", len(cases), "golden;", {k: len(v) for k, v in sorted(by_class.items())})


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
