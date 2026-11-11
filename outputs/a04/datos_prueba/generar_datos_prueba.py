"""Genera los datos de prueba de ml4-lactic-cnn-thermal-early-disease-detection.

inbox/a04/codigo llega vacío. Las imágenes salen del dataset público TIDS (Zenodo), versión 2
(record 18854578; su "TIDS Dataset.zip" es idéntico byte a byte al de la versión 1, la que usó el
equipo de IA). NO se usa la versión 3: cambia la imagen del animal 11, que es un golden case. Se
descargan solo los ficheros necesarios del ZIP mediante peticiones HTTP por rangos.

- golden: los 18 golden_cases del manifest (recortes TIDS_cropped del split de test oficial);
  `real_served_output` es la salida esperada del checkpoint servido.
- train: ID_Labels.csv + TIDS_cropped/{healthy,SCM}/<ID>.jpg con los N primeros IDs de cada clase
  (orden numérico) que no son golden cases.

Uso: .venv/bin/python -I outputs/a04/datos_prueba/generar_datos_prueba.py "$PWD"
"""
import base64
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import requests
import yaml

TIDS_V2_URL = "https://zenodo.org/api/records/18854578/files/TIDS%20Dataset.zip/content"
N_TRAIN_PER_CLASS = 25
CLASS_DIR = {0: "healthy", 1: "SCM"}


class _HttpRangeFile(io.RawIOBase):
    """Fichero remoto de solo lectura servido con cabeceras Range (para zipfile)."""

    def __init__(self, url: str):
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.url, self.size, self.pos = head.url, int(head.headers["Content-Length"]), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        resp = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        resp.raise_for_status()
        if resp.status_code != 206:
            raise RuntimeError(f"El servidor no admite Range (HTTP {resp.status_code})")
        buffer[: len(resp.content)] = resp.content
        self.pos += len(resp.content)
        return len(resp.content)


def main(repo: Path) -> None:
    out = repo / "outputs/a04/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in yaml.safe_load((repo / "inbox/a04/manifest.yaml").read_text(
        encoding="utf-8"))["golden_cases"] if c["id"].startswith("tids_test_")]

    remote = zipfile.ZipFile(io.BufferedReader(_HttpRangeFile(TIDS_V2_URL), buffer_size=1 << 20))
    labels_csv = remote.read("TIDS Dataset/ID_Labels.csv").decode("utf-8")
    label_rows = list(csv.DictReader(io.StringIO(labels_csv)))

    # ── golden: inline + batch ──
    rows = []
    with zipfile.ZipFile(out / "predict_batch_golden.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for c in cases:
            member = c["input"]["image_path"]                  # TIDS Dataset/TIDS_cropped/<cls>/<ID>.jpg
            data = remote.read(member)
            cls, name = member.split("/")[-2:]
            zf.writestr(f"{cls}/{name}", data)
            body = {"mode": "inline", "image_base64": base64.b64encode(data).decode("ascii")}
            (out / f"predict_inline_{c['id']}.json").write_text(json.dumps(body) + "\n",
                                                                encoding="utf-8")
            served = c["real_served_output"]
            rows.append({"id": c["id"], "filename": name, "ground_truth_label": c["expected"]["label"],
                         "esperado_prediction": served["prediction"],
                         "esperado_confidence": served["confidence"]})
    with open(out / "expected_golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # ── train: mismo layout que TIDS ──
    golden_ids = {Path(c["input"]["image_path"]).stem for c in cases}
    chosen = []
    for label, cls in CLASS_DIR.items():
        ids = sorted((r for r in label_rows if int(r["label"]) == label and r["ID"] not in golden_ids),
                     key=lambda r: int(r["ID"]))
        chosen += ids[:N_TRAIN_PER_CLASS]
    with zipfile.ZipFile(out / "train_tids_50.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(label_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(chosen, key=lambda r: int(r["ID"])))
        zf.writestr("ID_Labels.csv", buf.getvalue())
        for r in chosen:
            rel = f"TIDS_cropped/{CLASS_DIR[int(r['label'])]}/{r['ID']}.jpg"
            zf.writestr(rel, remote.read(f"TIDS Dataset/{rel}"))

    print("OK", len(cases), "golden;", len(chosen), "imágenes de train")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
