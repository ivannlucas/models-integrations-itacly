"""Genera los datos de prueba de ml8_cereals_img_anomaly_detector.

inbox/a08/codigo llega vacío y las 19 imágenes de los golden_cases (split de test de 7 datasets de
Kaggle) no están disponibles: no se pueden reproducir aquí. La única copia accesible de imágenes del
dataset real es s3://<STORAGE_BUCKET>/datasets/modelo-8/{cereal}/train/{categoria}/ (50 por cada una
de las 12 combinaciones que existen), que pertenece al split de TRAIN del modelo servido.

- train: ZIP {cereal}/{train,validation}/{categoria}/ con las primeras 8 imágenes (orden de clave,
  saltando las de más de 400 KB) de cada combinación (6 train + 2 validation).
- predict: la 9.ª imagen de cada combinación, con su etiqueta real (la carpeta). NO son golden
  cases: no hay salida esperada auditada, y el modelo vio estas imágenes al entrenar.

Uso (necesita las credenciales S3 del .env):
  .venv/bin/python -I outputs/a08/datos_prueba/generar_datos_prueba.py "$PWD"
"""
import base64
import csv
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import boto3

PREFIX = "datasets/modelo-8/"
N_TRAIN, N_VAL = 6, 2
MAX_BYTES = 400_000  # se saltan las pocas imágenes muy grandes (hasta 10 MB) para que el ZIP sea pequeño


def _env(repo: Path) -> dict:
    env = dict(os.environ)
    for line in (repo / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if m and m.group(1) not in env:
            env[m.group(1)] = m.group(2).strip().strip('"')
    return env


def main(repo: Path) -> None:
    out = repo / "outputs/a08/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    env = _env(repo)
    s3 = boto3.client("s3", endpoint_url=env["CUSTOM_S3_ENDPOINT"],
                      region_name=env.get("CUSTOM_REGION"),
                      aws_access_key_id=env.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY"))
    bucket = env["STORAGE_BUCKET"]

    by_combo = defaultdict(list)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            parts = obj["Key"][len(PREFIX):].split("/")       # cereal/train/categoria/fichero
            if (len(parts) == 4 and parts[1] == "train" and "." in parts[3]
                    and obj["Size"] <= MAX_BYTES):
                by_combo[(parts[0], parts[2])].append(obj["Key"])

    def get(key: str) -> bytes:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    rows = []
    with zipfile.ZipFile(out / "train_8_por_combinacion.zip", "w", zipfile.ZIP_DEFLATED) as tz, \
            zipfile.ZipFile(out / "predict_batch_12.zip", "w", zipfile.ZIP_DEFLATED) as bz:
        for (cereal, categoria), keys in sorted(by_combo.items()):
            keys = sorted(keys)
            for i, key in enumerate(keys[: N_TRAIN + N_VAL]):
                split = "train" if i < N_TRAIN else "validation"
                tz.writestr(f"{cereal}/{split}/{categoria}/{Path(key).name}", get(key))
            key = keys[N_TRAIN + N_VAL]
            data, name = get(key), Path(key).name
            bz.writestr(name, data)
            case_id = f"{cereal}_{categoria}"
            body = {"mode": "inline", "image_base64": base64.b64encode(data).decode("ascii")}
            (out / f"predict_inline_{case_id}.json").write_text(json.dumps(body) + "\n",
                                                               encoding="utf-8")
            rows.append({"id": case_id, "filename": name, "ground_truth_cereal": cereal,
                         "ground_truth_categoria": categoria})

    with open(out / "etiquetas_reales.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("OK", len(rows), "combinaciones")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
