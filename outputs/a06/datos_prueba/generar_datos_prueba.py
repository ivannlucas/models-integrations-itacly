"""Genera los datos de prueba de ml6-meat-poultry-disease-classification (dos ensembles).

inbox/a06/codigo/ llega vacío (los datos no venían en la entrega). Las imágenes salen de las mismas
fuentes públicas de las que salieron los golden_cases (manifest, memoria §3.1):

- thermal: Mendeley vwbb4dc9b7 v1, Healthy.rar y Sick.rar (se comprueba el sha256 publicado por la
  API de Mendeley). Si existe inbox/a06/codigo/data/thermal/raw/{healthy,sick} se usa esa copia.
  Para descomprimir el RAR hace falta 7zz/7z/unrar/unar/bsdtar en el PATH o, en WSL, el 7z.exe de
  Windows (C:\\Program Files\\7-Zip). Ojo: tar.exe de Windows se salta los 8 nombres en árabe.
- rgb: Zenodo 5801834 v3 (pcrhealthy/pcrcocci/pcrncd/pcrsalmo.zip, 4,4 GB). Solo se descargan los
  ficheros necesarios mediante peticiones HTTP por rangos.

El split se recalcula con el algoritmo de src/utils/io.py::ensure_data_splits (seed 42, 70/15/15) y
se comprueba que su test coincide con los golden_cases del manifest antes de escribir nada.

- predict (golden): thermal → los 40 golden (= split de test completo) en el batch y las 5 primeras
  de cada clase en inline; rgb → por clase, el golden de test de menor tamaño de fichero, más el
  empate a tres de menor tamaño (inline y batch). La regla no mira si el modelo acierta.
- train: los N primeros ficheros (orden de nombre) del split de TRAIN de cada clase. thermal va con
  los BMP/JPG originales. rgb va en PNG 256x256: el plugin aplica Resize((256, 256)) a cada imagen;
  aquí se aplica ese mismo Resize y se guarda sin pérdida, así que el tensor que recibe la red es
  idéntico (se comprueba) y el ZIP pasa de ~120 MB a unos pocos MB.

Uso: .venv/bin/python -I outputs/a06/datos_prueba/generar_datos_prueba.py "$PWD"
"""
import base64
import csv
import hashlib
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import requests
import yaml
from PIL import Image
from torchvision import transforms

N_TRAIN_PER_CLASS = 10
N_INLINE_THERMAL_PER_CLASS = 5
SEED, TRAIN_P, VAL_P = 42, 0.7, 0.15
RGB_RESIZE = (256, 256)                      # constants.RGB_RESIZE del plugin

MENDELEY = "https://data.mendeley.com/public-files/datasets/vwbb4dc9b7/files/{}/file_downloaded"
THERMAL_RARS = {  # clase -> (id de fichero en Mendeley, sha256 publicado)
    "healthy": ("3f7ed00c-dc4d-4c88-9793-009a297d62b6",
                "ba0c3a8777b074bf46ded1572e2e1373e93f3203866bfdabd66b662a0ea69678"),
    "sick": ("ab3756fd-530e-4713-b57c-c547a530e097",
             "de8464840fbebc90d12a06c2d60845219dfa0002f09411ef3e7e6277a031ac2f"),
}
THERMAL_COUNTS = {"healthy": 86, "sick": 168}
ZENODO = "https://zenodo.org/api/records/5801834/files/{}.zip/content"
RGB_FOLDERS = {"pcrhealthy": "healthy", "pcrcocci": "cocci", "pcrncd": "ncd", "pcrsalmo": "salmo"}
RGB_TEST_COUNTS = {"pcrhealthy": 53, "pcrcocci": 57, "pcrncd": 29, "pcrsalmo": 53}
MODELS = {"thermal": ("densenet121", "efficientnet_v2s", "resnet50"),
          "rgb": ("vgg19_bn", "efficientnet_b7", "mobilenet_v3l")}
WIN_7Z = Path("/mnt/c/Program Files/7-Zip/7z.exe")


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


def ensure_data_splits(names_by_folder: dict) -> dict:
    """src/utils/io.py::ensure_data_splits: carpetas y ficheros por nombre, un único
    random.Random(42) que baraja cada carpeta por turno, int(n*0.7) train, int(n*0.15) val."""
    rng, out = random.Random(SEED), {}
    for folder in sorted(names_by_folder):
        files = sorted(names_by_folder[folder])
        rng.shuffle(files)
        n_train, n_val = int(len(files) * TRAIN_P), int(len(files) * VAL_P)
        out[folder] = {"train": files[:n_train], "val": files[n_train:n_train + n_val],
                       "test": files[n_train + n_val:]}
    return out


def _extract_rar(rar: Path, dest: Path) -> None:
    for tool, args in (("7zz", ["x", "-y", f"-o{dest}", str(rar)]),
                       ("7z", ["x", "-y", f"-o{dest}", str(rar)]),
                       ("unrar", ["x", "-y", str(rar), f"{dest}/"]),
                       ("unar", ["-f", "-o", str(dest), str(rar)]),
                       ("bsdtar", ["-xf", str(rar), "-C", str(dest)])):
        if shutil.which(tool):
            subprocess.run([tool, *args], check=True, capture_output=True)
            return
    if WIN_7Z.exists():  # WSL: 7-Zip de Windows con rutas de Windows
        win = [subprocess.run(["wslpath", "-w", str(p)], check=True, capture_output=True,
                              text=True).stdout.strip() for p in (dest, rar)]
        subprocess.run([str(WIN_7Z), "x", "-y", f"-o{win[0]}", win[1]], check=True,
                       capture_output=True)
        return
    raise RuntimeError("No hay ninguna herramienta para descomprimir RAR (7zz, 7z, unrar, unar, "
                       "bsdtar o el 7z.exe de Windows en WSL)")


def _thermal_raw(repo: Path, tmp: Path) -> Path:
    """Carpeta con healthy/ y sick/ del dataset térmico completo."""
    local = repo / "inbox/a06/codigo/data/thermal/raw"
    if (local / "healthy").is_dir() and (local / "sick").is_dir():
        return local
    raw = tmp / "thermal_raw"
    for cls, (file_id, sha) in THERMAL_RARS.items():
        resp = requests.get(MENDELEY.format(file_id), timeout=300)
        resp.raise_for_status()
        if hashlib.sha256(resp.content).hexdigest() != sha:
            raise RuntimeError(f"sha256 de {cls}.rar distinto del publicado en Mendeley")
        rar = tmp / f"{cls}.rar"
        rar.write_bytes(resp.content)
        out = tmp / f"x_{cls}"
        _extract_rar(rar, out)
        (folder,) = [d for d in out.iterdir() if d.is_dir() and d.name.lower() == cls]
        raw.mkdir(exist_ok=True)
        folder.rename(raw / cls)
    return raw


def _write_inline(out: Path, case: dict, data: bytes) -> str:
    name = f"predict_inline_{case['id']}_{case['true_class']}.json"
    body = {"mode": "inline", "model_key": case["input"]["model_key"],
            "image_base64": base64.b64encode(data).decode("ascii")}
    (out / name).write_text(json.dumps(body) + "\n", encoding="utf-8")
    return name


def _expected_row(case: dict, folder: str, inline_name: str) -> dict:
    exp, task = case["expected"], case["input"]["model_key"]
    return {"id": case["id"], "filename": Path(case["input"]["image_path"]).name,
            "ruta_en_zip": f"{folder}/{Path(case['input']['image_path']).name}",
            "json_inline": inline_name, "true_class": case["true_class"],
            "esperado_prediction": exp["prediction"],
            **{f"esperado_{m}": exp["model_predictions"][m] for m in MODELS[task]}}


def _write_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _thermal(repo: Path, out: Path, cases: list, tmp: Path) -> str:
    raw = _thermal_raw(repo, tmp)
    names = {c: [p.name for p in (raw / c).iterdir() if p.is_file()] for c in THERMAL_COUNTS}
    counts = {c: len(v) for c, v in names.items()}
    if counts != THERMAL_COUNTS:
        raise RuntimeError(f"Dataset térmico incompleto: {counts}")
    split = ensure_data_splits(names)
    golden = {(Path(c["input"]["image_path"]).parent.name, Path(c["input"]["image_path"]).name)
              for c in cases}
    test = {(c, n) for c in split for n in split[c]["test"]}
    if golden != test:
        raise RuntimeError("El split de test térmico no coincide con los golden_cases del manifest")

    rows, per_class = [], {c: 0 for c in THERMAL_COUNTS}
    with zipfile.ZipFile(out / "predict_batch_thermal.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for c in cases:
            folder, name = Path(c["input"]["image_path"]).parent.name, Path(c["input"]["image_path"]).name
            data = (raw / folder / name).read_bytes()
            zf.writestr(f"{folder}/{name}", data)
            inline = ""
            if per_class[folder] < N_INLINE_THERMAL_PER_CLASS:
                per_class[folder] += 1
                inline = _write_inline(out, c, data)
            rows.append(_expected_row(c, folder, inline))
    _write_csv(out / "expected_golden_thermal.csv", rows)

    train_name = f"train_thermal_{N_TRAIN_PER_CLASS}_por_clase.zip"
    with zipfile.ZipFile(out / train_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls in sorted(split):
            for name in sorted(split[cls]["train"])[:N_TRAIN_PER_CLASS]:
                zf.write(raw / cls / name, f"{cls}/{name}")
    return f"thermal: {len(rows)} golden en batch, {sum(per_class.values())} inline"


def _resized_png(data: bytes) -> bytes:
    """PNG sin pérdida de la imagen tras el Resize((256, 256)) del plugin, comprobando que el
    plugin, al volver a aplicar ese Resize al PNG, obtiene exactamente los mismos píxeles."""
    resized = transforms.Resize(RGB_RESIZE)(Image.open(io.BytesIO(data)).convert("RGB"))
    buf = io.BytesIO()
    resized.save(buf, format="PNG", optimize=True)
    again = transforms.Resize(RGB_RESIZE)(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    if not np.array_equal(np.asarray(resized), np.asarray(again)):
        raise RuntimeError("El PNG 256x256 no reproduce la entrada del modelo")
    return buf.getvalue()


def _rgb(out: Path, cases: list) -> str:
    remote = {f: zipfile.ZipFile(io.BufferedReader(_HttpRangeFile(ZENODO.format(f)),
                                                   buffer_size=1 << 20)) for f in RGB_FOLDERS}
    sizes, names = {}, {}
    for folder, zf in remote.items():
        infos = [i for i in zf.infolist() if not i.is_dir()]
        names[folder] = [Path(i.filename).name for i in infos]
        sizes.update({(folder, Path(i.filename).name): i.file_size for i in infos})
    split = ensure_data_splits(names)
    if {f: len(split[f]["test"]) for f in split} != RGB_TEST_COUNTS:
        raise RuntimeError("El split de test rgb no tiene los tamaños de los CSV entregados")

    def key(c):
        p = Path(c["input"]["image_path"])
        return p.parent.name, p.name

    for c in cases:
        if key(c)[1] not in split[key(c)[0]]["test"]:
            raise RuntimeError(f"{c['id']} no está en el split de test regenerado")
    chosen = [min((c for c in cases if c["id"].startswith("rgb_test_") and key(c)[0] == f),
                  key=lambda c: (sizes[key(c)], c["id"])) for f in sorted(RGB_FOLDERS)]
    chosen.append(min((c for c in cases if c["id"].startswith("rgb_tie_")),
                      key=lambda c: (sizes[key(c)], c["id"])))

    rows = []
    with zipfile.ZipFile(out / "predict_batch_rgb.zip", "w", zipfile.ZIP_STORED) as zf:
        for c in chosen:
            folder, name = key(c)
            data = remote[folder].read(f"{folder}/{name}")
            zf.writestr(f"{folder}/{name}", data)
            rows.append(_expected_row(c, folder, _write_inline(out, c, data)))
    _write_csv(out / "expected_golden_rgb.csv", rows)

    train_name = f"train_rgb_{N_TRAIN_PER_CLASS}_por_clase.zip"
    with zipfile.ZipFile(out / train_name, "w", zipfile.ZIP_STORED) as zf:
        for folder in sorted(split):
            for name in sorted(split[folder]["train"])[:N_TRAIN_PER_CLASS]:
                png = _resized_png(remote[folder].read(f"{folder}/{name}"))
                zf.writestr(f"{folder}/{Path(name).stem}.png", png)
    return f"rgb: {len(rows)} golden (inline y batch)"


def main(repo: Path) -> None:
    out = repo / "outputs/a06/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("predict_inline_*.json"):
        old.unlink()
    cases = yaml.safe_load((repo / "inbox/a06/manifest.yaml").read_text(encoding="utf-8"))[
        "golden_cases"]
    with tempfile.TemporaryDirectory(prefix="a06_datos_") as tmp:
        msg_t = _thermal(repo, out, [c for c in cases if c["input"]["model_key"] == "thermal"],
                         Path(tmp))
    msg_r = _rgb(out, [c for c in cases if c["input"]["model_key"] == "rgb"])
    print("OK", msg_t, "|", msg_r, f"| train {N_TRAIN_PER_CLASS} por clase")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
