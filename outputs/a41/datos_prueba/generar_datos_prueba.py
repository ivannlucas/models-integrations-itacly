"""Genera los datos de prueba de ml41-meat-curing-machinery-acoustic-anomaly.

La entrega del equipo de IA (inbox/a41/codigo/data/raw) llega sin audio: el dataset MIMII es público
(Zenodo, record 3384388) y de él salen los 22 WAV de los golden_cases del manifest
("-6_dB_fan/id_00/{normal,abnormal}/000000NN.wav", ver golden_cases.audio_source_real). Este script
descarga solo los ficheros necesarios del ZIP de 10,8 GB mediante peticiones HTTP por rangos.

Los WAV de MIMII son de 8 canales (2,5 MB). Se guardan en mono float32 (640 KB): es exactamente la
señal que obtiene el plugin con librosa.load(mono=True), así que el espectrograma log-Mel es idéntico.

Uso: .venv/bin/python outputs/a41/datos_prueba/generar_datos_prueba.py "$PWD" [carpeta_cache_wav]
"""
import base64
import csv
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import librosa
import requests
import soundfile as sf
import yaml

ZENODO_URL = "https://zenodo.org/api/records/3384388/files/-6_dB_fan.zip/content"
SR = 16000
N_TRAIN_NORMAL = 40  # mínimo que admite fan: split 80/20 → 32 de train = batch_size (drop_last)
INLINE = {"caso_001": "anomalo", "caso_012": "normal", "caso_013": "falso_positivo"}


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
        data = resp.content
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _mono_wav_bytes(raw_wav: Path) -> bytes:
    y, _ = librosa.load(str(raw_wav), sr=SR, mono=True)
    out = io.BytesIO()
    sf.write(out, y, SR, subtype="FLOAT", format="WAV")
    return out.getvalue()


def _download(cache: Path, members: list[str]) -> None:
    missing = [m for m in members if not (cache / m).exists()]
    if not missing:
        return
    print(f"Descargando {len(missing)} WAV de Zenodo (MIMII -6_dB_fan.zip)...")
    with zipfile.ZipFile(io.BufferedReader(_HttpRangeFile(ZENODO_URL), buffer_size=1 << 20)) as zf:
        for member in missing:
            dest = cache / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))


def main(repo: Path, cache: Path) -> None:
    out = repo / "outputs/a41/datos_prueba"
    manifest = yaml.safe_load((repo / "inbox/a41/manifest.yaml").read_text(encoding="utf-8"))
    golden = manifest["golden_cases"]
    cases = golden["cases"]

    # sample_data/<clase>/<n>.wav  →  fan/id_00/<clase>/<n>.wav dentro de -6_dB_fan.zip
    golden_members = {c["id"]: "fan/id_00/" + c["input"]["audio_path"].split("sample_data/")[1]
                      for c in cases}
    train_normal = [f"fan/id_00/normal/{i:08d}.wav" for i in range(N_TRAIN_NORMAL)]
    train_abnormal = sorted(m for m in golden_members.values() if "/abnormal/" in m)
    _download(cache, sorted(set(golden_members.values()) | set(train_normal)))
    mono = {m: _mono_wav_bytes(cache / m)
            for m in set(golden_members.values()) | set(train_normal)}

    # ── predict batch: manifest.csv + 22 WAV, puntuados con el modelo fan/id_00/0_dB ──
    sel = cases[0]["input"]  # los 22 casos comparten machine/machine_id/snr
    rows = []
    with zipfile.ZipFile(out / "predict_batch_golden.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for c in cases:
            member = golden_members[c["id"]]
            filename = "/".join(member.split("/")[-2:])  # normal/00000000.wav
            zf.writestr(filename, mono[member])
            rows.append({"filename": filename, "machine": sel["machine"],
                         "machine_id": sel["machine_id"], "snr": sel["snr"]})
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["filename", "machine", "machine_id", "snr"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        zf.writestr("manifest.csv", buf.getvalue())

    # ── esperados ──
    with open(out / "expected_golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "filename", "machine", "machine_id", "snr", "true_label",
                         "esperado_maha_score", "esperado_predicted_label",
                         "esperado_threshold_used", "nota"])
        for c, row in zip(cases, rows):
            exp = c["expected"]
            writer.writerow([c["id"], row["filename"], row["machine"], row["machine_id"],
                             row["snr"], exp["true_label"], exp["maha_score"],
                             exp["predicted_label"], 6.6067, c.get("note", "")])

    # ── predict inline: 3 casos representativos en base64 ──
    by_id = {c["id"]: c for c in cases}
    for case_id, tag in INLINE.items():
        body = {"mode": "inline",
                "audio_base64": base64.b64encode(mono[golden_members[case_id]]).decode("ascii"),
                "machine": by_id[case_id]["input"]["machine"],
                "machine_id": by_id[case_id]["input"]["machine_id"],
                "snr": by_id[case_id]["input"]["snr"]}
        (out / f"predict_inline_{case_id}_{tag}.json").write_text(
            json.dumps(body, indent=2) + "\n", encoding="utf-8")

    # ── train: estructura {snr}_{machine}/{machine_id}/{normal,abnormal}/*.wav ──
    # El audio es de -6 dB, así que se etiqueta como tal: reentrena fan/id_00/-6_dB.
    with zipfile.ZipFile(out / "train_fan_id00_-6dB.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for member in train_normal + train_abnormal:
            zf.writestr("-6_dB_" + member, mono[member])

    print("OK:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    REPO = Path(sys.argv[1]).resolve()
    CACHE = (Path(sys.argv[2]) if len(sys.argv) > 2
             else Path(tempfile.gettempdir()) / "ml41_mimii_cache").resolve()
    main(REPO, CACHE)
