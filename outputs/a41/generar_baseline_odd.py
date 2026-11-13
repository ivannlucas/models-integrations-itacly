"""Genera el dataset del baseline ODD (detección de drift) de modelo-41 (ml41, MIMII).

El servicio retech-lote2-xai-odd-detection registra modelo-41 con "type": "audio",
"s3_dataset_prefix": "datasets/modelo-41/" y "extractor": "logmel". Al arrancar, si no existe
baselines/modelo-41/baseline.json, descarga los audios de ese prefijo, calcula el baseline y lo
sube. Este script prepara esos audios.

Qué baja: grabaciones NORMALES del dataset público MIMII (Zenodo, record 3384388), las mismas
de las que salen los checkpoints de ml41. Nunca clips "abnormal": el baseline describe la
distribución de entrada normal. Por defecto 20 clips por cada una de las 48 combinaciones
(4 máquinas x 4 unidades x 3 SNR) = 960 clips, repartidos uniformemente a lo largo de cada
carpeta normal (no los 20 primeros) para cubrir más variedad.

Solo se descargan esos ficheros de los ZIP de Zenodo (decenas de GB en total) mediante
peticiones HTTP por rangos. Los WAV de MIMII son de 8 canales (2,5 MB); se guardan en mono
float32 a 16 kHz (640 KB): la misma señal que obtienen el plugin y el extractor "logmel" del
ODD con librosa.load(sr=16000, mono=True).

Salida (por defecto artifacts/odd_baseline_modelo-41/, ignorado por git):
    datasets/modelo-41/<snr>/<machine>/<machine_id>/normal/<n>.wav
    README.md  (procedencia y comando de subida)
Subida:  aws s3 cp --recursive <salida>/datasets/modelo-41/ s3://<S3_BASELINE_BUCKET>/datasets/modelo-41/

Es reanudable: los ficheros ya generados se saltan. Reintenta ante 429/5xx de Zenodo.

Uso:
    .venv/bin/python outputs/a41/generar_baseline_odd.py [--out DIR] [--per-combo N]
        [--only=SNR_MACHINE ...] [--workers N]
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf

ZENODO_URL = "https://zenodo.org/api/records/3384388/files/{snr}_{machine}.zip/content"
# Mismos valores que app/plugins/ml41_meat_curing_machinery_acoustic_anomaly/constants.py.
MACHINES = ("fan", "pump", "slider", "valve")
MACHINE_IDS = ("id_00", "id_02", "id_04", "id_06")
SNRS = ("-6_dB", "0_dB", "6_dB")
SR = 16000
DATASET_PREFIX = "datasets/modelo-41"
REPO = Path(__file__).resolve().parents[2]


class _HttpRangeFile(io.RawIOBase):
    """Fichero remoto de solo lectura servido con cabeceras Range (para zipfile)."""

    def __init__(self, url: str):
        self.session = requests.Session()
        head = self._request("head", url, allow_redirects=True)
        self.url, self.size, self.pos = head.url, int(head.headers["Content-Length"]), 0

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(8):
            try:
                resp = self.session.request(method, url, timeout=120, **kwargs)
            except requests.RequestException:
                resp = None
            if resp is not None and resp.status_code < 400:
                return resp
            if resp is not None and resp.status_code not in (429, 500, 502, 503, 504):
                resp.raise_for_status()
            wait = int(resp.headers.get("Retry-After", 0)) if resp is not None else 0
            time.sleep(max(wait, 2 ** attempt))
        raise RuntimeError(f"Zenodo no responde tras varios reintentos: {url}")

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
        resp = self._request("get", self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        if resp.status_code != 206:
            raise RuntimeError(f"El servidor no admite Range (HTTP {resp.status_code})")
        data = resp.content
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def spread(items: list[str], n: int) -> list[str]:
    """n elementos repartidos uniformemente a lo largo de items (ordenado), sin repetir."""
    if n >= len(items):
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def mono_wav_bytes(raw: bytes) -> bytes:
    """Mono float32 a 16 kHz: la señal que ven el plugin y el extractor logmel del ODD."""
    y, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
    out = io.BytesIO()
    sf.write(out, y, SR, subtype="FLOAT", format="WAV")
    return out.getvalue()


def process_zip(snr: str, machine: str, per_combo: int, dataset_dir: Path) -> tuple[str, int, int]:
    """Descarga los clips normales elegidos de un ZIP {snr}_{machine}; devuelve (zip, nuevos, total)."""
    name = f"{snr}_{machine}"
    reader = io.BufferedReader(_HttpRangeFile(ZENODO_URL.format(snr=snr, machine=machine)),
                               buffer_size=4 << 20)  # 4 MB: un clip (2,5 MB) por petición
    created = total = 0
    with zipfile.ZipFile(reader) as zf:
        names = zf.namelist()
        for machine_id in MACHINE_IDS:
            folder = f"{machine}/{machine_id}/normal/"
            members = sorted(n for n in names if n.startswith(folder) and n.endswith(".wav"))
            if not members:
                raise RuntimeError(f"{name}.zip no tiene {folder}")
            for member in spread(members, per_combo):
                total += 1
                dest = dataset_dir / snr / member
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(mono_wav_bytes(zf.read(member)))
                tmp.rename(dest)
                created += 1
    return name, created, total


def write_readme(out: Path, per_combo: int) -> None:
    (out / "README.md").write_text(
        f"""# Baseline ODD de modelo-41 (ml41, anomalías acústicas)

Generado con `outputs/a41/generar_baseline_odd.py` (repo models-integrations-itacly).

- Origen: MIMII (Zenodo 3384388), solo grabaciones **normales**.
- {per_combo} clips por combinación x 48 combinaciones (máquinas {", ".join(MACHINES)};
  unidades {", ".join(MACHINE_IDS)}; SNR {", ".join(SNRS)}), repartidos uniformemente en cada carpeta.
- Mono float32 a {SR} Hz (misma señal que `librosa.load(sr={SR}, mono=True)`).

## Subida

```
aws s3 cp --recursive datasets/modelo-41/ s3://<S3_BASELINE_BUCKET>/{DATASET_PREFIX}/
```

Después, reiniciar el servicio ODD: al arrancar genera `baselines/modelo-41/baseline.json`.
Comprobar antes que `S3_BASELINE_BUCKET` (bootstrap) y `OOD_BASELINE_BUCKET` (/detect)
apuntan al mismo bucket.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "odd_baseline_modelo-41")
    parser.add_argument("--per-combo", type=int, default=20)
    parser.add_argument("--only", action="append",
                        help="ZIP concreto, repetible y con '=': --only=-6_dB_fan (por defecto los 12)")
    parser.add_argument("--workers", type=int, default=4, help="ZIP descargados en paralelo")
    args = parser.parse_args()

    zips = [(snr, m) for snr in SNRS for m in MACHINES
            if not args.only or f"{snr}_{m}" in args.only]
    dataset_dir = args.out / DATASET_PREFIX
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_readme(args.out, args.per_combo)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_zip, snr, m, args.per_combo, dataset_dir): f"{snr}_{m}"
                   for snr, m in zips}
        for fut in as_completed(futures):
            name, created, total = fut.result()
            print(f"{name}: {created} nuevos, {total} en total ({time.perf_counter() - started:.0f} s)",
                  flush=True)

    wavs = list(dataset_dir.rglob("*.wav"))
    size_mb = sum(p.stat().st_size for p in wavs) / 1e6
    print(f"OK: {len(wavs)} WAV, {size_mb:.0f} MB en {dataset_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
