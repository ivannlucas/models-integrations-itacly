"""Compara el batch de ml5 sobre predict_batch_video.mp4 con batch_video_anotaciones.csv.

Batch no tiene golden: cada fila de la salida (frame, track del plugin, caja xyxy) se empareja con la
vaca anotada en ese frame de mayor IoU (≥ 0,5). Hasta que un track acumula 32 frames, el plugin
devuelve el relleno del original (grazing, confianza 1,0), así que solo se puntúa a partir de que el
track tiene su primera predicción real: el frame 31 si se detecta desde el principio.

Uso (necesita artifacts/ml5_meat_cow_behaviour/):
  .venv/bin/python -I outputs/a05/datos_prueba/comprobar_batch_video.py "$PWD"
"""
import ast
import csv
import sys
from collections import defaultdict
from pathlib import Path

CLIP_LENGTH = 32
MIN_IOU = 0.5


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _annotations(path: Path) -> dict[int, list[tuple[str, str, list[float]]]]:
    by_frame = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            x, y, w, h = ast.literal_eval(row["bbox_xywh"])
            by_frame[int(row["frame"])].append(
                (row["track_id_anotado"], row["behavior_anotado"], [x, y, x + w, y + h]))
    return by_frame


def main(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    from app.plugins.ml5_meat_cow_behaviour.plugin import (  # pylint: disable=import-outside-toplevel
        Ml5MeatCowBehaviourPlugin,
    )

    here = repo / "outputs/a05/datos_prueba"
    gt = _annotations(here / "batch_video_anotaciones.csv")
    plugin = Ml5MeatCowBehaviourPlugin()
    plugin.load()
    frames = plugin.predict_batch(data_path=str(here / "predict_batch_video.mp4")).predictions

    seen = defaultdict(int)             # frames acumulados por track del plugin
    score = defaultdict(lambda: [0, 0])  # track anotado → [aciertos, frames puntuados]
    preds = defaultdict(lambda: defaultdict(int))
    behavior_of = {}
    last = {}                           # track anotado → (predicción, confianza) en su último frame
    relleno = sin_pareja = 0
    for fr in frames:
        for det in fr["detections"]:
            seen[det["track_id"]] += 1
            if seen[det["track_id"]] < CLIP_LENGTH:
                relleno += 1
                continue
            best = max(gt[fr["frame"]], key=lambda a, d=det: _iou(a[2], d["bbox"]), default=None)
            if best is None or _iou(best[2], det["bbox"]) < MIN_IOU:
                sin_pareja += 1
                continue
            track, behavior, _ = best
            behavior_of[track] = behavior
            score[track][1] += 1
            score[track][0] += det["behavior"] == behavior
            preds[track][det["behavior"]] += 1
            last[track] = (det["behavior"], det["behavior_confidence"])

    print(f"{len(frames)} frames; filas de relleno (track con < {CLIP_LENGTH} frames): {relleno}; "
          f"filas sin vaca anotada (IoU < {MIN_IOU}): {sin_pareja}")
    print(f"{'track':>5}  {'anotado':<20} {'aciertos':>9}  {'último frame':<26} predicho")
    total = [0, 0]
    for track in sorted(score, key=int):
        ok, n = score[track]
        total[0] += ok
        total[1] += n
        final = f"{last[track][0]} {last[track][1]:.2f}"
        print(f"{track:>5}  {behavior_of[track]:<20} {ok:>4}/{n:<4}  {final:<26} {dict(preds[track])}")
    print(f"total: {total[0]}/{total[1]} ({100 * total[0] / total[1]:.1f} %); "
          f"tracks anotados sin puntuar: {sorted(set(map(str, (t for f in gt.values() for t, _, _ in f))) - set(score), key=int)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
