"""Genera los datos de prueba de ml3-wine-disease-pest-forecast.

No hay copia del dataset etiquetado: inbox/a03/codigo no se commitea, el "contenedor externo" que cita
su README no tiene enlace, y outputs/a03/datasets/train.csv (= s3://…/datasets/modelo-3/train.csv) son
solo las 149 series de test SIN etiquetas. Este script reconstruye el dataset original con el pipeline
del propio equipo de IA, recuperado del historial de git:

1. restaura inbox/a03/codigo desde el commit 29dc9a0^ (último en el que estaba rastreado);
2. scripts/data/descarga_clima_historico.py  -> clima horario de Open-Meteo (5 D.O., 2015-2025);
3. scripts/data/data_cleaner.py;
4. desplaza Fecha +1 h: el dataset original lleva las marcas 1 h por detrás de lo que Open-Meteo
   devuelve hoy con timezone=Europe/Madrid (comprobado: con +1 h coinciden las 249 406 filas de test);
5. scripts/data/vid_simulator.py (semilla 42) -> data/raw/data_vin_raw.parquet;
6. COMPRUEBA que las 149 series de test regeneradas son idénticas, fila a fila y columna a columna, a
   outputs/a03/datasets/train.csv. Si no lo son, aborta: los datos no serían los del equipo.

Salidas (outputs/a03/datos_prueba/); cada entrada de predicción va en .parquet (float32, el formato
original: es la referencia exacta) y en .csv (para subirla desde el front):
- predict_batch_golden.*: las 20 series de los golden_cases del manifest (últimas 168 h).
- expected_golden.csv: salida esperada (manifest) + etiqueta real de la última hora.
- predict_batch_series_cortas.* / expected_series_cortas.csv: 3 series reales de < 168 h (rama de
  padding); esperado = run_inference ORIGINAL del equipo con sus .keras entregados, leyendo el
  parquet y con la comparación de umbrales de NumPy 1.26 (el entorno del manifest; ver
  _numpy1_wetness). Con eso, el mismo procedimiento reproduce los 20 golden del manifest (clase
  exacta, floats a <= 5e-5: el original redondea a 4 decimales).
- train_88_series.csv: 8 series por clase, del split de train/val original (ninguna de test).

Uso: .venv/bin/python outputs/a03/datos_prueba/generar_datos_prueba.py "$PWD"
     (descarga ~6 MB de Open-Meteo la primera vez; ~4 min en total)
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_COMMIT = "29dc9a0^"
SENSORS = ["Temp_Amb_C", "Hum_Rel_Pct", "Lluvia_mm", "Viento_kmh", "CO2_ppm", "VOC_ppb",
           "Hum_Suelo_Pct", "pH_Suelo"]
INPUT_COLS = ["Fecha", "ID_Serie", "Parcela_ID"] + SENSORS
TRAIN_COLS = INPUT_COLS + ["Etiqueta_Clase", "Clase_Entrenamiento", "Grado_Infeccion"]
WINDOW = 168
N_TRAIN_PER_CLASS = 8
MIN_TRAIN_ROWS = 2 * WINDOW
SHORT_SERIES_CLASSES = ["BLACK_ROT", "MILDIU", "BOTRYTIS"]


def regenerate_raw(repo: Path) -> Path:
    """Rebuild data/raw/data_vin_raw.parquet with the AI team's own scripts (cached)."""
    code = repo / "inbox/a03/codigo"
    raw = code / "data/raw/data_vin_raw.parquet"
    if raw.exists():
        return raw
    if not (code / "scripts/data/vid_simulator.py").exists():
        archive = subprocess.run(["git", "-C", str(repo), "archive", CODE_COMMIT, "inbox/a03/codigo"],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(repo)], input=archive, check=True)
    py = sys.executable
    subprocess.run([py, "scripts/data/descarga_clima_historico.py"], cwd=code, check=True)
    subprocess.run([py, "scripts/data/data_cleaner.py"], cwd=code, check=True)
    for f in sorted((code / "data/clima_real/clean").glob("clima_*_clean.parquet")):
        df = pd.read_parquet(f)
        df["Fecha"] = pd.to_datetime(df["Fecha"]) + pd.Timedelta(hours=1)
        df.to_parquet(f, index=False)
    subprocess.run([py, "scripts/data/vid_simulator.py"], cwd=code, check=True)
    return raw


def check_against_test_split(raw: pd.DataFrame, repo: Path) -> set:
    """Abort unless the regenerated test series equal outputs/a03/datasets/train.csv exactly."""
    ref = pd.read_csv(repo / "outputs/a03/datasets/train.csv", parse_dates=["Fecha"])
    merged = ref.merge(raw, on=["ID_Serie", "Fecha"], how="left", suffixes=("", "_r"))
    problems = [c for c in SENSORS
                if not ((merged[c] - merged[f"{c}_r"]).abs() <= 1e-4 + 1e-5 * merged[c].abs()).all()]
    if not (merged["Parcela_ID"] == merged["Parcela_ID_r"]).all():
        problems.append("Parcela_ID")
    lengths = raw.groupby("ID_Serie").size()
    ref_lengths = ref.groupby("ID_Serie").size()
    if not (lengths.reindex(ref_lengths.index) == ref_lengths).all():
        problems.append("longitud de serie")
    if problems:
        raise SystemExit(f"El dataset regenerado NO coincide con el split de test original: {problems}")
    print(f"OK: {len(ref)} filas de {len(ref_lengths)} series de test idénticas al original")
    return set(ref_lengths.index)


def _numpy1_wetness(preprocess) -> None:
    """Make the original leaf-wetness counter compare like NumPy 1.26 (the manifest's env).

    `lluvia[t] > 0.1` with a float32 element: NumPy 1.x promotes to float64 (0.1f = 0.10000000149
    > 0.1 -> wet hour); NumPy 2 (NEP 50) compares in float32 (0.1f > 0.1f -> dry). Casting the
    float32 values to float64 first gives the NumPy 1.x result under any NumPy.
    """
    original = preprocess._calcular_horas_mojado_serie  # pylint: disable=protected-access

    def patched(lluvia, hr):
        return original(np.asarray(lluvia, dtype="float32").astype("float64"),
                        np.asarray(hr, dtype="float32").astype("float64"))

    preprocess._calcular_horas_mojado_serie = patched  # pylint: disable=protected-access


def original_inference(repo: Path, data_path: Path) -> pd.DataFrame:
    """Run the AI team's src/predict/predictor.py::run_inference (NumPy 1.26 semantics)."""
    code = repo / "inbox/a03/codigo"
    out_dir = repo / "inbox/a03/codigo/data/predictions/datos_prueba"
    config = yaml.safe_load((code / "config/config.yaml").read_text(encoding="utf-8"))
    config["cols_features"] = config.get("model_features", [])
    config["paths"]["artifacts_dir"] = str(code / config["paths"]["artifacts_dir"])
    config["paths"]["predictions_dir"] = str(out_dir)
    sys.path.insert(0, str(code))
    # pylint: disable-next=import-outside-toplevel
    from src.data_processing import preprocess
    from src.predict.predictor import run_inference  # pylint: disable=import-outside-toplevel
    _numpy1_wetness(preprocess)
    return pd.read_csv(run_inference(str(data_path), config), encoding="utf-8-sig")


def write_input(df: pd.DataFrame, stem: Path) -> Path:
    """Write a prediction input as .parquet (float32, exact) and .csv; return the parquet path."""
    df = df[INPUT_COLS]
    df.to_parquet(stem.with_suffix(".parquet"), index=False)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    return stem.with_suffix(".parquet")


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(repo: Path) -> None:
    out = repo / "outputs/a03/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(regenerate_raw(repo))
    raw["Fecha"] = pd.to_datetime(raw["Fecha"])
    raw = raw.sort_values(["ID_Serie", "Fecha"], kind="stable")
    test_ids = check_against_test_split(raw, repo)
    by_series = dict(tuple(raw.groupby("ID_Serie")))

    # ── golden: las 20 series del manifest, últimas 168 h ──
    cases = yaml.safe_load((repo / "inbox/a03/manifest.yaml").read_text(encoding="utf-8"))["golden_cases"]
    windows, rows = [], []
    for c in cases:
        serie = by_series[c["input"]["id_serie"]]
        end = pd.Timestamp(c["input"]["window_end_timestamp"])
        assert serie["Fecha"].iloc[-1] == end, c["id"]
        windows.append(serie.tail(WINDOW))
        last = serie.iloc[-1]
        rows.append({"id": c["id"], "id_serie": c["input"]["id_serie"], "fecha_evaluacion": str(end),
                     "clase_real_ultima_hora": last["Clase_Entrenamiento"],
                     "grado_infeccion_real": round(float(last["Grado_Infeccion"]), 6),
                     "esperado_diagnostico_ia": c["expected"]["diagnostico_ia"],
                     "esperado_confianza_clasificacion": c["expected"]["confianza_clasificacion"],
                     "esperado_grado_severidad": c["expected"]["grado_severidad"]})
    write_input(pd.concat(windows), out / "predict_batch_golden")
    write_csv(rows, out / "expected_golden.csv")

    # ── series cortas (< 168 h): rama de padding, esperado con el código original ──
    lengths = raw.groupby("ID_Serie").agg(n=("Fecha", "size"), cls=("Etiqueta_Clase", "first"))
    short_ids = [lengths[(lengths.cls == cls) & (lengths.n < WINDOW)].n.idxmin()
                 for cls in SHORT_SERIES_CLASSES]
    orig = original_inference(
        repo, write_input(raw[raw.ID_Serie.isin(short_ids)], out / "predict_batch_series_cortas"))
    write_csv([{"id_serie": int(r.ID_Serie), "n_filas": int(lengths.n[r.ID_Serie]),
                "clase_real_ultima_hora": by_series[r.ID_Serie]["Clase_Entrenamiento"].iloc[-1],
                "fecha_evaluacion_original": r.Fecha_Evaluacion,
                "esperado_diagnostico_ia": r.Diagnostico_IA,
                "esperado_confianza_clasificacion": round(float(r.Confianza_Clasificacion.rstrip("%")) / 100, 4),
                "esperado_grado_severidad": r.Grado_Severidad} for r in orig.itertuples()],
              out / "expected_series_cortas.csv")

    # ── train: 8 series por clase, fuera del split de test, de ≥ 336 h (las más cortas) ──
    pool = lengths[~lengths.index.isin(test_ids) & (lengths.n >= MIN_TRAIN_ROWS)]
    chosen = pool.reset_index().sort_values(["n", "ID_Serie"]).groupby("cls").head(N_TRAIN_PER_CLASS)
    train = raw[raw.ID_Serie.isin(chosen.ID_Serie)][TRAIN_COLS]
    train.to_csv(out / f"train_{len(chosen)}_series.csv", index=False)

    print(f"golden: {len(cases)} series | cortas: {short_ids} | train: {len(chosen)} series, "
          f"{len(train)} filas, {int((chosen.n - WINDOW).sum())} ventanas")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
