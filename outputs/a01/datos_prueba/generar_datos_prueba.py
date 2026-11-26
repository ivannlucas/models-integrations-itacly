"""Genera los datos de prueba de ml1-wine-pest-disease-detection.

Fuentes (sin valores inventados):
- golden_cases de inbox/a01/manifest.yaml: 20 muestras reales (imagen + 9 sensores) del split de
  test del equipo de IA, con la salida esperada del modelo servido (predicho_*) y la etiqueta y el
  riesgo reales. La imagen de cada caso se localiza en inbox/a01/codigo/data/splits/test_split.csv
  por nombre de fichero + clase + sensores (hay nombres repetidos entre carpetas de clase) y se toma
  de data/processed/IMAGENES_RESIZED224/, la misma con la que se midieron los predicho_*;
- data/processed/to_predict/input.csv + images/ y data/predictions/predictions.csv: la entrada de
  ejemplo del equipo (3 filas, 2 de ellas sin imagen) y su salida, para el modo sin imagen;
- golden_cases.rechazo_sin_imagen_ni_rf del manifest: el caso de contrato que debe dar 422;
- data/splits/train_rf.csv para el CSV de /train (primeras N filas de cada clase, en el orden del
  fichero y con sus columnas originales).

Uso: .venv/bin/python -I outputs/a01/datos_prueba/generar_datos_prueba.py "$PWD"
"""
import base64
import csv
import json
import sys
import zipfile
from pathlib import Path, PureWindowsPath

import pandas as pd
import yaml

N_TRAIN_PER_CLASS = 1000
SENSORS = ["Mes", "Temp_Max_Media_5d_C", "Temp_Min_Media_5d_C", "Hum_Rel_Max_5d_Pct",
           "Hum_Rel_Min_5d_Pct", "Hum_Suelo_Pct", "Lluvia_Acum_5d_mm", "Horas_Humedad_Foliar",
           "GDD_Acumulado"]
GASES = ["CO2_ppm", "VOC_ppb"]  # el plugin los acepta y los ignora (exclude_gases=true)
FILA_SIN_IMAGEN_INLINE = 2  # fila de input.csv con Path_Imagen=N/A


def _localizar_golden(caso: dict, test: pd.DataFrame, codigo: Path) -> tuple[Path, pd.Series]:
    """Fila de test_split.csv del golden case e imagen 224 correspondiente (falla si no es única)."""
    cand = test[(test.Nombre_Archivo == caso["imagen"]) & (test.Etiqueta == caso["esperado_clase"])]
    cand = cand[[all(abs(float(r[s]) - float(caso["sensores"][s])) < 1e-9 for s in SENSORS)
                 for _, r in cand.iterrows()]]
    if len(cand) != 1:
        raise SystemExit(f"{caso['id']}: {len(cand)} filas en test_split.csv (se esperaba 1)")
    row = cand.iloc[0]
    if abs(row.Indice_Riesgo_Cientifico - caso["esperado_riesgo"]) > 1e-9:
        raise SystemExit(f"{caso['id']}: riesgo real distinto del de test_split.csv")
    carpeta = PureWindowsPath(row.Path_Imagen).parent.name
    img = codigo / "data/processed/IMAGENES_RESIZED224" / carpeta / caso["imagen"]
    if not img.is_file():
        raise SystemExit(f"{caso['id']}: no existe {img}")
    return img, row


def _escribir_json(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body, ensure_ascii=False) + "\n", encoding="utf-8")


def main(repo: Path) -> None:  # pylint: disable=too-many-locals
    out = repo / "outputs/a01/datos_prueba"
    out.mkdir(parents=True, exist_ok=True)
    codigo = repo / "inbox/a01/codigo"
    golden = yaml.safe_load((repo / "inbox/a01/manifest.yaml").read_text(encoding="utf-8"))[
        "golden_cases"]
    test = pd.read_csv(codigo / "data/splits/test_split.csv")

    filas_batch, imagenes_batch, esperados = [], {}, []

    # 1) Golden cases: imagen + sensores -> HIBRIDO_ENSAMBLE
    for caso in golden["cases"]:
        img, row = _localizar_golden(caso, test, codigo)
        data = img.read_bytes()
        sensores = {s: caso["sensores"][s] for s in SENSORS}
        nombre = f"predict_inline_{caso['id']}_{caso['esperado_clase'].lower()}.json"
        _escribir_json(out / nombre, {"mode": "inline",
                                      "image_base64": base64.b64encode(data).decode("ascii"),
                                      "sensors": sensores})
        filas_batch.append({"Path_Imagen": img.name, **sensores,
                            **{g: row[g] for g in GASES}})
        imagenes_batch[img.name] = data
        esperados.append({
            "id": caso["id"], "fichero_inline": nombre, "origen": "golden_case_manifest",
            "imagen": img.name, "etiqueta_real": caso["esperado_clase"],
            "riesgo_real": caso["esperado_riesgo"],
            "esperado_enfermedad": caso["predicho_clase"],
            "esperado_confianza": caso["predicho_confianza"],
            "esperado_riesgo": caso["predicho_riesgo"], "esperado_metodo": "HIBRIDO_ENSAMBLE",
            "acierto_clase_modelo": caso["predicho_clase"] == caso["esperado_clase"],
        })

    # 2) Ejemplo del equipo (input.csv -> predictions.csv): 1 fila con imagen, 2 sin ella
    #    (fila 1: la imagen no viene en la entrega -> contingencia tabular; fila 2: N/A).
    to_predict = codigo / "data/processed/to_predict"
    # keep_default_na=False: "N/A" se conserva como texto, igual que lo lee el plugin del ZIP
    entrada = pd.read_csv(to_predict / "input.csv", keep_default_na=False)
    salida = pd.read_csv(codigo / "data/predictions/predictions.csv")
    for i, row in enumerate(entrada.to_dict("records")):
        row = pd.Series(row, dtype=object)  # escalares nativos de Python (JSON serializable)
        pred = salida.iloc[i]
        sensores = {s: row[s] for s in SENSORS}
        img = to_predict / "images" / str(row.Path_Imagen)
        if img.is_file():
            imagenes_batch[img.name] = img.read_bytes()
        nombre = ""
        if i == FILA_SIN_IMAGEN_INLINE:
            nombre = f"predict_inline_sin_imagen_input_fila_{i}.json"
            _escribir_json(out / nombre, {"mode": "inline", "sensors": sensores})
        filas_batch.append({"Path_Imagen": row.Path_Imagen, **sensores,
                            **{g: row[g] for g in GASES}})
        esperados.append({
            "id": f"equipo_fila_{i}", "fichero_inline": nombre,
            "origen": "data/predictions/predictions.csv",
            "imagen": (row.Path_Imagen if img.is_file() or row.Path_Imagen == "N/A"
                       else f"{row.Path_Imagen} (no viene en la entrega)"),
            "etiqueta_real": "", "riesgo_real": "",
            "esperado_enfermedad": pred.Prediccion,
            "esperado_confianza": round(float(str(pred.Confianza).rstrip("%")) / 100, 4),
            "esperado_riesgo": pred.Riesgo, "esperado_metodo": pred.Metodo,
            "acierto_clase_modelo": "",
        })

    # 3) Rechazo de contrato (manifest): faltan 8 de los 9 sensores -> 422
    rechazo = golden["rechazo_sin_imagen_ni_rf"][0]
    _escribir_json(out / "predict_inline_rechazo_sensores_incompletos.json",
                   {"mode": "inline", **rechazo["input"]})

    # Batch: CSV con el contrato de input.csv (Path_Imagen + 9 sensores + CO2/VOC) e images/
    # Fecha fija en cada entrada: el ZIP sale idéntico byte a byte en cada regeneración
    def _info(nombre: str) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(nombre, date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        return info

    with zipfile.ZipFile(out / "predict_batch_golden.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_info("entrada.csv"), pd.DataFrame(filas_batch).to_csv(index=False))
        for nombre, data in imagenes_batch.items():
            zf.writestr(_info(f"images/{nombre}"), data)

    with open(out / "expected_golden.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(esperados[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(esperados)  # mismo orden que las filas de entrada.csv

    # Train: primeras N filas de cada clase de train_rf.csv, en el orden del fichero
    train = pd.read_csv(codigo / "data/splits/train_rf.csv")
    sub = train[train.groupby("Etiqueta").cumcount() < N_TRAIN_PER_CLASS]
    sub.to_csv(out / f"train_{N_TRAIN_PER_CLASS}_por_clase.csv", index=False)

    print("OK", len(golden["cases"]), "golden +", len(entrada), "filas del equipo;",
          len(imagenes_batch), "imagenes en el ZIP; train", len(sub), "filas,",
          sub.Etiqueta.nunique(), "clases")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
