#!/usr/bin/env python3
"""Transforma IndPag4.csv de formato de bloques por categoría a serie temporal mensual.

El archivo de entrada tiene un formato irregular: la primera columna contiene el nombre
de categoría una vez y las filas siguientes tienen la primera celda vacía con el año y
12 valores mensuales. El script reconstruye la jerarquía de etiquetas (prefijos +, -, *)
y genera un CSV ancho con una fila por mes y una columna por categoría.

Entradas:
  - data/processed/manual/ppagados1_2/IndPag4.csv
    (formato de bloques con jerarquía de categorías, años como filas y 12 meses como columnas)

Salidas:
  - data/processed/manual/ppagados1_2/IndPag4_fixed.csv
    (columnas: date, <categoría1>, <categoría2>, ...; una fila por mes)

Uso:
  python src/data_processing/ppagados/reshape_indpag4_timeseries.py
  python src/data_processing/ppagados/reshape_indpag4_timeseries.py OtroArchivo.csv

La función `reshape_rows_to_wide` es reutilizada por
`ingest_indices_precios_pagados.py` para procesar directamente hojas de Excel
del MAPA con el mismo formato de bloques, sin duplicar la lógica de parseo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import csv
import re
import argparse
import pandas as pd


def clean_label(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # remove leading dashes or bullets
    s = re.sub(r"^[\-\+–—\*·\s]+", "", s)
    return s


def reshape_rows_to_wide(rows: Iterable[Sequence[str]]) -> pd.DataFrame:
    """Convierte filas en formato de bloques MAPA (PRODUCTOS/AÑO/12 meses, con
    jerarquía de prefijos +, -, *) en un DataFrame ancho: índice 'date' (mensual)
    y una columna por categoría de producto.

    Reutilizada tanto por el CLI original (CSV ya extraído de una hoja) como por
    ingest_indices_precios_pagados.py (hoja de Excel leída directamente), para no
    duplicar la lógica de parseo del formato irregular del MAPA.
    """
    data: dict[str, dict[str, float]] = {}
    categories: list[str] = []
    current_cat: str | None = None
    level0: str | None = None
    level1: str | None = None

    for row in rows:
        row = [str(c) if c is not None else "" for c in row]
        if not row:
            continue
        first_cell = row[0].strip() if len(row) > 0 else ""
        if first_cell.lower() in ("productos", "producto", "año", "ano"):
            continue

        if first_cell != "":
            raw_label = first_cell
            stripped = raw_label.strip()
            if re.match(r"^[\+\*]\s*", stripped):
                sub = re.sub(r"^[\+\*\s]+", "", stripped)
                if level1:
                    current_cat = f"{level1} {sub}"
                elif level0:
                    current_cat = f"{level0} {sub}"
                else:
                    current_cat = clean_label(sub)
            elif re.match(r"^[\-–—\s]+", raw_label):
                sub = re.sub(r"^[\-–—\s]+", "", raw_label).strip()
                if level0:
                    level1 = f"{level0} {clean_label(sub)}"
                    current_cat = level1
                else:
                    level1 = clean_label(sub)
                    current_cat = level1
            else:
                level0 = clean_label(raw_label)
                level1 = None
                current_cat = level0

            if current_cat not in categories:
                categories.append(current_cat)
            year_idx = 1
        else:
            year_idx = 1

        if current_cat is None:
            continue

        if len(row) <= year_idx:
            continue
        year_cell = row[year_idx].strip()
        if year_cell == "":
            continue
        m = re.search(r"(\d{4})", year_cell)
        if not m:
            continue
        year = int(m.group(1))

        vals = []
        for cell in row[year_idx + 1 :]:
            if len(vals) >= 12:
                break
            vals.append(cell.strip())
        while len(vals) < 12:
            vals.append("")

        for mth, raw in enumerate(vals, start=1):
            date = f"{year}-{mth:02d}-01"
            v = raw.replace(",", ".") if isinstance(raw, str) else raw
            try:
                num = float(v) if v != "" else float("nan")
            except Exception:
                num = float("nan")
            data.setdefault(date, {})[current_cat] = num

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data, orient="index")
    for c in categories:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[categories]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # replace NaN with 0 where upstream used 0 for missing (keeps existing zeros)
    df = df.fillna(0)
    df.index.name = "date"
    return df


def main():
    parser = argparse.ArgumentParser(description="Reshape grouped CSV into monthly wide CSV")
    parser.add_argument("input", nargs="?", default="IndPag4.csv", help="input filename in data/processed/manual/ppagados1_2")
    args = parser.parse_args()

    BASE = Path("data/processed/manual/ppagados1_2")
    infile = BASE / args.input
    stem = Path(args.input).stem
    outfile = BASE / f"{stem}_fixed.csv"

    if not infile.exists():
        print(
            f"Input not found: {infile}\n"
            "Este paso requiere datos manuales que no se distribuyen en el repositorio "
            "(ver README, seccion 'Advertencia sobre datos manuales'). Es seguro omitir "
            "este paso: la salida ya generada esta disponible en data/processed/auto/utils/."
        )
        return

    with infile.open("r", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))

    df = reshape_rows_to_wide(rows)
    if df.empty:
        print("No data parsed from file — check input format.")
        return

    df.to_csv(outfile, date_format="%Y-%m-%d", float_format="%.2f")
    print(f"WROTE {outfile} — rows={len(df)} cols={len(df.columns)}")


if __name__ == "__main__":
    main()
