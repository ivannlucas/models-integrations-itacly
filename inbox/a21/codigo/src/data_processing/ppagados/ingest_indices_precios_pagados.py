#!/usr/bin/env python3
"""Ingiere un boletín MAPA "Índices y Precios Pagados Agrarios" (Excel) y genera
los CSV anchos (`_fixed.csv`) que consume `build_ppagados_unified.py`.

El MAPA publica este boletín periódicamente como un libro Excel con (al menos)
4 hojas: PrePag1, PrePag2, IndPag3, IndPag4 — en el mismo formato de bloques por
categoría (PRODUCTOS/AÑO/12 meses, con jerarquía de prefijos +, -, *) que ya usa
el resto del pipeline (ver `reshape_indpag4_timeseries.py`, cuya función de
parseo se reutiliza aquí sin duplicar lógica).

Permite a un auditor actualizar los datos MAPA sin más intervención manual que
descargar el Excel más reciente de la web del MAPA. Por defecto (sin
`--no-apply`), el propio script encadena todo lo necesario para dejar
`data/processed/auto/utils/` listo para `build_dataset.py`:

    python -m src.data_processing.ppagados.ingest_indices_precios_pagados \
        "ruta/al/boletin_mas_reciente.xlsx"

    # equivalente a encadenar, sin pasos manuales intermedios:
    #   1. parsear el Excel -> data/processed/manual/ppagados4/*_fixed.csv
    #   2. python -m src.data_processing.ppagados.build_ppagados_unified
    #   3. copiar data/processed/auto/ppagadostotal/*.csv a data/processed/auto/utils/

Tras ejecutar este script, el único paso manual restante es el propio
`build_dataset` (deliberadamente no encadenado, para no reentrenar/reconstruir
nada sin que el usuario lo pida explícitamente):

    python -m src.data_processing.build_dataset
    python -m src.data_processing.prepare_data --save-splits

Usar `--no-apply` para solo generar los CSV en data/processed/manual/ppagados4/
sin tocar `ppagadostotal/` ni `utils/` (comportamiento original, por si se
quiere revisar el resultado antes de fusionarlo).

Por defecto escribe en data/processed/manual/ppagados4/ — el tier de mayor
prioridad en `build_ppagados_unified.FOLDERS` (se solapa/gana sobre datos más
antiguos donde exista, y se completa con ellos donde no hay dato nuevo). Si se
usa un `--tier` distinto de "ppagados4", hay que añadirlo también a `FOLDERS`
en build_ppagados_unified.py para que se tenga en cuenta.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.data_processing.ppagados import build_ppagados_unified
from src.data_processing.ppagados.reshape_indpag4_timeseries import reshape_rows_to_wide

SHEET_NAMES = ["PrePag1", "PrePag2", "IndPag3", "IndPag4"]


def _sheet_to_rows(df: pd.DataFrame) -> list[list[str]]:
    """Convierte un DataFrame leído con header=None en filas de strings, tal
    como las devolvería csv.reader sobre el CSV equivalente (mismo formato que
    consume reshape_rows_to_wide)."""
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float) and v == int(v):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        rows.append(cells)
    return rows


def ingest(xlsx_path: Path, tier: str = "ppagados4") -> list[Path]:
    out_dir = Path("data/processed/manual") / tier
    out_dir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(xlsx_path)
    written: list[Path] = []
    for sheet in SHEET_NAMES:
        matching = [s for s in xl.sheet_names if s.strip().lower() == sheet.lower()]
        if not matching:
            print(f"AVISO: hoja '{sheet}' no encontrada en {xlsx_path.name}, se omite.")
            continue
        raw = xl.parse(matching[0], header=None)
        rows = _sheet_to_rows(raw)
        wide = reshape_rows_to_wide(rows)
        if wide.empty:
            print(f"AVISO: no se pudo parsear la hoja '{sheet}' (formato inesperado).")
            continue
        out_path = out_dir / f"{tier}__{sheet}_fixed.csv"
        wide.to_csv(out_path, date_format="%Y-%m-%d", float_format="%.2f")
        written.append(out_path)
        print(f"WROTE {out_path} - rows={len(wide)} cols={len(wide.columns)}")
    return written


def apply_and_stage() -> list[Path]:
    """Ejecuta build_ppagados_unified y copia los totales resultantes a
    data/processed/auto/utils/, dejando el panel listo para build_dataset.py
    sin ningun paso manual intermedio."""
    build_ppagados_unified.main()

    src_dir = Path("data/processed/auto/ppagadostotal")
    dst_dir = Path("data/processed/auto/utils")
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for csv_path in sorted(src_dir.glob("*.csv")):
        dst = dst_dir / csv_path.name
        shutil.copy2(csv_path, dst)
        copied.append(dst)
        print(f"COPIED {csv_path} -> {dst}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingiere un boletin MAPA 'Indices y Precios Pagados Agrarios' (xlsx)."
    )
    parser.add_argument("xlsx", help="Ruta al libro Excel descargado del MAPA.")
    parser.add_argument(
        "--tier",
        default="ppagados4",
        help="Carpeta destino en data/processed/manual/ (por defecto: ppagados4, "
        "la de mayor prioridad en build_ppagados_unified.FOLDERS).",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Solo genera los CSV en data/processed/manual/<tier>/, sin fusionarlos "
        "en ppagadostotal/ ni copiarlos a utils/ (comportamiento original).",
    )
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        print(f"ERROR: no existe el fichero {xlsx_path}")
        return 1

    written = ingest(xlsx_path, tier=args.tier)
    if not written:
        print("ERROR: no se genero ningun fichero. Revisa el nombre de las hojas del Excel.")
        return 1

    print(f"\nOK: {len(written)} ficheros escritos en data/processed/manual/{args.tier}/.")

    if args.no_apply:
        print(
            "Siguiente paso: python -m src.data_processing.ppagados.build_ppagados_unified"
        )
        return 0

    print("\n=== Fusionando y actualizando data/processed/auto/utils/ ===")
    copied = apply_and_stage()
    if not copied:
        print("ERROR: build_ppagados_unified no genero ficheros en ppagadostotal/.")
        return 1

    print(
        f"\nOK: {len(copied)} ficheros actualizados en data/processed/auto/utils/. "
        "El panel esta listo para: python -m src.data_processing.build_dataset"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
