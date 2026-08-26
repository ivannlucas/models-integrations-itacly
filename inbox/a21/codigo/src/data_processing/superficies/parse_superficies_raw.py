"""Limpia y parsea los CSVs de superficies provinciales con cabeceras multi-nivel.

Para cada CSV de entrada:
  1. Extrae el título (texto antes de ':') y lo usa como nombre del CSV limpio.
  2. Construye cabeceras multi-nivel con forward-fill jerárquico (dentro del span del padre).
  3. Elimina filas vacías y filas de separación.
  4. Conserva solo filas de datos (provincias / comunidades / ESPAÑA).
  5. Reemplaza '–' por vacío.

Entradas:
  - data/raw/superficies_provincial/*.csv
    (CSVs exportados directamente desde los ficheros Excel originales del MAPA,
     con cabeceras multi-nivel y filas de unidades)

Salidas:
  - data/processed/superficies_provincial_processed/<carpeta_año>/<nombre_cultivo>.csv
    (un CSV limpio por cultivo, con Provincia como primera columna y columnas
     de Superficie (ha) y Producción (t) bien nombradas)

Uso:
  python src/data_processing/superficies/parse_superficies_raw.py
"""

import os
import re
import csv
import sys
from pathlib import Path


def is_units_row(row):
    """Detecta si una fila es la fila de unidades (hectáreas, toneladas, kg/ha, %, etc.)."""
    non_empty = [c.strip() for c in row[1:] if c.strip()]
    if not non_empty:
        return False
    unit_count = 0
    for c in non_empty:
        if c.startswith("(") or re.match(r"^(kg/ha|%|t/ha)$", c, re.IGNORECASE):
            unit_count += 1
    return unit_count / len(non_empty) > 0.4


def build_column_names(header_rows, ncols):
    """Construye nombres de columna combinando niveles jerárquicos con forward-fill por spans."""

    skip_words = {
        "provincias", "y", "comunidades autónomas", "comunidades autonomas",
        "provincias y comunidades autónomas", "",
    }

    # Separar fila de unidades (última) de filas de categorías
    units_row = header_rows[-1]
    cat_rows = header_rows[:-1]

    if not cat_rows:
        # Solo hay fila de unidades
        names = ["Provincia"]
        for j in range(1, ncols):
            val = units_row[j].strip() if j < len(units_row) else ""
            names.append(val if val else f"Col_{j}")
        return names

    # --- Forward-fill jerárquico con clave compuesta ---
    filled = []

    # Nivel 0: forward-fill completo (categorías más amplias)
    level0 = cat_rows[0][:]
    last = ""
    for j in range(1, ncols):
        if j < len(level0) and level0[j].strip():
            last = level0[j].strip()
        if j < len(level0):
            level0[j] = last
    filled.append(level0)

    # Niveles siguientes: forward-fill dentro del span definido por la clave
    # compuesta de TODOS los niveles anteriores, no solo el padre inmediato.
    # Esto evita que el fill cruce fronteras cuando dos categorías distintas
    # comparten el mismo valor vacío en un nivel intermedio.
    for lvl in range(1, len(cat_rows)):
        row = cat_rows[lvl][:]

        current_key = None
        last_fill = ""
        for j in range(1, ncols):
            # Clave compuesta = tupla de valores en todos los niveles 0..lvl-1
            key = tuple(
                filled[l][j] if j < len(filled[l]) else ""
                for l in range(lvl)
            )
            if key != current_key:
                current_key = key
                last_fill = ""

            cell = row[j].strip() if j < len(row) else ""
            if cell:
                last_fill = cell
            row[j] = last_fill

        filled.append(row)

    # --- Construir nombres combinados ---
    names = ["Provincia"]
    for j in range(1, ncols):
        parts = []
        seen = set()
        for row in filled:
            val = row[j] if j < len(row) else ""
            if val and val.lower() not in skip_words and val not in seen:
                parts.append(val)
                seen.add(val)

        unit = units_row[j].strip() if j < len(units_row) else ""
        if unit and unit not in seen:
            parts.append(unit)

        names.append(" - ".join(parts) if parts else f"Col_{j}")

    return names


def clean_csv(filepath):
    """Limpia un CSV de superficies provinciales."""

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        raw_rows = list(reader)

    if not raw_rows:
        return None

    ncols = max(len(r) for r in raw_rows)
    for r in raw_rows:
        while len(r) < ncols:
            r.append("")

    # --- 1. Encontrar la fila del título ---
    title = None
    title_idx = None
    for i, row in enumerate(raw_rows):
        for cell in row:
            s = cell.strip()
            # Buscar patrón tipo "7.1.8.4. CEREALES..." con ':' o '*' o ','
            if re.match(r"^\d+\.\d+", s) and (":" in s or "," in s or "*" in s):
                title = s
                title_idx = i
                break
        if title:
            break

    if not title:
        print(f"  WARN: No se encontró título en {filepath}")
        return None

    # Extraer nombre corto: parte antes de ':', o antes de '*', o antes de ','
    if ":" in title:
        title_short = title.split(":")[0].strip()
    elif "*" in title:
        title_short = title.split("*")[0].strip()
    else:
        title_short = title.split(",")[0].strip()
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title_short)

    # --- 2. Recoger filas de header y encontrar inicio de datos ---
    header_rows = []
    data_start = None

    for i in range(title_idx + 1, len(raw_rows)):
        row = raw_rows[i]

        # Saltar filas vacías
        if all(c.strip() == "" for c in row):
            continue

        # Si es la fila de unidades, es la última fila de header
        if is_units_row(row):
            header_rows.append(row)
            # La siguiente fila no-vacía es datos
            for j in range(i + 1, len(raw_rows)):
                if any(c.strip() for c in raw_rows[j]):
                    data_start = j
                    break
            break

        header_rows.append(row)

    if not header_rows or data_start is None:
        print(f"  WARN: No se pudieron parsear las cabeceras en {filepath}")
        return None

    # --- 3. Construir nombres de columna ---
    col_names = build_column_names(header_rows, ncols)

    # --- 4. Extraer filas de datos ---
    data_rows = []
    for i in range(data_start, len(raw_rows)):
        row = raw_rows[i]

        if all(c.strip() == "" for c in row):
            continue

        first = row[0].strip()
        if not first:
            continue

        cleaned = [first]
        for j in range(1, ncols):
            val = row[j].strip()
            if val in ("–", "-", ""):
                val = "0"
            cleaned.append(val)

        data_rows.append(cleaned)

    return safe_title, col_names, data_rows


def process_file(filepath):
    """Procesa un CSV y genera el archivo limpio en la misma carpeta."""
    result = clean_csv(filepath)
    if result is None:
        print(f"  SKIP: {filepath}")
        return False

    safe_title, col_names, data_rows = result
    folder = os.path.dirname(filepath)
    out_path = os.path.join(folder, f"{safe_title}.csv")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        writer.writerows(data_rows)

    print(f"  OK: {os.path.basename(filepath)} -> {safe_title}.csv ({len(data_rows)} filas)")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for fpath in sys.argv[1:]:
            process_file(fpath)
    else:
        project_root = Path(__file__).resolve().parents[3]
        base = project_root / "data" / "processed" / "manual" / "superficies_provincial_processed"
        for folder in sorted(os.listdir(base)):
            folder_path = base / folder
            if not os.path.isdir(folder_path):
                continue
            print(f"\n{folder}:")
            for fname in sorted(os.listdir(folder_path)):
                if not fname.endswith(".csv"):
                    continue
                process_file(str(folder_path / fname))
