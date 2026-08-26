"""Unifica los CSVs de superficies limpios por año en un único total_YYYY.csv.

Para cada carpeta de año en superficies_provincial_processed:
  1. Detecta el año de la carpeta (nombre tipo AE02-... → año 2001).
  2. Lee todos los CSVs limpios generados por parse_superficies_raw.py.
  3. Normaliza nombres de cultivo al canónico (Trigo duro, Trigo blando, Cebada 2/6, Maíz híbrido, Otros maíces).
  4. Genera un CSV unificado por año con Provincia + pares Superficie/Producción.

Entradas:
  - data/processed/superficies_provincial_processed/<carpeta_año>/<cultivo>.csv
    (generados por parse_superficies_raw.py)

Salidas:
  - data/processed/superficies_provincial_processed/<carpeta_año>/total_YYYY.csv
    (una fila por provincia con columnas: Provincia, <Cultivo> - Superficie (ha),
     <Cultivo> - Producción (t) para cada cultivo canónico)

Uso:
  python src/data_processing/superficies/unify_superficies_by_year.py
"""

import os
import re
import csv
from pathlib import Path


# Orden canónico de columnas de salida
CROP_COLUMNS = [
    "Trigo duro",
    "Trigo blando y semiduro",
    "Cebada 2 carreras",
    "Cebada 6 carreras",
    "Maíz híbrido",
    "Otros maíces",
]

# Mapeo de variantes de nombre → nombre canónico
CROP_NAME_MAP = {
    "trigo duro": "Trigo duro",
    "trigo blando y semiduro": "Trigo blando y semiduro",
    "cebada de 2 carreras": "Cebada 2 carreras",
    "cebada 2 carreras": "Cebada 2 carreras",
    "cebada de 6 carreras": "Cebada 6 carreras",
    "cebada 6 carreras": "Cebada 6 carreras",
    "maíz híbrido": "Maíz híbrido",
    "maíz hibrido": "Maíz híbrido",
    "maíz híbirido": "Maíz híbrido",  # typo en 2002-2004
    "maiz hibrido": "Maíz híbrido",
    "maíz grano": "Maíz híbrido",  # 2024 uses this
    "otros maíces": "Otros maíces",
    "otros maices": "Otros maíces",
}


def normalize_crop_name(raw_name):
    """Normaliza un nombre de cultivo a su forma canónica."""
    key = raw_name.strip().lower()
    if key in CROP_NAME_MAP:
        return CROP_NAME_MAP[key]
    # Intento parcial
    for pattern, canon in CROP_NAME_MAP.items():
        if pattern in key or key in pattern:
            return canon
    return raw_name.strip()


def get_year(folder):
    """Extrae el año de la carpeta."""
    m = re.match(r"AE(\d{2})-", folder)
    if m:
        return 2000 + int(m.group(1))
    m = re.match(r"AE_(\d{4})_", folder)
    if m:
        return int(m.group(1))
    return None


def extract_crop_data_multi(filepath):
    """Extrae datos de un CSV con formato multi-cultivo (5 cols).
    Returns dict: {crop_name: {provincia: (superficie, produccion)}}
    """
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # header: Provincia, Crop1 - Superficie - ..., Crop1 - Producción - ..., Crop2 - Sup, Crop2 - Prod
    crops = {}
    # Parse pairs of columns (1,2), (3,4), etc.
    for i in range(1, len(header), 2):
        if i + 1 >= len(header):
            break
        # Extract crop name from column header
        raw = header[i].split(" - ")[0].strip()
        crop = normalize_crop_name(raw)
        if crop not in crops:
            crops[crop] = {}
        for row in rows:
            prov = row[0].strip()
            sup = row[i] if i < len(row) else "0"
            prod = row[i + 1] if i + 1 < len(row) else "0"
            crops[crop][prov] = (sup, prod)

    return crops


def extract_crop_data_single(filepath, crop_name):
    """Extrae datos de un CSV con formato simple (3 cols: Provincia, Sup, Prod).
    Returns dict: {crop_name: {provincia: (superficie, produccion)}}
    """
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    crop = normalize_crop_name(crop_name)
    data = {}
    for row in rows:
        prov = row[0].strip()
        sup = row[1] if len(row) > 1 else "0"
        prod = row[2] if len(row) > 2 else "0"
        data[prov] = (sup, prod)

    return {crop: data}


def extract_crop_name_from_filename(fname):
    """Extrae el nombre del cultivo del nombre del archivo limpio de AE24."""
    # "7.1.2.5.1 Trigo duro.csv" -> "Trigo duro"
    name = fname.replace(".csv", "")
    # Remove the numeric prefix
    parts = name.split(" ", 1)
    if len(parts) > 1:
        return parts[1]
    return name


def process_folder(folder_path, folder_name):
    """Procesa una carpeta y devuelve los datos unificados."""
    year = get_year(folder_name)
    if year is None:
        return None, None

    # Solo CSVs limpios (los que tienen espacio en el nombre)
    cleaned = [f for f in os.listdir(folder_path)
               if f.endswith(".csv") and " " in f]

    if not cleaned:
        return year, None

    all_crops = {}
    all_provincias = []

    for fname in sorted(cleaned):
        fpath = os.path.join(folder_path, fname)

        with open(fpath, encoding="utf-8-sig") as f:
            header = next(csv.reader(f))

        if len(header) == 3 and header[1].startswith("Superficie"):
            # Formato simple (AE24)
            crop_name = extract_crop_name_from_filename(fname)
            crops = extract_crop_data_single(fpath, crop_name)
        else:
            # Formato multi-cultivo
            crops = extract_crop_data_multi(fpath)

        for crop, data in crops.items():
            all_crops[crop] = data
            for prov in data:
                if prov not in all_provincias:
                    all_provincias.append(prov)

    return year, (all_crops, all_provincias)


def main():
    # Ruta relativa al proyecto (raiz del repo)
    project_root = Path(__file__).resolve().parents[3]
    base = project_root / "data" / "processed" / "manual" / "superficies_provincial_processed"

    if not base.exists():
        print(
            f"Manual data not found: {base}\n"
            "Este paso requiere datos manuales que no se distribuyen en el repositorio "
            "(ver README, seccion 'Advertencia sobre datos manuales'). Es seguro omitir "
            "este paso: los totales por anio ya generados estan disponibles en "
            "data/processed/auto/utils/superficies_provinciales/."
        )
        return

    for folder in sorted(os.listdir(base)):
        folder_path = os.path.join(base, folder)
        if not os.path.isdir(folder_path):
            continue

        year, result = process_folder(folder_path, folder)
        if year is None or result is None:
            print(f"SKIP: {folder}")
            continue

        all_crops, all_provincias = result
        out_year = year - 1

        # Build unified header
        header = ["Provincia"]
        for crop in CROP_COLUMNS:
            header.append(f"{crop} - Superficie (hectáreas)")
            header.append(f"{crop} - Producción (toneladas)")

        # Build rows
        rows = []
        for prov in all_provincias:
            row = [prov]
            for crop in CROP_COLUMNS:
                if crop in all_crops and prov in all_crops[crop]:
                    sup, prod = all_crops[crop][prov]
                    row.extend([sup, prod])
                else:
                    row.extend(["0", "0"])
            rows.append(row)

        out_path = os.path.join(folder_path, f"total_{out_year}.csv")
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        found_crops = [c for c in CROP_COLUMNS if c in all_crops]
        print(f"OK: {folder} -> total_{out_year}.csv ({len(rows)} filas, cultivos: {found_crops})")


if __name__ == "__main__":
    main()
