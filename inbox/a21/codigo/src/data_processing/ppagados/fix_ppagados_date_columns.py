"""Normaliza columnas de fecha en todos los CSVs de ppagados1_2 y genera _fixed.csv.

Para cada CSV: normaliza nombres de columna (unicode, saltos de línea), detecta la
columna de fecha (date/fecha/fecha_inicio/mes o usa la primera columna como fallback),
la convierte a datetime y reordena para que 'date' sea la primera columna.

Entradas:
  - data/processed/manual/ppagados1_2/*.csv
    (archivos de índices de precios pagados convertidos desde XLSX)

Salidas:
  - data/processed/manual/ppagados1_2/*_fixed.csv
    (mismo contenido en formato ancho, columna date normalizada y colocada al inicio)

Uso:
  python src/data_processing/ppagados/fix_ppagados_date_columns.py
"""
import os
import glob
import pandas as pd
import unicodedata

FOLDER = r'data/processed/manual/ppagados1_2'
# Only process original CSVs to avoid repeated *_fixed_fixed outputs.
files = [
    f for f in sorted(glob.glob(os.path.join(FOLDER, '*.csv')))
    if "_fixed" not in os.path.basename(f)
]

def normalize_col(c):
    if pd.isna(c):
        return ''
    s = str(c).strip()
    s = unicodedata.normalize('NFKD', s)
    s = s.replace('\n',' ').replace('\r',' ')
    return s

for f in files:
    try:
        df = pd.read_csv(f)
    except Exception:
        try:
            df = pd.read_csv(f, encoding='latin-1')
        except Exception as e:
            print('ERR read', f, e)
            continue
    # normalize columns
    df.columns = [normalize_col(c) for c in df.columns]
    # ensure date column exists and is parsed
    date_cols = [c for c in df.columns if c.lower().strip() in ('date','fecha','fecha_inicio','mes')]
    if date_cols:
        dcol = date_cols[0]
        try:
            df['date'] = pd.to_datetime(df[dcol], errors='coerce')
            df = df.drop(columns=[dcol]) if dcol != 'date' else df
        except Exception:
            df['date'] = pd.to_datetime(df[dcol].astype(str), errors='coerce')
    else:
        # assume first column is date
        first = df.columns[0]
        df['date'] = pd.to_datetime(df[first], errors='coerce')
        if first != 'date':
            df = df.drop(columns=[first])
    # reorder to have date first
    cols = ['date'] + [c for c in df.columns if c!='date']
    df = df[cols]
    out = os.path.splitext(f)[0] + '_fixed.csv'
    df.to_csv(out, index=False)
    print('WROTE', out)
