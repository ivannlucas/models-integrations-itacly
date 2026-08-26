#!/usr/bin/env python3
"""Construye los totales unificados de ppagados con rebasing a base 2020 (versión heurística).

Une las cuatro carpetas de series históricas (ppagados1/1_2/2/3) fusionando por
nombre de producto con matching difuso y calculando factores de conversión mediante
la mediana del ratio en las fechas de solapamiento. Prefiere los valores más recientes
cuando hay solapamiento entre versiones.

Entradas:
  - data/processed/manual/ppagados{1,1_2,2,3}/*_fixed.csv
    (series de precios en formato ancho con columna date, generados por fix_ppagados_date_columns.py
     o reshape_indpag4_timeseries.py)

Salidas:
  - data/processed/auto/ppagadostotal/prepag1_total.csv   (PrePag INPUT I, sin rebasing)
  - data/processed/auto/ppagadostotal/prepag2_total.csv   (PrePag INPUT II, sin rebasing)
  - data/processed/auto/ppagadostotal/Indpag1_total.csv   (IndPag INPUT I, rebased a 2020)
  - data/processed/auto/ppagadostotal/Indpag2_total.csv   (IndPag INPUT II, rebased a 2020)
  - data/processed/auto/ppagadostotal/report.txt          (log de factores y productos sin factor)

Uso:
  python src/data_processing/ppagados/build_ppagados_unified.py
"""
from pathlib import Path
import os
import glob
import re
import unicodedata
import difflib
import numpy as np
import pandas as pd

OUT = Path('data/processed/auto/ppagadostotal')
OUT.mkdir(parents=True, exist_ok=True)

# Orden de antigueda a actualidad: cada tier nuevo (mas a la derecha) tiene
# prioridad sobre los anteriores donde se solapen (ver apply_factors_and_merge).
# ppagados4 se genera con ingest_indices_precios_pagados.py a partir del ultimo
# boletin MAPA "Indices y Precios Pagados Agrarios" descargado.
FOLDERS = ['ppagados1','ppagados1_2','ppagados2','ppagados3','ppagados4']

def normalize_prod(s):
    if pd.isna(s):
        return ''
    s = str(s)
    s = s.strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\(.*?\)", '', s)  # remove parenthesis
    s = re.sub(r"[^a-z0-9\s]", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s

def read_fixed(folder, pattern):
    """Read a single fixed CSV (pattern like '*IndPag3*_fixed.csv') and return pivoted DataFrame."""
    files = glob.glob(str(Path('data/processed/manual')/folder/pattern))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f, parse_dates=['date'])
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    m = df_all.melt(id_vars=['date'], var_name='product', value_name='val')
    m['prod_norm'] = m['product'].apply(normalize_prod)
    m['val'] = pd.to_numeric(m['val'], errors='coerce')
    pv = m.pivot_table(index='date', columns='prod_norm', values='val', aggfunc='first')
    pv = pv.sort_index()
    return pv

def match_and_compute_factors(target_pv, source_pv, min_overlap=6, score_thresh=0.7):
    """For each column in source_pv, match to best in target_pv and compute median factor target/source."""
    factors = {}
    mapping = {}
    target_cols = list(target_pv.columns)
    for src in source_pv.columns:
        # best fuzzy match
        best = None
        best_score = 0
        for t in target_cols:
            score = difflib.SequenceMatcher(None, src, t).ratio()
            if score > best_score:
                best_score = score; best = t
        if best_score < score_thresh:
            mapping[src] = (None, best_score)
            continue
        # compute median ratio over overlap
        idx = target_pv.index.intersection(source_pv.index)
        s1 = target_pv[best].loc[idx].dropna()
        s2 = source_pv[src].loc[idx].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < min_overlap:
            mapping[src] = (best, best_score)
            continue
        r = (s1.loc[common] / s2.loc[common]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) == 0:
            mapping[src] = (best, best_score)
            continue
        factors[src] = float(np.median(r.values))
        mapping[src] = (best, best_score)
    return factors, mapping


def compute_chain_factors(per_folder_pvs, min_overlap=6, score_thresh=0.7):
    """Compute pairwise factors for consecutive folders: 1->1_2, 1_2->2, 2->3."""
    # per_folder_pvs: dict folder->pv
    f_1_1_2, m_1_1_2 = {}, {}
    f_1_2_2, m_1_2_2 = {}, {}
    f_2_3, m_2_3 = {}, {}
    tgt = per_folder_pvs.get('ppagados3', pd.DataFrame())
    pv2 = per_folder_pvs.get('ppagados2', pd.DataFrame())
    pv1_2 = per_folder_pvs.get('ppagados1_2', pd.DataFrame())
    pv1 = per_folder_pvs.get('ppagados1', pd.DataFrame())
    if not pv2.empty and not tgt.empty:
        f_2_3, m_2_3 = match_and_compute_factors(tgt, pv2, min_overlap=min_overlap, score_thresh=score_thresh)
    if not pv1_2.empty and not pv2.empty:
        f_1_2_2, m_1_2_2 = match_and_compute_factors(pv2, pv1_2, min_overlap=min_overlap, score_thresh=score_thresh)
    if not pv1.empty and not pv1_2.empty:
        f_1_1_2, m_1_1_2 = match_and_compute_factors(pv1_2, pv1, min_overlap=min_overlap, score_thresh=score_thresh)
    return (f_1_1_2, m_1_1_2, f_1_2_2, m_1_2_2, f_2_3, m_2_3)

def apply_factors_and_merge(per_folder_pvs, canonical_cols):
    """Apply factors per folder to map to 2020 base and merge preferring newest folders."""
    # per_folder_pvs: dict folder->pv
    # canonical_cols: list of normalized product names from ppagados3
    merged = pd.DataFrame()
    details = []
    target = per_folder_pvs['ppagados3']

    per_folder_rebased = {}
    # precompute chain factors to allow composition
    (f_1_1_2, m_1_1_2, f_1_2_2, m_1_2_2, f_2_3, m_2_3) = compute_chain_factors(per_folder_pvs)

    for folder in FOLDERS:
        pv = per_folder_pvs.get(folder)
        if pv is None or pv.empty:
            continue
        # match to target and compute direct factor
        factors, mapping = match_and_compute_factors(target, pv)
        # apply factor where available, otherwise leave as-is but mark
        pv_rebased = pv.copy()
        for col in pv.columns:
            f = factors.get(col)
            mapped_to = mapping.get(col, (None,0))[0]
            # if no direct factor, try to compose along the chain
            if f is None:
                # folder -> ppagados2 -> ppagados3
                if folder == 'ppagados2':
                    # try f from 2->3 already computed
                    f = f_2_3.get(col)
                    mapped_to = m_2_3.get(col, (None,0))[0] if m_2_3.get(col) else mapped_to
                elif folder == 'ppagados1_2':
                    # try 1_2->2 then 2->3
                    f12 = f_1_2_2.get(col)
                    mapped_in_2 = m_1_2_2.get(col, (None,0))[0] if m_1_2_2.get(col) else None
                    if f12 is not None and mapped_in_2 is not None:
                        f23 = f_2_3.get(mapped_in_2)
                        if f23 is not None:
                            f = f12 * f23
                            mapped_to = mapped_in_2
                elif folder == 'ppagados1':
                    # try 1->1_2 -> 1_2->2 -> 2->3
                    f11 = f_1_1_2.get(col)
                    mapped_in_b = m_1_1_2.get(col, (None,0))[0] if m_1_1_2.get(col) else None
                    if f11 is not None and mapped_in_b is not None:
                        f_b2 = f_1_2_2.get(mapped_in_b)
                        mapped_in_c = m_1_2_2.get(mapped_in_b, (None,0))[0] if m_1_2_2.get(mapped_in_b) else None
                        if f_b2 is not None and mapped_in_c is not None:
                            f_c3 = f_2_3.get(mapped_in_c)
                            if f_c3 is not None:
                                f = f11 * f_b2 * f_c3
                                mapped_to = mapped_in_c

            if f is not None:
                pv_rebased[col] = pv_rebased[col] * f
                details.append((folder, col, mapped_to, f))
            else:
                details.append((folder, col, mapped_to, None))
        # rename columns to mapped canonical name when mapping exists
        rename = {col: (mapping[col][0] if mapping.get(col) and mapping[col][0] else col) for col in pv.columns}
        pv_rebased = pv_rebased.rename(columns=rename)
        # collapse duplicate column labels by taking first non-null across duplicates
        if pv_rebased.columns.duplicated().any():
            cols_ordered = list(dict.fromkeys(list(pv_rebased.columns)))
            new = pd.DataFrame(index=pv_rebased.index)
            for col in cols_ordered:
                dup = [c for c in pv_rebased.columns if c == col]
                if len(dup) == 1:
                    new[col] = pv_rebased[dup[0]]
                else:
                    tmp = pv_rebased[dup].bfill(axis=1).iloc[:, 0]
                    new[col] = tmp
            pv_rebased = new
        # store per-folder rebased pivot for later priority-merge
        per_folder_rebased[folder] = pv_rebased

    # merge per-folder rebased pvs preferring newest folders (ppagados3 highest priority)
    merged = pd.DataFrame()
    for folder in reversed(FOLDERS):
        pv = per_folder_rebased.get(folder)
        if pv is None or pv.empty:
            continue
        if merged.empty:
            merged = pv.copy()
        else:
            # keep existing merged values (higher priority), fill missing from pv
            merged = merged.combine_first(pv)

    # ensure canonical columns present
    for c in canonical_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged.reindex(sorted(merged.columns), axis=1)
    merged = merged.sort_index()
    return merged, details

def write_csv(pv, path):
    pv.index.name = 'date'
    pv.to_csv(path, date_format='%Y-%m-%d', float_format='%.2f')

def main():
    # Read IndPag3 (INPUT I) and IndPag4 (INPUT II) per folder
    per_folder_ind3 = {f: read_fixed(f, '*IndPag3*_fixed.csv') for f in FOLDERS}
    per_folder_ind4 = {f: read_fixed(f, '*IndPag4*_fixed.csv') for f in FOLDERS}
    per_folder_prepag1 = {f: read_fixed(f, '*PrePag1*_fixed.csv') for f in FOLDERS}
    per_folder_prepag2 = {f: read_fixed(f, '*PrePag2*_fixed.csv') for f in FOLDERS}

    target3_ind3 = per_folder_ind3.get('ppagados3', pd.DataFrame())
    target3_ind4 = per_folder_ind4.get('ppagados3', pd.DataFrame())

    canonical_ind3 = list(target3_ind3.columns) if not target3_ind3.empty else []
    canonical_ind4 = list(target3_ind4.columns) if not target3_ind4.empty else []

    ind3_merged, ind3_details = apply_factors_and_merge(per_folder_ind3, canonical_ind3)
    ind4_merged, ind4_details = apply_factors_and_merge(per_folder_ind4, canonical_ind4)

    # Prepag: merge without rebasing, prefer newest folders
    pre1_merged, pre1_details = apply_factors_and_merge(per_folder_prepag1, list(per_folder_prepag1.get('ppagados3', pd.DataFrame()).columns if not per_folder_prepag1.get('ppagados3', pd.DataFrame()).empty else []))
    pre2_merged, pre2_details = apply_factors_and_merge(per_folder_prepag2, list(per_folder_prepag2.get('ppagados3', pd.DataFrame()).columns if not per_folder_prepag2.get('ppagados3', pd.DataFrame()).empty else []))

    # Write outputs
    write_csv(pre1_merged, OUT/'prepag1_total.csv')
    write_csv(pre2_merged, OUT/'prepag2_total.csv')
    write_csv(ind3_merged, OUT/'Indpag1_total.csv')
    write_csv(ind4_merged, OUT/'Indpag2_total.csv')

    # Report
    rpt = []
    rpt.append('ppagadostotal build report')
    rpt.append('Folders: ' + ','.join(FOLDERS))
    rpt.append('\nINDPAG3 (INPUT I) mapping details:')
    for d in ind3_details[:200]:
        rpt.append(str(d))
    rpt.append('\nINDPAG4 (INPUT II) mapping details:')
    for d in ind4_details[:200]:
        rpt.append(str(d))
    rpt.append('\nPREPAG1 details:')
    for d in pre1_details[:200]:
        rpt.append(str(d))

    (OUT/'report.txt').write_text('\n'.join(rpt), encoding='utf-8')
    print('WROTE 4 totals and report to', OUT)


if __name__ == '__main__':
    main()
