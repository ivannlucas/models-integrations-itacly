"""
Carga de datos desde data/processed/auto/utils/:
    - prepag2_total.csv             -> precios nacionales MAPA (mensual)
    - climate_monthly_provinces.csv -> clima por provincia (year, month, province_name, temp_mean_C, precip_total_mm)
    - superficies_provinciales/     -> superficie cultivada por provincia y ano
    - Indpag1_total.csv             -> indices de costes grupo 1
    - Indpag2_total.csv             -> indices de costes grupo 2
    - mercados_internacionales.csv  -> mercados internacionales (semanales, USD)
    - eur_usd_weekly.csv            -> tipo de cambio EUR/USD (semanal)
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

try:
    from src.utils.constants import UTILS_DIR
except Exception:
    UTILS_DIR = Path("data/processed/auto/utils")

log = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROVINCE_ALIASES_PATH = PROJECT_ROOT / "config" / "province_aliases.csv"


# --- Normalizacion de nombres de provincia -----------------------------------

def norm_key(s: str | None) -> str | None:
    """Normaliza un nombre a clave canonica (mayusculas, sin tildes ni signos)."""
    if pd.isna(s):
        return None
    s = str(s).upper().strip()
    # Replace en-dash / em-dash / hyphens with space before stripping
    s = re.sub(r"[\u2013\u2014\-]", " ", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9\s]", "", s)
    return " ".join(s.split())


AGG_KEYS = {
    norm_key(k) for k in [
        "ESPANA", "TOTAL", "NACIONAL", "COMUNIDAD AUTONOMA", "REGION",
        "ANDALUCIA", "ARAGON", "C VALENCIANA", "CANARIAS",
        "CASTILLA Y LEON", "CASTILLA LA MANCHA", "CATALUNA",
        "EXTREMADURA", "GALICIA", "PAIS VASCO",
    ]
}
NON_PROVINCE_KEYS = {"PROVINCIAS", "COMUNIDADES AUTONOMAS", "Y"}
NON_PROVINCE_PATTERN = re.compile(r"^\d+\s+(TRIGO|CEBADA|MAIZ)\s+ANALISIS PROVINCIAL")

def _strip_accents_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a ASCII (elimina tildes)."""
    new_cols = {}
    for c in df.columns:
        nc = unicodedata.normalize("NFKD", c)
        nc = "".join(ch for ch in nc if not unicodedata.combining(ch))
        new_cols[c] = nc
    return df.rename(columns=new_cols)


def load_province_aliases(path: Path = PROVINCE_ALIASES_PATH) -> dict[str, str]:
    """Carga alias provinciales normalizados (alias_key -> canonical_key)."""
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el catalogo de aliases provinciales: {path}. "
            "Revisa config/province_aliases.csv."
        )
    aliases = pd.read_csv(path)
    required_cols = {"alias_key", "canonical_key"}
    missing = required_cols - set(aliases.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {path}: {sorted(missing)}. "
            "Se requieren alias_key y canonical_key."
        )

    aliases = aliases[["alias_key", "canonical_key"]].copy()
    aliases["alias_key"] = aliases["alias_key"].apply(norm_key)
    aliases["canonical_key"] = aliases["canonical_key"].apply(norm_key)
    aliases = aliases.dropna(subset=["alias_key", "canonical_key"])
    aliases = aliases[(aliases["alias_key"] != "") & (aliases["canonical_key"] != "")]
    return dict(zip(aliases["alias_key"], aliases["canonical_key"]))


# --- Funciones publicas -------------------------------------------------------

def load_targets(
    utils_dir: Path = UTILS_DIR,
    price_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Carga prepag2_total.csv y devuelve DataFrame con columnas
    ['date'] + price_cols.
    """
    path = utils_dir / "prepag2_total.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.replace(0, np.nan)

    if price_cols is None:
        price_cols = ["trigo", "cebada", "maiz"]

    missing = [c for c in price_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas de precio ausentes en prepag2_total.csv: {missing}")

    df = df[["date"] + price_cols].copy()
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=price_cols, how="all")

    log.info(
        f"Precios nacionales cargados: {df.shape}  "
        f"{df['date'].min()} -> {df['date'].max()}"
    )
    return df


def load_climate(utils_dir: Path = UTILS_DIR) -> pd.DataFrame:
    """
    Carga climate_monthly_provinces.csv.
    El CSV tiene columnas: year, month, province_name, temp_mean_C, precip_total_mm.
    Anade prov_key (nombre normalizado) para cruce con superficies.
    """
    path = utils_dir / "climate_monthly_provinces.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df["year"].astype(str)
        + "-" + df["month"].astype(str).str.zfill(2)
        + "-01"
    )
    df["prov_key"] = df["province_name"].fillna("").apply(norm_key)
    log.info(f"Clima cargado: {df.shape}")
    return df


def load_superficies(utils_dir: Path = UTILS_DIR) -> pd.DataFrame:
    """
    Lee todos los CSV de superficies_provinciales/ y devuelve DataFrame anual
    con superficies y produccion por provincia y cereal.

    Columnas de salida:
        year, province_name,
        total_sup_trigo, total_sup_cebada, total_sup_maiz, total_sup_cereales,
        total_prod_trigo, total_prod_cebada, total_prod_maiz, total_prod_cereales
    """
    sup_dir = utils_dir / "superficies_provinciales"
    clima_df = pd.read_csv(utils_dir / "climate_monthly_provinces.csv")
    alias_to_canonical = load_province_aliases()
    clima_map_norm = {norm_key(k): v for k, v in
                      zip(clima_df["province_name"], clima_df["province_name"])}

    sup_all = []
    for f in sorted(os.listdir(sup_dir)):
        if not f.endswith(".csv"):
            continue
        year_str = f.split("_")[-1].replace(".csv", "")
        try:
            year = int(year_str)
        except ValueError:
            continue
        df = _strip_accents_cols(pd.read_csv(sup_dir / f))
        if "Provincia" not in df.columns:
            continue
        for c in df.columns:
            if c != "Provincia":
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        prov_keys = df["Provincia"].astype(str).fillna("").str.strip().apply(norm_key)

        # Filas no provinciales (cabeceras/interlineados)
        non_prov_mask = prov_keys.isin(NON_PROVINCE_KEYS) | prov_keys.str.match(
            NON_PROVINCE_PATTERN, na=False
        )
        df = df[~non_prov_mask]
        prov_keys = prov_keys[~non_prov_mask]

        # Eliminar agregados
        df = df[~prov_keys.isin(AGG_KEYS)]
        df = df[~df["Provincia"].str.startswith("*", na=False)]
        df["year"] = year
        sup_all.append(df)

    sup = pd.concat(sup_all, ignore_index=True)
    sup["prov_key"] = sup["Provincia"].fillna("").apply(norm_key)
    sup = sup[sup["prov_key"].notna() & (sup["prov_key"] != "")].copy()

    # Resolucion determinista: directo contra clima
    sup["prov_clima"] = sup["prov_key"].map(clima_map_norm)

    # Fallback determinista via catalogo de aliases
    mask_miss = sup["prov_clima"].isna()
    canonical_keys = sup.loc[mask_miss, "prov_key"].map(alias_to_canonical)
    sup.loc[mask_miss, "prov_clima"] = canonical_keys.map(clima_map_norm)

    unresolved = sorted(sup.loc[sup["prov_clima"].isna(), "prov_key"].dropna().unique())
    if unresolved:
        raise ValueError(
            "Provincias sin resolver en superficies->clima: "
            f"{unresolved}. Revisa config/province_aliases.csv."
        )

    sup["province_name"] = sup["prov_clima"]

    sup_cols_map = {
        "total_sup_trigo": [
            "Trigo duro - Superficie (hectareas)",
            "Trigo blando y semiduro - Superficie (hectareas)",
        ],
        "total_sup_cebada": [
            "Cebada 2 carreras - Superficie (hectareas)",
            "Cebada 6 carreras - Superficie (hectareas)",
        ],
        "total_sup_maiz": [
            "Maiz hibrido - Superficie (hectareas)",
            "Otros maices - Superficie (hectareas)",
        ],
    }

    prod_cols_map = {
        "total_prod_trigo": [
            "Trigo duro - Produccion (toneladas)",
            "Trigo blando y semiduro - Produccion (toneladas)",
        ],
        "total_prod_cebada": [
            "Cebada 2 carreras - Produccion (toneladas)",
            "Cebada 6 carreras - Produccion (toneladas)",
        ],
        "total_prod_maiz": [
            "Maiz hibrido - Produccion (toneladas)",
            "Otros maices - Produccion (toneladas)",
        ],
    }

    for new_col, cols in sup_cols_map.items():
        available = [c for c in cols if c in sup.columns]
        if available:
            sup[new_col] = sup[available].sum(axis=1)
        else:
            sup[new_col] = 0.0

    for new_col, cols in prod_cols_map.items():
        available = [c for c in cols if c in sup.columns]
        if available:
            sup[new_col] = sup[available].sum(axis=1)
        else:
            sup[new_col] = 0.0

    sup["total_sup_cereales"] = sup[[
        "total_sup_trigo",
        "total_sup_cebada",
        "total_sup_maiz",
    ]].sum(axis=1)

    sup["total_prod_cereales"] = sup[[
        "total_prod_trigo",
        "total_prod_cebada",
        "total_prod_maiz",
    ]].sum(axis=1)

    keep_cols = [
        "year",
        "province_name",
        "total_sup_trigo",
        "total_sup_cebada",
        "total_sup_maiz",
        "total_sup_cereales",
        "total_prod_trigo",
        "total_prod_cebada",
        "total_prod_maiz",
        "total_prod_cereales",
    ]
    sup = sup[keep_cols].groupby(["year", "province_name"], as_index=False).sum()

    log.info(
        f"Superficies cargadas: {len(sup_all)} anos  provincias mapeadas OK -> {sup.shape}"
    )
    return sup


def load_indices(utils_dir: Path = UTILS_DIR) -> pd.DataFrame:
    """
    Carga Indpag1_total.csv e Indpag2_total.csv y devuelve DataFrame mensual
    con los indices de costes padres (sin sub-indices redundantes).

    Columnas de salida:
        date, idx_fertilizantes, idx_piensos, idx_semillas, idx_bienes_servicios,
        idx_energia, idx_bienes_inversion, idx_fitosanitarios, idx_gastos_generales
    """
    # --- Indpag1 ------------------------------------------------------------
    indpag1 = pd.read_csv(utils_dir / "Indpag1_total.csv", parse_dates=["date"])
    KEEP_1 = [
        "date",
        "fertilizantes",
        "alimentos de ganado",
        "semillas y plantones",
        "bienes y servicios de uso corriente",
    ]
    # Filtrar solo columnas existentes
    keep_1_found = [c for c in KEEP_1 if c in indpag1.columns]
    idx1 = indpag1[keep_1_found].rename(columns={
        "fertilizantes":                          "idx_fertilizantes",
        "alimentos de ganado":                    "idx_piensos",
        "semillas y plantones":                   "idx_semillas",
        "bienes y servicios de uso corriente":    "idx_bienes_servicios",
    })

    # --- Indpag2 ------------------------------------------------------------
    indpag2 = pd.read_csv(utils_dir / "Indpag2_total.csv", parse_dates=["date"])
    KEEP_2 = [
        "date",
        "energia y lubricantes",
        "bienes de inversion",
        "proteccion fitopatologica",
        "gastos generales",
    ]
    keep_2_found = [c for c in KEEP_2 if c in indpag2.columns]
    idx2 = indpag2[keep_2_found].rename(columns={
        "energia y lubricantes":      "idx_energia",
        "bienes de inversion":        "idx_bienes_inversion",
        "proteccion fitopatologica":  "idx_fitosanitarios",
        "gastos generales":           "idx_gastos_generales",
    })

    indices = idx1.merge(idx2, on="date", how="outer").sort_values("date").reset_index(drop=True)
    log.info(f"Indices cargados: {indices.shape}")
    return indices


def load_markets_intl(utils_dir: Path = UTILS_DIR) -> pd.DataFrame:
    """
    Carga mercados_internacionales.csv + eur_usd_weekly.csv (ambos semanales),
    hace resample mensual y convierte precios a EUR.

    Columnas de salida:
        date, wheat_intl_eur, corn_intl_eur, fao_cereals_idx
    """
    intl   = pd.read_csv(utils_dir / "mercados_internacionales.csv",  parse_dates=["week_start"])
    eurusd = pd.read_csv(utils_dir / "eur_usd_weekly.csv",             parse_dates=["week_start"])

    # Semanal -> mensual (media)
    intl_m   = intl.set_index("week_start").resample("MS").mean()
    eurusd_m = eurusd.set_index("week_start").resample("MS").mean()

    mkts = intl_m.merge(eurusd_m, left_index=True, right_index=True, how="inner")

    # Convertir a EUR
    if "Wheat_Global_Proxy_USD" in mkts.columns and "EUR_USD" in mkts.columns:
        mkts["wheat_intl_eur"] = mkts["Wheat_Global_Proxy_USD"] / mkts["EUR_USD"]
    if "ZC_USD" in mkts.columns and "EUR_USD" in mkts.columns:
        mkts["corn_intl_eur"] = mkts["ZC_USD"] / mkts["EUR_USD"]
    if "CerealsFAO Price Index" in mkts.columns:
        mkts["fao_cereals_idx"] = mkts["CerealsFAO Price Index"]

    output_cols = [c for c in ["wheat_intl_eur", "corn_intl_eur", "fao_cereals_idx"]
                   if c in mkts.columns]
    mkts = mkts[output_cols].reset_index().rename(columns={"week_start": "date"})
    log.info(f"Mercados intl (EUR mensual): {mkts.shape}  "
             f"{mkts['date'].min()} -> {mkts['date'].max()}")
    return mkts

