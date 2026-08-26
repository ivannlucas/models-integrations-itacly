"""Construye un dataset panel espacial con fuentes macro y provinciales."""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.load_raw import (
    load_climate,
    load_indices,
    load_markets_intl,
    load_superficies,
    load_targets,
)
from src.utils.io import save_dataset
from src.utils.logging import get_logger


UTILS_DIR = Path("data/processed/auto/utils")
PROCESSED = Path("data/processed")
SPATIAL_MASTER_PATH = Path("data/processed/auto/utils/provincias_master_final.csv")
SPATIAL_NEIGHBOR_KM = 250.0
# Desfase administrativo de publicacion oficial del MAPA.
# Garantiza que las features derivadas de MAPA solo usen datos disponibles en produccion.
MAPA_ADMIN_LAG: int = 3

warnings.filterwarnings("ignore")
log = get_logger(__name__)


def _to_month_start(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Normaliza fechas al primer dia del mes.

    Args:
        df: DataFrame con columna de fechas.
        date_col: Nombre de la columna de fechas.

    Returns:
        DataFrame con fechas normalizadas.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    return df


def _load_prepag(path: Path, prefix: str | None = None) -> pd.DataFrame:
    """Carga un CSV de prepag y convierte columnas numericas.

    Args:
        path: Ruta al CSV.
        prefix: Prefijo opcional para columnas (excepto date).

    Returns:
        DataFrame con columnas normalizadas.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _to_month_start(df, "date")
    if prefix:
        rename_map = {c: f"{prefix}{c}" for c in df.columns if c != "date"}
        df = df.rename(columns=rename_map)
    return df


def expand_surfaces_monthly(sup_df: pd.DataFrame) -> pd.DataFrame:
    """Expande superficies anuales a mensual.

    Args:
        sup_df: DataFrame anual de superficies.

    Returns:
        DataFrame mensual.
    """
    rows = []
    for row in sup_df.itertuples(index=False):
        for m in range(1, 13):
            rows.append(
                {
                    "date": pd.Timestamp(int(row.year), m, 1),
                    "province_name": row.province_name,
                    "total_sup_trigo": row.total_sup_trigo,
                    "total_sup_cebada": row.total_sup_cebada,
                    "total_sup_maiz": row.total_sup_maiz,
                    "total_sup_cereales": row.total_sup_cereales,
                }
            )
    return pd.DataFrame(rows)


def select_top_combinations_by_surface(
    sup_df: pd.DataFrame,
    clima_df: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Selecciona combinaciones provincia-cereal por hectareas medias.

    Args:
        sup_df: DataFrame anual de superficies.
        clima_df: DataFrame mensual de clima.
        top_n: Numero de combinaciones a conservar.

    Returns:
        DataFrame con combinaciones provincia-cereal.
    """
    sup_avg = sup_df.groupby("province_name")[
        ["total_sup_trigo", "total_sup_cebada", "total_sup_maiz"]
    ].mean()
    sup_long = sup_avg.reset_index().melt(
        id_vars=["province_name"],
        value_vars=["total_sup_trigo", "total_sup_cebada", "total_sup_maiz"],
        var_name="cereal_col",
        value_name="hectareas_mean",
    )
    col_to_cereal = {
        "total_sup_trigo": "trigo",
        "total_sup_cebada": "cebada",
        "total_sup_maiz": "maiz",
    }
    sup_long["cereal_predominante"] = sup_long["cereal_col"].map(col_to_cereal)
    sup_long = sup_long.drop(columns=["cereal_col"])
    sup_long = sup_long[sup_long["hectareas_mean"] > 0].copy()

    clima_prov = set(clima_df["province_name"].dropna().unique().tolist())
    sup_long = sup_long[sup_long["province_name"].isin(clima_prov)]

    sup_long = sup_long.sort_values("hectareas_mean", ascending=False)
    return sup_long.head(top_n).reset_index(drop=True)


def add_expanding_zscore(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    z_col: str,
    date_col: str = "date",
    min_periods: int = 12,
) -> pd.DataFrame:
    """Calcula z-score con ventana expansiva usando solo historia (t-1).

    Args:
        df: DataFrame de entrada.
        group_cols: Columnas de agrupacion.
        value_col: Columna de valores.
        z_col: Nombre de salida para el z-score.
        date_col: Columna de fechas.

    Returns:
        DataFrame con z-score agregado.
    """
    df = df.sort_values(group_cols + [date_col]).copy()

    def _expanding_z(s: pd.Series) -> pd.Series:
        history = s.shift(1)
        mean = history.expanding(min_periods=min_periods).mean()
        std = history.expanding(min_periods=min_periods).std(ddof=0)
        z = (s - mean) / std.replace(0, np.nan)
        return z.fillna(0.0)

    df[z_col] = df.groupby(group_cols, sort=False)[value_col].transform(_expanding_z)
    return df


def add_expanding_mean(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    out_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """Calcula media expansiva desplazada para evitar leakage.

    Args:
        df: DataFrame de entrada.
        group_cols: Columnas de agrupacion.
        value_col: Columna de valores.
        out_col: Nombre de salida.
        date_col: Columna de fechas.

    Returns:
        DataFrame con media expansiva.
    """
    df = df.sort_values(group_cols + [date_col]).copy()

    def _expanding_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding().mean()

    df[out_col] = df.groupby(group_cols, sort=False)[value_col].transform(_expanding_mean)
    return df


def add_lags(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    lags: list[int],
    date_col: str = "date",
) -> pd.DataFrame:
    """Agrega columnas de lags para un conjunto de columnas.

    Args:
        df: DataFrame de entrada.
        group_cols: Columnas de agrupacion.
        value_cols: Columnas a desplazar.
        lags: Lista de lags.
        date_col: Columna de fechas.

    Returns:
        DataFrame con lags.
    """
    df = df.sort_values(group_cols + [date_col]).copy()
    for col in value_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df.groupby(group_cols, sort=False)[col].shift(lag)
    return df


def extend_prices_ffill(
    prices_long: pd.DataFrame,
    group_col: str,
    value_col: str,
    max_date: pd.Timestamp,
    date_col: str = "date",
) -> pd.DataFrame:
    """Extiende una serie de precios mensual hasta max_date, arrastrando
    (forward-fill) el ultimo valor real conocido por grupo.

    Uso exclusivo para derivar features de lag/rolling en el frente de
    produccion (meses sin precio MAPA aun publicado todavia): el valor real
    de `value_col` no se toca donde ya existe (ffill nunca reescribe un dato
    presente), solo rellena huecos futuros para que los lags y medias
    moviles calculados aguas abajo no salgan en NaN. No debe usarse para
    construir el target semi-sintetico (`precio_provincial_TARGET` usa la
    columna original, sin extender, ver `merge_spatial_macro`).
    """
    groups = prices_long[group_col].unique()
    all_dates = pd.date_range(prices_long[date_col].min(), max_date, freq="MS")
    full_idx = pd.MultiIndex.from_product([all_dates, groups], names=[date_col, group_col])
    extended = (
        prices_long.set_index([date_col, group_col])
        .reindex(full_idx)
        .reset_index()
    )
    extended[value_col] = extended.groupby(group_col)[value_col].ffill()
    return extended


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    windows: list[int],
    date_col: str = "date",
) -> pd.DataFrame:
    """Agrega medias moviles y volatilidad con ventana pasada.

    Args:
        df: DataFrame de entrada.
        group_cols: Columnas de agrupacion.
        value_col: Columna base.
        windows: Ventanas para rolling.
        date_col: Columna de fechas.

    Returns:
        DataFrame con features rolling.
    """
    df = df.sort_values(group_cols + [date_col]).copy()

    def _roll_mean(s: pd.Series, window: int) -> pd.Series:
        return s.shift(1).rolling(window=window).mean()

    def _roll_std(s: pd.Series, window: int) -> pd.Series:
        return s.shift(1).rolling(window=window).std(ddof=0)

    for w in windows:
        df[f"{value_col}_ma{w}"] = df.groupby(group_cols, sort=False)[value_col].transform(
            lambda s: _roll_mean(s, w)
        )
        df[f"{value_col}_vol{w}"] = df.groupby(group_cols, sort=False)[value_col].transform(
            lambda s: _roll_std(s, w)
        )
    return df


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calcula distancia Haversine en km para arrays.

    Args:
        lat1: Latitudes punto 1.
        lon1: Longitudes punto 1.
        lat2: Latitudes punto 2.
        lon2: Longitudes punto 2.

    Returns:
        Distancias en km.
    """
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def add_spatial_price_lag(
    panel: pd.DataFrame,
    threshold_km: float = SPATIAL_NEIGHBOR_KM,
    price_col: str = "precio_provincial_lag_1",
    output_col: str = "precio_vecinos_media_lag1",
) -> pd.DataFrame:
    """Calcula la media espacial de precios vecinos en t-1.

    Args:
        panel: DataFrame con columnas provincia_id, lat_centroide, lon_centroide.
        threshold_km: Umbral de vecindad en km.
        price_col: Columna con precio en t-1.

    Returns:
        DataFrame con columna precio_vecinos_media_lag1.
    """
    panel = panel.copy()
    provinces = panel[["provincia_id", "lat_centroide", "lon_centroide"]].drop_duplicates()
    provinces = provinces.sort_values("provincia_id").reset_index(drop=True)

    lat = provinces["lat_centroide"].to_numpy()
    lon = provinces["lon_centroide"].to_numpy()
    lat1 = lat[:, None]
    lon1 = lon[:, None]
    lat2 = lat[None, :]
    lon2 = lon[None, :]
    dist = _haversine_km(lat1, lon1, lat2, lon2)
    neighbor_mask = (dist < threshold_km) & (dist > 0)
    neighbor_mask = neighbor_mask.astype(float)

    idx_map = {pid: i for i, pid in enumerate(provinces["provincia_id"].tolist())}
    n = len(provinces)

    def _compute_group(group: pd.DataFrame) -> pd.DataFrame:
        v = np.full(n, np.nan)
        idxs = group["provincia_id"].map(idx_map).to_numpy()
        v[idxs] = group[price_col].to_numpy()
        v_filled = np.nan_to_num(v, nan=0.0)
        valid = (~np.isnan(v)).astype(float)
        neighbor_sum = neighbor_mask @ v_filled
        neighbor_count = neighbor_mask @ valid
        mean = np.divide(
            neighbor_sum,
            neighbor_count,
            out=np.full_like(neighbor_sum, np.nan),
            where=neighbor_count > 0,
        )
        group = group.copy()
        group[output_col] = mean[idxs]
        return group

    date_series = panel["date"]
    cereal_series = panel["cereal_predominante"]
    result = panel.groupby(["date", "cereal_predominante"], sort=False, group_keys=False).apply(_compute_group)
    if "date" not in result.columns:
        result["date"] = date_series.reindex(result.index)
    if "cereal_predominante" not in result.columns:
        result["cereal_predominante"] = cereal_series.reindex(result.index)
    return result


def load_all_sources(
    utils_dir: Path,
    price_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """Carga todas las fuentes necesarias desde utils.

    Args:
        utils_dir: Directorio de fuentes procesadas.
        price_cols: Columnas de precios nacionales.

    Returns:
        Diccionario de DataFrames.
    """
    clima = load_climate(utils_dir)
    superficies = load_superficies(utils_dir)

    prepag1 = _load_prepag(utils_dir / "prepag1_total.csv", prefix="prepag1_")
    prepag2_full = _load_prepag(utils_dir / "prepag2_total.csv")

    precios = load_targets(utils_dir, price_cols=price_cols)
    indices = load_indices(utils_dir)
    mercados = load_markets_intl(utils_dir)

    eurusd = pd.read_csv(utils_dir / "eur_usd_weekly.csv", parse_dates=["week_start"])
    eurusd_m = (
        eurusd.set_index("week_start").resample("MS").mean().reset_index()
    )
    eurusd_m = eurusd_m.rename(columns={"week_start": "date", "EUR_USD": "eur_usd"})
    eurusd_m = _to_month_start(eurusd_m, "date")

    prepag2_macro = prepag2_full.drop(columns=price_cols, errors="ignore")
    if len(prepag2_macro.columns) > 1:
        prepag2_macro = prepag2_macro.rename(
            columns={c: f"prepag2_{c}" for c in prepag2_macro.columns if c != "date"}
        )

    return {
        "clima": clima,
        "superficies": superficies,
        "prepag1": prepag1,
        "prepag2": prepag2_full,
        "prepag2_macro": prepag2_macro,
        "precios": precios,
        "indices": indices,
        "mercados": mercados,
        "eurusd": eurusd_m,
    }


def merge_spatial_macro(
    sources: dict[str, pd.DataFrame],
    top_n: int,
    price_cols: list[str],
) -> pd.DataFrame:
    """Combina variables provinciales y macro por fecha.

    Args:
        sources: Diccionario de fuentes cargadas.
        top_n: Numero de combinaciones provincia-cereal.
        price_cols: Columnas de precios nacionales.

    Returns:
        DataFrame panel combinado.
    """
    clima = sources["clima"].copy()
    sup = sources["superficies"].copy()
    sup_monthly = expand_surfaces_monthly(sup)

    top_combos = select_top_combinations_by_surface(sup, clima, top_n=top_n)
    if top_combos.empty:
        raise ValueError("No se encontraron combinaciones validas para el panel.")

    log.info("Top %s combinaciones por hectareas absolutas:", top_n)
    log.info(
        "\n%s",
        top_combos.rename(columns={"province_name": "provincia"}).to_string(index=False)
    )
    selected = sorted(top_combos["province_name"].unique().tolist())

    panel = clima.merge(
        sup_monthly,
        on=["province_name", "date"],
        how="left",
    )
    panel = panel[panel["province_name"].isin(selected)].copy()
    prov_cereal_df = top_combos[["province_name", "cereal_predominante"]].drop_duplicates()
    panel = panel.merge(prov_cereal_df, on="province_name", how="inner")

    surface_cols = [
        "total_sup_trigo",
        "total_sup_cebada",
        "total_sup_maiz",
        "total_sup_cereales",
    ]
    panel = panel.sort_values(["province_name", "date"])
    panel[surface_cols] = panel.groupby("province_name")[surface_cols].ffill().bfill()
    panel[surface_cols] = panel[surface_cols].fillna(0.0)

    panel["sup_cereal_predominante"] = np.select(
        [
            panel["cereal_predominante"] == "trigo",
            panel["cereal_predominante"] == "cebada",
            panel["cereal_predominante"] == "maiz",
        ],
        [
            panel["total_sup_trigo"],
            panel["total_sup_cebada"],
            panel["total_sup_maiz"],
        ],
        default=np.nan,
    )

    prices_long_real = sources["precios"].melt(
        id_vars=["date"],
        value_vars=price_cols,
        var_name="cereal_predominante",
        value_name="precio_nacional_base",
    )

    # Los lags/medias moviles del precio nacional se calculan sobre una
    # version extendida (forward-fill) hasta el ultimo mes del panel, para
    # que el frente de produccion (meses sin precio MAPA aun publicado) no
    # se quede sin estas features -- igual que ya ocurre con los indices de
    # coste via MAPA_ADMIN_LAG, que usan lag_3/lag_4 y por tanto ya alcanzan
    # el frente de produccion sin ayuda. El precio "real" (columna
    # precio_nacional_base sin extender) se restaura despues, intacto, para
    # que el target semi-sintetico nunca se fabrique donde no hay dato MAPA
    # real.
    panel_max_date = panel["date"].max()
    prices_long = extend_prices_ffill(
        prices_long_real, "cereal_predominante", "precio_nacional_base", panel_max_date,
    )
    prices_long = add_lags(
        prices_long,
        group_cols=["cereal_predominante"],
        value_cols=["precio_nacional_base"],
        lags=[1, 2, 3],
    )
    prices_long = add_rolling_features(
        prices_long,
        group_cols=["cereal_predominante"],
        value_col="precio_nacional_base",
        windows=[3, 6],
    )
    real_base = prices_long_real.set_index(["date", "cereal_predominante"])["precio_nacional_base"]
    prices_long = prices_long.set_index(["date", "cereal_predominante"])
    prices_long["precio_nacional_base"] = real_base.reindex(prices_long.index)
    prices_long = prices_long.reset_index()

    panel = panel.merge(prices_long, on=["date", "cereal_predominante"], how="left")

    macro = sources["prepag1"].merge(
        sources["indices"], on="date", how="outer"
    ).merge(
        sources["mercados"], on="date", how="outer"
    ).merge(
        sources["eurusd"], on="date", how="outer"
    )
    prepag2_macro = sources["prepag2_macro"]
    if "date" in prepag2_macro.columns:
        macro = macro.merge(prepag2_macro, on="date", how="outer")
    macro = macro.sort_values("date").reset_index(drop=True)

    panel = panel.merge(macro, on="date", how="left")

    # Suelo temporal explicito: antes, exigir precio_nacional_base no-nulo en
    # este dropna truncaba implicitamente el panel a partir de la primera
    # fecha con precio MAPA real (climaticamente hay historia bastante
    # anterior, ver data/processed/manual/climate_provinces_GEE_by_year, que
    # arranca en 2002). precio_nacional_base ya NO esta en este dropna (ver
    # mas abajo, es una columna de la blacklist anti-leakage, solo alimenta
    # el target), asi que ese suelo hay que fijarlo aqui explicitamente: sin
    # esto, filas anteriores a la primera fecha con precio MAPA real (2003-
    # 2005 en los datos actuales) se colarian en apply_feature_engineering y
    # alterarian el historial de los z-scores expansivos (z_clima_adverso,
    # z_sup_cereal_predominante) para TODO el periodo de entrenamiento --
    # verificado: sin este suelo, cambiaban celdas de 2015-2017 en adelante.
    min_real_price_date = prices_long_real.loc[
        prices_long_real["precio_nacional_base"].notna(), "date"
    ].min()
    panel = panel[panel["date"] >= min_real_price_date]

    # precio_nacional_base deliberadamente NO esta en este dropna: es una
    # columna de la blacklist anti-leakage (solo alimenta el target
    # semi-sintetico, no es feature), y exigirla aqui bloqueaba filas enteras
    # del frente de produccion (meses sin precio MAPA aun publicado) aunque
    # sus features derivadas ya sean validas via extend_prices_ffill. La
    # validez del target por horizonte ya se aplica en prepare_data.py,
    # que es el lugar correcto (mismo criterio que precio_provincial_TARGET_H1).
    panel = panel.dropna(subset=["temp_mean_C", "precip_total_mm"])
    return panel


def _deterministic_row_noise(
    dates: pd.Series,
    provincias: pd.Series,
    cereales: pd.Series,
    noise_std: float,
    random_seed: int,
) -> np.ndarray:
    """Ruido gaussiano estable por identidad de fila (fecha+provincia+cereal).

    Antes, el ruido se generaba con ``rng.normal(0, std, len(panel))`` ANTES del
    sort final del panel: dependia de la POSICION de cada fila en el array en
    ese momento, no de su identidad. Cualquier cambio aguas arriba que alterase
    cuantas filas sobreviven al filtrado previo (p.ej. refrescar datos MAPA y
    que aparezcan meses nuevos) desplazaba silenciosamente el ruido -y por
    tanto el target semi-sintetico completo- de la mayoria de filas
    posteriores al primer punto de insercion, incluyendo años de entrenamiento
    ya congelados (detectado al ampliar el panel con MAPA hasta marzo 2026).

    Esta version fija el ruido de cada fila a partir de un hash estable de su
    propia clave (fecha, provincia, cereal), por lo que anadir filas nuevas
    (mas meses, mas fuentes) nunca cambia el ruido -ni el target- de una fila
    que ya existia: el dataset de entrenamiento queda anclado a los mismos
    años para siempre, independientemente de cuantas veces se reconstruya el
    panel con datos MAPA mas recientes.
    """
    keys = (
        dates.dt.strftime("%Y-%m-%d") + "|" + provincias.astype(str) + "|" + cereales.astype(str)
    )
    noise = np.empty(len(keys), dtype=float)
    for i, key in enumerate(keys):
        digest = hashlib.sha256(f"{random_seed}|{key}".encode("utf-8")).digest()
        row_seed = int.from_bytes(digest[:8], "big")
        noise[i] = np.random.default_rng(row_seed).normal(0.0, noise_std)
    return noise


def apply_feature_engineering(
    panel: pd.DataFrame,
    alpha: float,
    beta: float,
    noise_std: float,
    random_seed: int,
) -> pd.DataFrame:
    """Aplica ingenieria de caracteristicas y target semi-sintetico.

    El target semi-sintetico se construye mediante la siguiente formula matematica:

        TARGET = Precio_Nacional_Base * (1 + alpha * Z_Clima_Adverso
                                         - beta * Z_Superficie_Predominante
                                         + Ruido_Gaussiano)

    Donde:
        - Precio_Nacional_Base: Precio de referencia nacional (trigo, cebada, maiz)
        - Z_Clima_Adverso: Z-score ponderado de temperatura y precipitacion
          (z_temp_mean_C * 0.3 - z_precip_total_mm * 0.7)
        - Z_Superficie_Predominante: Z-score de la superficie cultivada del cereal
          predominante en la provincia
        - Ruido_Gaussiano: Término estocastico N(0, noise_std) para evitar
          determinismo total y forzar al modelo a aprender patrones robustos
        - alpha (default=0.02): Sensibilidad al clima adverso (2% por desviacion estandar)
        - beta (default=0.02): Sensibilidad a la oferta local (2% por desviacion estandar)

    Esta formulacion refleja la teoria economica agraria:
        - alpha > 0: Mayor adversidad climatica -> mayor presion al alza en precios
        - beta > 0: Mayor oferta local -> moderacion de precios (ley de oferta/demanda)

    Args:
        panel: DataFrame combinado.
        alpha: Peso del clima adverso (default=0.02).
        beta: Peso de superficie (default=0.02).
        noise_std: Desvio del ruido gaussiano (default=0.01). Se probo reducirlo a 0.005
            tras corregir el ruido posicional (ver _deterministic_row_noise) para ver si
            el Pearson en test mejoraba; no fue el caso (el gap train/test en H3 se
            mantuvo igual de grande, 0.84->0.22), lo que confirma que el problema no es la
            magnitud del ruido sino la dificultad de generalizacion ante el "Cisne Negro"
            de 2022. Se mantiene el valor original 0.01.
        random_seed: Semilla aleatoria (default=42).

    Returns:
        DataFrame con features y target calculado segun formula semi-sintetica.
    """
    panel = panel.copy()
    panel["month"] = panel["date"].dt.month

    panel = add_expanding_zscore(
        panel,
        ["province_name"],
        "temp_mean_C",
        "z_temp_mean_C",
    )
    panel = add_expanding_zscore(
        panel,
        ["province_name"],
        "precip_total_mm",
        "z_precip_total_mm",
    )
    panel = add_expanding_zscore(
        panel,
        ["province_name", "month"],
        "sup_cereal_predominante",
        "z_sup_cereal_predominante",
    )

    panel = add_expanding_mean(
        panel,
        ["province_name", "month"],
        "temp_mean_C",
        "temp_mean_hist",
    )
    panel["temp_anomalia"] = panel["temp_mean_C"] - panel["temp_mean_hist"]

    panel["z_clima_adverso"] = (panel["z_temp_mean_C"] * 0.3) - (
        panel["z_precip_total_mm"] * 0.7
    )

    panel["ruido_gauss"] = _deterministic_row_noise(
        panel["date"], panel["province_name"], panel["cereal_predominante"],
        noise_std, random_seed,
    )
    panel["precio_provincial_TARGET"] = panel["precio_nacional_base"] * (
        1.0
        + (alpha * panel["z_clima_adverso"])
        - (beta * panel["z_sup_cereal_predominante"])
        + panel["ruido_gauss"]
    )

    panel["precio_provincial"] = panel["precio_provincial_TARGET"]
    panel = panel.sort_values(["province_name", "cereal_predominante", "date"])

    # precio_provincial_lag_1/2/3 SI son features del modelo (a diferencia de
    # precio_provincial en si, que esta en la blacklist anti-leakage), asi
    # que necesitan alcanzar el frente de produccion. Se calculan sobre una
    # version de precio_provincial derivada del precio nacional extendido
    # (forward-fill, ver extend_prices_ffill mas arriba), no sobre la
    # columna real: donde hay dato MAPA real, ffill no cambia nada y el lag
    # sale identico; solo en los meses sin precio MAPA aun publicado aporta
    # un valor (el ultimo real conocido, ajustado por el clima/superficie de
    # ese mes) en vez de NaN. El target (precio_provincial_TARGET, arriba)
    # sigue usando exclusivamente el precio real, sin extender.
    panel_max_date = panel["date"].max()
    prices_long_real = (
        panel[["date", "cereal_predominante", "precio_nacional_base"]]
        .drop_duplicates()
    )
    ffill_base = extend_prices_ffill(
        prices_long_real, "cereal_predominante", "precio_nacional_base", panel_max_date,
    )[["date", "cereal_predominante", "precio_nacional_base"]].rename(
        columns={"precio_nacional_base": "_precio_nacional_base_ffill"}
    )
    panel = panel.merge(ffill_base, on=["date", "cereal_predominante"], how="left")
    panel["_precio_provincial_ffill"] = panel["_precio_nacional_base_ffill"] * (
        1.0
        + (alpha * panel["z_clima_adverso"])
        - (beta * panel["z_sup_cereal_predominante"])
        + panel["ruido_gauss"]
    )
    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=["_precio_provincial_ffill"],
        lags=[1, 2, 3],
    )
    for lag in (1, 2, 3):
        panel[f"precio_provincial_lag_{lag}"] = panel[f"_precio_provincial_ffill_lag_{lag}"]
    panel = panel.drop(
        columns=[
            "_precio_nacional_base_ffill",
            "_precio_provincial_ffill",
            "_precio_provincial_ffill_lag_1",
            "_precio_provincial_ffill_lag_2",
            "_precio_provincial_ffill_lag_3",
        ]
    )

    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=["temp_mean_C", "precip_total_mm"],
        lags=[1, 2],
    )

    idx_cols = [c for c in panel.columns if c.startswith("idx_")]
    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=idx_cols,
        lags=[MAPA_ADMIN_LAG, MAPA_ADMIN_LAG + 1],
    )
    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=["z_clima_adverso"],
        lags=[1, 2],
    )

    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=["wheat_intl_eur"],
        lags=[1, 2, 3],
    )
    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=["corn_intl_eur", "sup_cereal_predominante"],
        lags=[1],
    )
    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=["prepag2_paja cereales"],
        lags=[MAPA_ADMIN_LAG],
    )

    mapa_rescue_cols = [
        "prepag1_urea 46",
        "prepag1_dap",
        "prepag2_torta de girasol",
    ]
    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=mapa_rescue_cols,
        lags=[MAPA_ADMIN_LAG, MAPA_ADMIN_LAG + 1],
    )
    non_mapa_rescue_cols = [
        "temp_std_C",
        "precip_std_mm",
        "eur_usd",
    ]
    panel = add_lags(
        panel,
        group_cols=["province_name", "cereal_predominante"],
        value_cols=non_mapa_rescue_cols,
        lags=[1, 2],
    )

    surface_cols = ["total_sup_trigo", "total_sup_cebada"]
    surface_lag_cols = []
    for col in surface_cols:
        if col not in panel.columns:
            continue
        changed = (
            panel.sort_values(["province_name", "date"])  # deterministic diff
            .groupby("province_name", sort=False)[col]
            .diff()
            .abs()
            .gt(0)
            .any()
        )
        if changed:
            surface_lag_cols.append(col)
    if surface_lag_cols:
        panel = add_lags(
            panel,
            group_cols=["province_name"],
            value_cols=surface_lag_cols,
            lags=[1],
        )

    if "wheat_intl_eur" not in panel.columns:
        panel["wheat_intl_eur"] = np.nan
    if "corn_intl_eur" not in panel.columns:
        panel["corn_intl_eur"] = np.nan
    if "idx_fertilizantes" not in panel.columns:
        panel["idx_fertilizantes"] = np.nan

    panel["precio_intl_eur"] = np.select(
        [
            panel["cereal_predominante"].isin(["trigo", "cebada"]),
            panel["cereal_predominante"] == "maiz",
        ],
        [
            panel["wheat_intl_eur"],
            panel["corn_intl_eur"],
        ],
        default=np.nan,
    )
    # Calculate rolling features on unique monthly series to avoid mixing provinces
    for col in ["wheat_intl_eur", "corn_intl_eur"]:
        if col not in panel.columns:
            continue
        monthly = (
            panel[["date", col]]
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
        for w in [3]:
            rolled = (
                monthly[col]
                .shift(1)
                .rolling(window=w)
                .mean()
                .rename(f"{col}_ma{w}")
            )
            vol = (
                monthly[col]
                .shift(1)
                .rolling(window=w)
                .std(ddof=0)
                .rename(f"{col}_vol{w}")
            )
            monthly = monthly.assign(**{f"{col}_ma{w}": rolled, f"{col}_vol{w}": vol})
        panel = panel.merge(
            monthly[["date", f"{col}_ma3", f"{col}_vol3"]],
            on="date",
            how="left",
        )

    panel["ratio_prov_nacional"] = (
        panel["precio_provincial_lag_1"]
        / panel["precio_nacional_base"].replace(0, np.nan)
    )
    panel["spread_intl_prov"] = (
        panel["precio_intl_eur"] - panel["precio_provincial_lag_1"]
    )
    panel["ratio_presion_costes"] = (
        panel["precio_provincial_lag_1"]
        / panel["idx_fertilizantes"].replace(0, np.nan)
    )

    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12.0)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12.0)

    is_trigo_cebada = panel["cereal_predominante"].isin(["trigo", "cebada"])
    is_maiz = panel["cereal_predominante"] == "maiz"
    m = panel["month"]

    panel["fase_siembra"] = (
        (is_trigo_cebada & m.isin([10, 11, 12]))
        | (is_maiz & m.isin([3, 4]))
    ).astype(int)
    panel["fase_crecimiento"] = (
        (is_trigo_cebada & m.isin([1, 2, 3, 4]))
        | (is_maiz & m.isin([5, 6, 7]))
    ).astype(int)
    panel["fase_cosecha"] = (
        (is_trigo_cebada & m.isin([6, 7]))
        | (is_maiz & m.isin([9, 10]))
    ).astype(int)

    rename_map = {
        "province_name": "provincia",
    }
    for col in panel.columns:
        if col.startswith("temp_mean_C"):
            rename_map[col] = col.replace("temp_mean_C", "temp_provincial")
        if col.startswith("precip_total_mm"):
            rename_map[col] = col.replace("precip_total_mm", "precip_provincial")
        if col.startswith("temp_mean_hist"):
            rename_map[col] = col.replace("temp_mean_hist", "temp_provincial_hist")
    panel = panel.rename(columns=rename_map)
    panel = panel.sort_values(["provincia", "cereal_predominante", "date"])
    grouped = panel.groupby(["provincia", "cereal_predominante"], sort=False)
    panel["precio_provincial_TARGET_H1"] = grouped["precio_provincial"].shift(-1)
    panel["precio_provincial_TARGET_H2"] = grouped["precio_provincial"].shift(-2)
    panel["precio_provincial_TARGET_H3"] = grouped["precio_provincial"].shift(-3)
    return panel


def get_training_features(panel: pd.DataFrame) -> list[str]:
    """Genera la lista de columnas permitidas para entrenamiento."""
    identifiers = ["date", "provincia", "cereal_predominante"]
    targets = [
        "precio_provincial_TARGET_H1",
        "precio_provincial_TARGET_H2",
        "precio_provincial_TARGET_H3",
    ]
    static_cols = [
        "total_sup_trigo",
        "total_sup_cebada",
        "total_sup_maiz",
        "lat_centroide",
        "lon_centroide",
        "dist_puerto_min_km",
        "puerto_referencia",
        "provincia_id",
        "precio_vecinos_media_lag1",
        "presion_combustible_puerto",
        "precip_relative_shock",
        "month_sin",
        "month_cos",
        "fase_siembra",
        "fase_crecimiento",
        "fase_cosecha",
    ]
    allowed = set(identifiers + targets + static_cols)
    for col in panel.columns:
        if col.startswith("precio_nacional_base_lag_"):
            continue
        if any(token in col for token in ["_lag_", "_ma", "_vol"]):
            allowed.add(col)

    ordered = [c for c in panel.columns if c in allowed]
    extra = [c for c in allowed if c not in panel.columns]
    if extra:
        ordered.extend(extra)
    return ordered


def build_dataset(
    utils_dir: Path = UTILS_DIR,
    output_path: Path = PROCESSED / "dataset_espacial_final.csv",
    top_n: int = 30,
    price_cols: list[str] | None = None,
    alpha: float = 0.02,
    beta: float = 0.02,
    noise_std: float = 0.01,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Construye el dataset espacial final con features macro y provinciales.

    Args:
        utils_dir: Directorio de fuentes procesadas.
        output_path: Ruta de salida.
        top_n: Numero de combinaciones provincia-cereal.
        price_cols: Columnas de precios nacionales.
        alpha: Peso del clima adverso.
        beta: Peso de superficie.
        noise_std: Desvio del ruido gaussiano.
        random_seed: Semilla aleatoria.

    Returns:
        DataFrame final.
    """
    if price_cols is None:
        price_cols = ["trigo", "cebada", "maiz"]

    sources = load_all_sources(utils_dir, price_cols)
    panel = merge_spatial_macro(sources, top_n=top_n, price_cols=price_cols)
    panel = apply_feature_engineering(
        panel,
        alpha=alpha,
        beta=beta,
        noise_std=noise_std,
        random_seed=random_seed,
    )

    provincias = pd.read_csv(SPATIAL_MASTER_PATH)
    provincias["provincia_id"] = provincias["provincia_id"].astype(str).str.zfill(2)
    panel["provincia_id"] = panel["provincia"].astype(str)
    panel = panel.merge(
        provincias,
        left_on="provincia",
        right_on="nombre",
        how="left",
    )
    panel["provincia_id"] = panel["provincia_id_y"].fillna(panel["provincia_id_x"]).astype(str)
    panel = panel.drop(columns=["provincia_id_x", "provincia_id_y", "nombre"], errors="ignore")

    panel["presion_combustible_puerto"] = panel[f"idx_energia_lag_{MAPA_ADMIN_LAG}"] * panel["dist_puerto_min_km"]
    panel = add_spatial_price_lag(panel, threshold_km=SPATIAL_NEIGHBOR_KM)
    panel = add_spatial_price_lag(
        panel,
        threshold_km=SPATIAL_NEIGHBOR_KM,
        price_col="precip_provincial_lag_1",
        output_col="precip_vecinos_media_lag1",
    )
    panel["precip_relative_shock"] = panel["precip_provincial_lag_1"] / panel[
        "precip_vecinos_media_lag1"
    ].replace(0, np.nan)
    # Fill NaN with 1.0 (no shock) for provinces without spatial neighbors
    # (e.g., Leon-maiz, the only maize province in the panel).
    panel["precip_relative_shock"] = panel["precip_relative_shock"].fillna(1.0)

    # NOTA: precio_provincial_TARGET_H1/H2/H3 se excluyen deliberadamente de este
    # dropna. Son el objetivo a predecir (dependen de precios futuros), no una
    # feature de entrada: exigir que existan aqui recortaria el panel base a la
    # ultima fecha con target conocido, dejando fuera del todo las filas mas
    # recientes que si tienen inputs validos y son precisamente las que necesita
    # el modo produccion de predict_v1.py (inferencia sin precio futuro). El
    # filtrado por disponibilidad de target para train/test ya lo aplica
    # prepare_data.py por horizonte, que es el lugar correcto para esa regla.
    required_cols = [
        "precio_provincial_lag_3",
        "temp_provincial_lag_2",
        "precip_provincial_lag_2",
        "precio_nacional_base_ma6",
        "precio_nacional_base_vol6",
        "dist_puerto_min_km",
    ]
    required_cols = [c for c in required_cols if c in panel.columns]
    panel = panel.dropna(subset=required_cols).copy()

    # Validate that all top_n combinations are preserved after dropna
    final_combos = panel[["provincia", "cereal_predominante"]].drop_duplicates()
    if len(final_combos) < top_n:
        log.warning(
            "Pérdida de combinaciones tras dropna: %d de %d conservadas. "
            "Combinaciones finales: %s",
            len(final_combos),
            top_n,
            sorted(final_combos.itertuples(index=False, name=None)),
        )

    panel = panel.sort_values(["date", "provincia_id"]).reset_index(drop=True)
    panel = panel.drop(columns=["year", "month", "prov_key"], errors="ignore")
    preferred = ["date", "provincia", "cereal_predominante"]

    def _is_lag(col: str) -> bool:
        return "_lag_" in col or re.search(r"lag\d+$", col) is not None

    def _is_roll(col: str) -> bool:
        return re.search(r"_(ma|vol)\d+$", col) is not None

    def _is_future(col: str) -> bool:
        return any(token in col for token in ["TARGET_H1", "TARGET_H2", "TARGET_H3"])

    cols = list(panel.columns)
    current_cols = [
        c
        for c in cols
        if c not in preferred and not _is_lag(c) and not _is_roll(c) and not _is_future(c)
    ]
    lag_cols = [c for c in cols if _is_lag(c)]
    roll_cols = [c for c in cols if _is_roll(c)]
    future_cols = [c for c in cols if _is_future(c)]
    remaining = [
        c
        for c in cols
        if c not in preferred
        and c not in current_cols
        and c not in lag_cols
        and c not in roll_cols
        and c not in future_cols
    ]
    panel = panel[preferred + current_cols + lag_cols + roll_cols + future_cols + remaining]

    training_cols = get_training_features(panel)
    training_output = PROCESSED / "dataset_entrenamiento_final.csv"
    training_df = panel[training_cols].copy()

    allowed_tokens = ("_lag_", "_ma", "_vol")
    training_ids = {"date", "provincia", "cereal_predominante"}
    training_targets = {
        "precio_provincial_TARGET_H1",
        "precio_provincial_TARGET_H2",
        "precio_provincial_TARGET_H3",
    }
    training_static = {
        "total_sup_trigo",
        "total_sup_cebada",
        "total_sup_maiz",
        "lat_centroide",
        "lon_centroide",
        "dist_puerto_min_km",
        "puerto_referencia",
        "provincia_id",
        "precio_vecinos_media_lag1",
        "presion_combustible_puerto",
        "precip_relative_shock",
        "month_sin",
        "month_cos",
        "fase_siembra",
        "fase_crecimiento",
        "fase_cosecha",
    }
    contemporaneous = [
        c
        for c in training_cols
        if c not in training_ids
        and c not in training_targets
        and c not in training_static
        and not any(tok in c for tok in allowed_tokens)
    ]
    excluded_vars = [c for c in panel.columns if c not in training_cols]
    log.info("Variables contemporaneas detectadas en entrenamiento: %s", contemporaneous)
    log.info("Variables eliminadas por anti-leakage: %s", excluded_vars)
    save_dataset(training_df, training_output)
    date_min = pd.to_datetime(panel["date"]).min()
    date_max = pd.to_datetime(panel["date"]).max()
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    save_dataset(panel, output_path)
    log.info(
        "Dataset panel construido: %s  %s -> %s  -> %s",
        panel.shape,
        date_min.strftime("%Y-%m"),
        date_max.strftime("%Y-%m"),
        output_path.name,
    )
    return panel


if __name__ == "__main__":
    build_dataset()
