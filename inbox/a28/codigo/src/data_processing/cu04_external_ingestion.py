from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
from src.reproducibility.runtime import ensure_optional_dependency, repo_root

ensure_optional_dependency("requests", repo_root_path=repo_root())
ensure_optional_dependency("bs4", repo_root_path=repo_root())
ensure_optional_dependency("openpyxl", repo_root_path=repo_root())
import requests
from bs4 import BeautifulSoup
from src.utils import find_repo_root, to_repo_relative_path

LOGGER = logging.getLogger(__name__)

MAPA_SLAUGHTER_URL = (
    "https://www.mapa.gob.es/es/estadistica/temas/estadisticas-agrarias/"
    "ganaderia/encuestas-sacrificio-ganado"
)
MAPA_OM_URL = (
    "https://www.mapa.gob.es/es/alimentacion/temas/observatorio-cadena/"
    "cadenas-valor/sistema-de-precios-om"
)
MAPA_CONSUMPTION_URL = (
    "https://www.mapa.gob.es/es/alimentacion/temas/consumo-tendencias/"
    "panel-de-consumo-alimentario/series-anuales"
)
INE_CPI_76128_URL = "https://www.ine.es/jaxiT3/files/t/csv_bdsc/76128.csv"
EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def _portable_path(path: Path, repo_root: Path | None = None) -> str:
    active_repo_root = repo_root or find_repo_root(path.parent if path.is_absolute() else Path.cwd())
    return to_repo_relative_path(path, active_repo_root)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value).strip().replace("\xa0", "").replace(" ", "")
    if text in {"", "-", "...", "nan", "NaN"}:
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_any_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return pd.Timestamp(ts)
    if isinstance(value, (int, float)) and pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    if re.match(r"^\d{8}$", text):
        dt = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        return pd.Timestamp(dt) if pd.notna(dt) else pd.NaT

    if re.match(r"^\d{4}-\d{2}$", text):
        dt = pd.to_datetime(f"{text}-01", format="%Y-%m-%d", errors="coerce")
        return pd.Timestamp(dt) if pd.notna(dt) else pd.NaT

    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        dt = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        return pd.Timestamp(dt) if pd.notna(dt) else pd.NaT

    dt = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return pd.NaT
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.tz_convert(None)
    return pd.Timestamp(dt)


def _candidate_date_columns(columns: list[str]) -> list[str]:
    keywords = ["date", "fecha", "periodo", "period", "mes", "semana", "time"]
    out: list[str] = []
    for col in columns:
        norm = _normalize_text(col)
        if any(key in norm for key in keywords):
            out.append(col)
    return out


def _make_unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for raw in columns:
        base = str(raw).strip()
        count = seen.get(base, 0) + 1
        seen[base] = count
        unique.append(base if count == 1 else f"{base}__{count}")
    return unique


def try_parse_date_column(series: pd.Series, min_success_ratio: float = 0.30) -> pd.Series:
    parsed = series.map(parse_any_date)
    ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
    if ratio < min_success_ratio:
        return pd.Series([pd.NaT] * len(series), index=series.index)
    return parsed


def _parse_period_to_month(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return pd.Timestamp(value.year, value.month, 1)
    text = str(value).strip()
    if not text:
        return pd.NaT

    if isinstance(value, (int, float)) and not pd.isna(value):
        number = int(value)
        if 1900 <= number <= 2100:
            return pd.Timestamp(number, 1, 1)

    m = re.search(r"/Date\((\d+)\)/", text)
    if m:
        dt = pd.to_datetime(int(m.group(1)), unit="ms", utc=True).tz_convert(None)
        return pd.Timestamp(dt.year, dt.month, 1)

    norm = _normalize_text(text).replace(" ", "")
    for pattern in [r"^(\d{4})[-/](\d{1,2})$", r"^(\d{4})m(\d{1,2})$"]:
        m = re.match(pattern, norm)
        if m:
            y, mth = int(m.group(1)), int(m.group(2))
            if 1 <= mth <= 12:
                return pd.Timestamp(y, mth, 1)
    m = re.match(r"^(\d{1,2})[-/](\d{4})$", norm)
    if m:
        mth, y = int(m.group(1)), int(m.group(2))
        if 1 <= mth <= 12:
            return pd.Timestamp(y, mth, 1)
    m = re.match(r"^(\d{4})$", norm)
    if m:
        return pd.Timestamp(int(m.group(1)), 1, 1)

    dt = parse_any_date(text)
    if pd.notna(dt):
        return pd.Timestamp(dt.year, dt.month, 1)
    return pd.NaT


def _to_week_monday(values: pd.Series) -> pd.Series:
    ts = values.map(parse_any_date)
    return (ts - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.normalize()


def _to_base100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s
    base = float(valid.iloc[0])
    if base == 0:
        return s
    return s / base * 100.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _http_get(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    retries: int = 3,
    timeout: int = 60,
    backoff: float = 1.6,
    logger: Optional[logging.Logger] = None,
) -> requests.Response:
    active_logger = logger or LOGGER
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                raise
            sleep_seconds = backoff * (2 ** attempt)
            active_logger.warning("Retry %s/%s GET %s error=%s", attempt + 1, retries + 1, url, exc)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Could not request {url}: {last_error}")


def _download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    *,
    source: str,
    registry: list[dict[str, Any]],
    force_download: bool,
    retries: int,
    timeout: int,
    backoff: float,
    logger: Optional[logging.Logger] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force_download:
        registry.append(
            {
                "source": source,
                "url": url,
                "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": _sha256_file(output_path),
                "size_bytes": output_path.stat().st_size,
                "status": "cached",
                "path": str(output_path),
            }
        )
        return output_path

    response = _http_get(
        session,
        url,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        logger=logger,
    )
    output_path.write_bytes(response.content)
    registry.append(
        {
            "source": source,
            "url": url,
            "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": _sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "status": "downloaded",
            "path": str(output_path),
        }
    )
    return output_path


def _extract_xlsx_links(
    session: requests.Session,
    page_url: str,
    *,
    required_keywords: Optional[list[str]] = None,
    retries: int = 3,
    timeout: int = 60,
    backoff: float = 1.6,
    logger: Optional[logging.Logger] = None,
) -> list[dict[str, str]]:
    response = _http_get(
        session,
        page_url,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        logger=logger,
    )
    soup = BeautifulSoup(response.text, "html.parser")
    required_norm = [_normalize_text(x) for x in (required_keywords or [])]
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for a in soup.select("a[href]"):
        href = str(a.get("href", "")).strip()
        if not href:
            continue
        absolute = urljoin(page_url, href)
        if ".xlsx" not in urlparse(absolute).path.lower():
            continue
        text = " ".join(a.stripped_strings)
        blob = _normalize_text(f"{text} {absolute}")
        if required_norm and not any(k in blob for k in required_norm):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append({"url": absolute, "text": text})
    return links


def _find_keyword_column(columns: list[str], keywords: list[str]) -> Optional[str]:
    col_norm = {col: _normalize_text(col) for col in columns}
    for keyword in keywords:
        key = _normalize_text(keyword)
        for col, norm in col_norm.items():
            if key in norm:
                return col
    return None


def _numeric_columns(df: pd.DataFrame, min_ratio: float = 0.25) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        parsed = df[col].map(_parse_float)
        if parsed.notna().mean() >= min_ratio:
            out.append(col)
    return out


def _read_excel_book(path: Path, logger: Optional[logging.Logger] = None) -> dict[str, pd.DataFrame]:
    active_logger = logger or LOGGER
    try:
        return pd.read_excel(path, sheet_name=None)
    except Exception as exc:
        active_logger.warning("Could not read Excel %s error=%s", path, exc)
        return {}


def _local_xlsx_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([path for path in directory.rglob("*.xlsx") if path.is_file()])


def _drop_nat_and_assert_unique_date(df: pd.DataFrame, context: str) -> pd.DataFrame:
    if "date" not in df.columns:
        return df.copy()
    out = df.copy()
    out["date"] = out["date"].map(parse_any_date)
    out = out[out["date"].notna()].copy()
    dup = int(out.duplicated(subset=["date"]).sum())
    assert dup == 0, f"Duplicados por date detectados en {context}: {dup}"
    return out


def _flat_index_to_coords(flat_idx: int, sizes: list[int]) -> list[int]:
    coords: list[int] = []
    rem = int(flat_idx)
    for size in reversed(sizes):
        coords.append(rem % size)
        rem //= size
    return list(reversed(coords))


def _jsonstat_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict) or "dimension" not in payload or "value" not in payload:
        return pd.DataFrame()
    dim = payload.get("dimension", {})
    ids = payload.get("id", [])
    size = payload.get("size", [])
    if not ids or not size:
        return pd.DataFrame()

    pos_to_code: dict[str, list[str]] = {}
    pos_to_label: dict[str, list[str]] = {}
    for dim_id in ids:
        info = dim.get(dim_id, {})
        cat = info.get("category", {}) if isinstance(info, dict) else {}
        index_map = cat.get("index", {}) if isinstance(cat, dict) else {}
        label_map = cat.get("label", {}) if isinstance(cat, dict) else {}
        if isinstance(index_map, dict):
            codes = [None] * len(index_map)
            for code, pos in index_map.items():
                codes[int(pos)] = str(code)
        elif isinstance(index_map, list):
            codes = [str(x) for x in index_map]
        else:
            codes = [str(x) for x in label_map.keys()]
        labels = [str(label_map.get(code, code)) for code in codes]
        pos_to_code[str(dim_id)] = codes
        pos_to_label[str(dim_id)] = labels

    values = payload.get("value")
    if isinstance(values, dict):
        items = [(int(k), v) for k, v in values.items()]
    elif isinstance(values, list):
        items = list(enumerate(values))
    else:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for flat_idx, value in items:
        if value is None:
            continue
        coords = _flat_index_to_coords(int(flat_idx), [int(x) for x in size])
        row: dict[str, Any] = {"value": float(value)}
        for dim_id, coord in zip(ids, coords):
            did = str(dim_id)
            codes = pos_to_code.get(did, [])
            labels = pos_to_label.get(did, [])
            if coord >= len(codes):
                continue
            row[did] = codes[coord]
            row[f"{did}_label"] = labels[coord] if coord < len(labels) else codes[coord]
        rows.append(row)
    return pd.DataFrame(rows)


def ingest_mapa_slaughter(
    session: requests.Session,
    *,
    raw_dir: Path,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    force_download: bool,
    retries: int,
    timeout: int,
    backoff: float,
    species_keywords: list[str],
    registry: list[dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_logger = logger or LOGGER
    local_files = _local_xlsx_files(raw_dir / "mapa_sacrificio")
    if local_files and not force_download:
        active_logger.info("Using cached MAPA slaughter files=%s", len(local_files))
        file_paths = local_files
    else:
        try:
            links = _extract_xlsx_links(
                session,
                MAPA_SLAUGHTER_URL,
                retries=retries,
                timeout=timeout,
                backoff=backoff,
                logger=active_logger,
            )
        except Exception as exc:
            active_logger.warning("Could not refresh MAPA slaughter links: %s", exc)
            links = []
        active_logger.info("MAPA slaughter XLSX links=%s", len(links))

        file_paths = []
        for item in links:
            url = item["url"]
            filename = Path(urlparse(url).path).name or f"mapa_sacrificio_{len(file_paths)+1}.xlsx"
            out = raw_dir / "mapa_sacrificio" / filename
            try:
                file_paths.append(
                    _download_file(
                        session,
                        url,
                        out,
                        source="MAPA_SLAUGHTER",
                        registry=registry,
                        force_download=force_download,
                        retries=retries,
                        timeout=timeout,
                        backoff=backoff,
                        logger=active_logger,
                    )
                )
            except Exception as exc:
                active_logger.warning("Failed MAPA slaughter download url=%s error=%s", url, exc)

    records: list[dict[str, Any]] = []
    for path in file_paths:
        book = _read_excel_book(path, logger=active_logger)
        for sheet_name, df in book.items():
            if df.empty:
                continue
            frame = df.copy()
            frame.columns = [str(c).strip() for c in frame.columns]

            date_col = _find_keyword_column(frame.columns.tolist(), ["periodo", "mes", "fecha", "anyo", "ano"])
            if date_col is None:
                for col in frame.columns:
                    ratio = frame[col].map(_parse_period_to_month).notna().mean()
                    if ratio >= 0.35:
                        date_col = col
                        break
            if date_col is None:
                continue

            parsed_dates = frame[date_col].map(_parse_period_to_month)
            if parsed_dates.notna().sum() == 0:
                continue

            species_col = _find_keyword_column(frame.columns.tolist(), ["especie", "categoria", "ganado", "tipo"])
            num_cols = [c for c in _numeric_columns(frame, min_ratio=0.30) if c not in {date_col, species_col}]
            if not num_cols:
                continue

            for col in num_cols:
                col_norm = _normalize_text(col)
                if any(k in col_norm for k in ["peso", "canal", "ton", "tm"]):
                    metric, unit = "peso_canal_t", "t"
                elif any(k in col_norm for k in ["cabeza", "cabezas"]):
                    metric, unit = "cabezas", "cabezas"
                else:
                    metric, unit = "value", ""

                values = frame[col].map(_parse_float)
                if species_col:
                    species = frame[species_col].fillna("TOTAL").astype(str).str.strip()
                else:
                    species = pd.Series(["TOTAL"] * len(frame), index=frame.index)

                tmp = pd.DataFrame(
                    {
                        "date": parsed_dates,
                        "species": species,
                        "metric": metric,
                        "value": values,
                        "unit": unit,
                        "notes": f"file={path.name}; sheet={sheet_name}; col={col}",
                    }
                )
                tmp = tmp[tmp["date"].notna() & tmp["value"].notna()]
                tmp = tmp[(tmp["date"] >= start_month) & (tmp["date"] <= end_month)]
                records.extend(tmp.to_dict(orient="records"))

    long_cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]
    if not records:
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    raw = pd.DataFrame(records)
    raw["species_norm"] = raw["species"].map(_normalize_text)
    species_mask = raw["species_norm"].apply(lambda x: any(k in x for k in species_keywords))
    focus = raw[species_mask].copy() if species_mask.any() else raw.copy()

    metric_mask = focus["metric"].eq("peso_canal_t")
    base = focus[metric_mask].copy() if metric_mask.any() else focus.copy()
    supply_index = base.groupby("date", as_index=False)["value"].sum(min_count=1).sort_values("date")

    long_df = raw.copy()
    long_df["source"] = "MAPA"
    long_df["dataset"] = "SLAUGHTER_MAPA"
    long_df["subseries"] = long_df["species"].astype(str) + "|" + long_df["metric"].astype(str)
    long_df = long_df[long_cols]
    return long_df, supply_index


def ingest_eurostat_slaughter(
    session: requests.Session,
    *,
    dataset_id: str,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    retries: int,
    timeout: int,
    backoff: float,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_logger = logger or LOGGER
    endpoint = f"{EUROSTAT_BASE}/{dataset_id}"
    try:
        sample_resp = _http_get(
            session,
            endpoint,
            params={"geo": "ES", "time": str(max(start_month.year, end_month.year - 1))},
            retries=retries,
            timeout=timeout,
            backoff=backoff,
            logger=active_logger,
        )
        sample_payload = sample_resp.json()
        sample_dims = sample_payload.get("id", []) if isinstance(sample_payload, dict) else []
        active_logger.info("Eurostat sample dims for %s: %s", dataset_id, sample_dims)

        full_resp = _http_get(
            session,
            endpoint,
            params={"geo": "ES"},
            retries=retries,
            timeout=timeout,
            backoff=backoff,
            logger=active_logger,
        )
        payload = full_resp.json()
    except Exception as exc:
        active_logger.warning("Eurostat supply series unavailable, continuing without it: %s", exc)
        long_cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])
    df = _jsonstat_to_dataframe(payload)
    long_cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]
    if df.empty:
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    time_col = None
    for col in df.columns:
        if col == "value" or col.endswith("_label"):
            continue
        norm = _normalize_text(col)
        if "time" in norm or "period" in norm:
            time_col = col
            break
    if time_col is None:
        time_col = "time" if "time" in df.columns else None
    if time_col is None:
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    df["date"] = df[time_col].map(_parse_period_to_month)
    df = df[df["date"].notna()].copy()
    df = df[(df["date"] >= start_month) & (df["date"] <= end_month)]

    if "geo" in df.columns:
        df = df[df["geo"].astype(str).str.upper().eq("ES")].copy()

    if "unit" in df.columns:
        unit_upper = df["unit"].astype(str).str.upper().str.strip()
        available_units = sorted(set(unit_upper.dropna().tolist()))
        preferred_units = ["THS_T", "T", "THS_HD"]
        selected_unit = None
        for candidate in preferred_units:
            if candidate in available_units:
                selected_unit = candidate
                break
        if selected_unit is None and available_units:
            selected_unit = available_units[0]
        if selected_unit is not None:
            active_logger.info(
                "Eurostat unit selection dataset=%s available=%s selected=%s",
                dataset_id,
                available_units,
                selected_unit,
            )
            df = df[unit_upper.eq(selected_unit)].copy()

    keep_excluded = {"value", "date", time_col, "unit", "unit_label", "geo", "geo_label"}
    dims = [c for c in df.columns if c not in keep_excluded and not c.endswith("_label")]
    total_df = df
    for dim in dims:
        mask = total_df[dim].astype(str).str.upper().str.contains("TOTAL|TOT")
        if mask.any():
            total_df = total_df[mask].copy()

    supply_index = total_df.groupby("date", as_index=False)["value"].sum(min_count=1).sort_values("date")

    if dims:
        df["subseries"] = df[dims].astype(str).agg("|".join, axis=1)
    else:
        df["subseries"] = "EUROSTAT_TOTAL"

    out = pd.DataFrame(
        {
            "date": df["date"],
            "source": "EUROSTAT",
            "dataset": "SLAUGHTER_EUROSTAT",
            "subseries": df["subseries"],
            "value": pd.to_numeric(df["value"], errors="coerce"),
            "unit": df["unit"].astype(str) if "unit" in df.columns else "",
            "notes": f"dataset={dataset_id}; dims={'|'.join(sample_dims)}",
        }
    )
    out = out[out["value"].notna()]
    return out[long_cols], supply_index


def _infer_price_level(text: str) -> str:
    norm = _normalize_text(text)
    if "origen" in norm:
        return "origen"
    if "mayorista" in norm or "destino" in norm:
        return "mayorista"
    return "price"


def _infer_price_unit(text: str) -> str:
    norm = _normalize_text(text)
    if "eur/kg" in norm or "€/kg" in text.lower() or "euro/kg" in norm:
        return "EUR_KG"
    if "eur" in norm or "€" in text:
        return "EUR"
    return ""


def extract_mapa_prices_om(
    xlsx_path: Path,
    price_products_keywords: list[str],
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_logger = logger or LOGGER
    long_cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]

    try:
        books = pd.read_excel(xlsx_path, sheet_name=None)
    except Exception as exc:
        active_logger.warning("Could not read MAPA O-M xlsx=%s error=%s", xlsx_path, exc)
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    def _extract_from_frame(frame: pd.DataFrame, sheet_name: str) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        df = frame.copy()
        df.columns = _make_unique_columns([str(c).strip() for c in df.columns])

        date_candidates = _candidate_date_columns(df.columns.tolist())
        regex_like_date_cols = []
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty:
                continue
            regex_hits = sample.str.match(r"^\d{8}$|^\d{4}-\d{2}$|^\d{4}-\d{2}-\d{2}$", na=False).mean()
            if regex_hits >= 0.10:
                regex_like_date_cols.append(col)
        for col in regex_like_date_cols:
            if col not in date_candidates:
                date_candidates.append(col)
        if not date_candidates:
            return []

        best_col = None
        best_dates = None
        best_ratio = 0.0
        for col in date_candidates:
            parsed = try_parse_date_column(df[col], min_success_ratio=0.0)
            ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = col
                best_dates = parsed
        if best_col is None or best_dates is None or best_ratio < 0.10:
            return []

        product_col = _find_keyword_column(df.columns.tolist(), ["producto", "articulo", "denominacion"])

        price_col_keywords = ["precio", "€/kg", "eur", "pvp", "origen", "destino", "mayorista"]
        price_cols: list[str] = []
        for col in df.columns:
            if col == best_col or col == product_col:
                continue
            norm = _normalize_text(col)
            if any(_normalize_text(k) in norm for k in price_col_keywords):
                price_cols.append(col)
        if not price_cols:
            price_cols = [c for c in _numeric_columns(df, min_ratio=0.05) if c not in {best_col, product_col}]
        if not price_cols:
            return []

        rows: list[dict[str, Any]] = []
        dates = _to_week_monday(best_dates)
        for col in price_cols:
            values = df[col].map(_parse_float)
            if values.notna().sum() == 0:
                continue
            if product_col:
                products = df[product_col].fillna(col).astype(str).str.strip()
            else:
                products = pd.Series([str(col)] * len(df), index=df.index)
            level = _infer_price_level(col)
            unit = _infer_price_unit(col)
            tmp = pd.DataFrame(
                {
                    "date": dates,
                    "product": products,
                    "level": level,
                    "value": values,
                    "unit": unit,
                    "notes": f"file={xlsx_path.name}; sheet={sheet_name}; col={col}",
                }
            )
            tmp = tmp[tmp["date"].notna() & tmp["value"].notna()]
            rows.extend(tmp.to_dict(orient="records"))
        return rows

    all_rows: list[dict[str, Any]] = []
    for sheet_name, df in books.items():
        rows = _extract_from_frame(df, sheet_name)
        if rows:
            all_rows.extend(rows)
            continue

        # Fallback: retry parsing by scanning potential header rows.
        try:
            raw_sheet = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
        except Exception:
            raw_sheet = pd.DataFrame()
        if raw_sheet.empty:
            continue

        best_rows: list[dict[str, Any]] = []
        max_scan = min(20, max(0, len(raw_sheet) - 2))
        for header_idx in range(max_scan + 1):
            head_values = raw_sheet.iloc[header_idx].fillna("").astype(str).str.strip().tolist()
            trial = raw_sheet.iloc[header_idx + 1 :].copy()
            trial.columns = _make_unique_columns(head_values)
            parsed_rows = _extract_from_frame(trial, f"{sheet_name}#h{header_idx}")
            if len(parsed_rows) > len(best_rows):
                best_rows = parsed_rows
        if best_rows:
            all_rows.extend(best_rows)

    if not all_rows:
        active_logger.warning("No rows extracted from MAPA O-M file=%s", xlsx_path)
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    raw = pd.DataFrame(all_rows)
    raw["product_norm"] = raw["product"].map(_normalize_text)
    keyword_mask = raw["product_norm"].apply(lambda x: any(k in x for k in [kw.lower() for kw in price_products_keywords]))
    if keyword_mask.any():
        raw = raw[keyword_mask].copy()
    else:
        active_logger.warning("MAPA O-M: product keyword filter matched 0 rows; using all products")

    raw["series_id"] = raw["product"].astype(str).str.strip() + "|" + raw["level"].astype(str).str.strip()
    raw = raw.dropna(subset=["date", "value"]).copy()
    raw = (
        raw.groupby(["date", "series_id", "unit", "notes"], as_index=False)["value"]
        .mean()
        .sort_values(["date", "series_id"])
        .reset_index(drop=True)
    )

    weekly = raw.groupby("date", as_index=False)["value"].mean().sort_values("date")
    weekly["value"] = _to_base100(weekly["value"])

    long_df = pd.DataFrame(
        {
            "date": raw["date"],
            "source": "MAPA",
            "dataset": "PRICES_OM",
            "subseries": raw["series_id"],
            "value": pd.to_numeric(raw["value"], errors="coerce"),
            "unit": raw["unit"].replace("", "EUR_KG"),
            "notes": raw["notes"],
        }
    )
    long_df = long_df[long_df["value"].notna()]

    return long_df[long_cols], weekly[["date", "value"]]


def ingest_mapa_prices_om(
    session: requests.Session,
    *,
    raw_dir: Path,
    start_month: pd.Timestamp,
    end_raw: pd.Timestamp,
    force_download: bool,
    retries: int,
    timeout: int,
    backoff: float,
    price_keywords: list[str],
    registry: list[dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_logger = logger or LOGGER
    long_cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]
    local_files = _local_xlsx_files(raw_dir / "mapa_precios_om")
    if local_files and not force_download:
        latest_path = sorted(local_files, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        active_logger.info("Using cached MAPA O-M file=%s", latest_path.name)
    else:
        try:
            links = _extract_xlsx_links(
                session,
                MAPA_OM_URL,
                required_keywords=["origen", "mayorista"],
                retries=retries,
                timeout=timeout,
                backoff=backoff,
                logger=active_logger,
            )
            if not links:
                links = _extract_xlsx_links(
                    session,
                    MAPA_OM_URL,
                    retries=retries,
                    timeout=timeout,
                    backoff=backoff,
                    logger=active_logger,
                )
        except Exception as exc:
            active_logger.warning("Could not refresh MAPA O-M links: %s", exc)
            links = []
        if not links:
            return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

        def _recency_key(item: dict[str, str]) -> tuple[int, int, str]:
            blob = _normalize_text(item.get("url", "") + " " + item.get("text", ""))
            years = re.findall(r"(20\d{2})", blob)
            year = int(years[-1]) if years else 0
            months = re.findall(r"(?:m|mes)(\d{1,2})", blob)
            month = int(months[-1]) if months else 0
            return (year, month, blob)

        latest = sorted(links, key=_recency_key, reverse=True)[0]
        latest_url = latest["url"]
        latest_name = Path(urlparse(latest_url).path).name or "mapa_precios_om_latest.xlsx"
        latest_path = _download_file(
            session,
            latest_url,
            raw_dir / "mapa_precios_om" / latest_name,
            source="MAPA_PRICES_OM",
            registry=registry,
            force_download=force_download,
            retries=retries,
            timeout=timeout,
            backoff=backoff,
            logger=active_logger,
        )

    out_long, price_index = extract_mapa_prices_om(
        latest_path,
        price_products_keywords=price_keywords,
        logger=active_logger,
    )
    if out_long.empty or price_index.empty:
        active_logger.warning("MAPA O-M extraction returned empty for file=%s", latest_path)
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    out_long["date"] = _to_week_monday(out_long["date"])
    out_long["value"] = pd.to_numeric(out_long["value"], errors="coerce")
    out_long = out_long[(out_long["date"] >= start_month) & (out_long["date"] <= end_raw)]
    out_long = out_long[out_long["date"].notna() & out_long["value"].notna()]

    price_index = price_index.copy()
    price_index["date"] = _to_week_monday(price_index["date"])
    price_index["value"] = pd.to_numeric(price_index["value"], errors="coerce")
    price_index = price_index[(price_index["date"] >= start_month) & (price_index["date"] <= end_raw)]
    price_index = price_index[price_index["date"].notna() & price_index["value"].notna()]
    price_index = price_index.groupby("date", as_index=False)["value"].mean().sort_values("date")

    active_logger.info(
        "MAPA O-M parsed rows=%s unique_weeks=%s index_points=%s",
        len(out_long),
        out_long["date"].nunique(),
        len(price_index),
    )
    return out_long[long_cols], price_index[["date", "value"]]


def ingest_mapa_consumption_panel(
    session: requests.Session,
    *,
    raw_dir: Path,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    force_download: bool,
    retries: int,
    timeout: int,
    backoff: float,
    mapa_years: list[int],
    registry: list[dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_logger = logger or LOGGER
    long_cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]
    local_files = _local_xlsx_files(raw_dir / "mapa_consumo_panel")
    if local_files and not force_download:
        candidate_files = [
            path
            for path in local_files
            if any(str(year) in str(path) for year in mapa_years) or "misc" in str(path).lower()
        ]
        if not candidate_files:
            candidate_files = local_files

        preferred_files = []
        seen_years: set[str] = set()
        for path in sorted(candidate_files):
            blob = _normalize_text(str(path.name))
            year_match = re.findall(r"(20\d{2})", str(path))
            year_key = year_match[-1] if year_match else "misc"
            if year_key in seen_years:
                continue
            if any(token in blob for token in ["anual", "series-anuales", "mensuales-panel-consumo-hogares-canales"]):
                preferred_files.append(path)
                seen_years.add(year_key)
        files = preferred_files or candidate_files[: min(12, len(candidate_files))]
        active_logger.info("Using cached MAPA panel files=%s", len(files))
    else:
        try:
            links = _extract_xlsx_links(
                session,
                MAPA_CONSUMPTION_URL,
                retries=retries,
                timeout=timeout,
                backoff=backoff,
                logger=active_logger,
            )
        except Exception as exc:
            active_logger.warning("Could not refresh MAPA consumption links: %s", exc)
            links = []
        if not links:
            return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

        selected: list[dict[str, str]] = []
        for item in links:
            blob = _normalize_text(item["url"] + " " + item["text"])
            if "anual" not in blob:
                continue
            years = re.findall(r"(20\d{2})", blob)
            if years:
                if int(years[-1]) in mapa_years:
                    selected.append(item)
            else:
                selected.append(item)
        if not selected:
            selected = links[: min(6, len(links))]

        files = []
        for item in selected:
            url = item["url"]
            filename = Path(urlparse(url).path).name or f"mapa_panel_{len(files)+1}.xlsx"
            years = re.findall(r"(20\d{2})", item["url"] + " " + item["text"])
            year_dir = years[-1] if years else "misc"
            out = raw_dir / "mapa_consumo_panel" / year_dir / filename
            try:
                files.append(
                    _download_file(
                        session,
                        url,
                        out,
                        source="MAPA_PANEL_CONSUMO",
                        registry=registry,
                        force_download=force_download,
                        retries=retries,
                        timeout=timeout,
                        backoff=backoff,
                        logger=active_logger,
                    )
                )
            except Exception as exc:
                active_logger.warning("Failed MAPA panel download url=%s error=%s", url, exc)

    rows: list[dict[str, Any]] = []
    for path in files:
        for sheet, df in _read_excel_book(path, logger=active_logger).items():
            if df.empty:
                continue
            frame = df.copy()
            frame.columns = [str(c).strip() for c in frame.columns]
            text_col = _find_keyword_column(frame.columns.tolist(), ["producto", "categoria", "descripcion", "concepto"])
            if text_col is None:
                text_col = frame.columns[0]
            row_mask = frame[text_col].astype(str).map(_normalize_text).str.contains("carne|preparad|elaborad", na=False)
            if not row_mask.any():
                continue
            subset = frame[row_mask].copy()
            value_cols = [c for c in subset.columns if c != text_col]
            period_cols = []
            for col in value_cols:
                norm = _normalize_text(col)
                if re.search(r"20\d{2}", norm) or any(m in norm for m in ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]):
                    period_cols.append(col)
            if not period_cols:
                period_cols = value_cols
            melted = subset.melt(id_vars=[text_col], value_vars=period_cols, var_name="period", value_name="value_raw")
            melted["date"] = melted["period"].map(_parse_period_to_month)
            melted["value"] = melted["value_raw"].map(_parse_float)
            melted = melted[melted["date"].notna() & melted["value"].notna()]
            melted = melted[(melted["date"] >= start_month) & (melted["date"] <= end_month)]
            if melted.empty:
                continue
            melted["subseries"] = melted[text_col].astype(str).str.strip()
            melted["notes"] = f"file={path.name}; sheet={sheet}"
            rows.extend(melted[["date", "subseries", "value", "notes"]].to_dict(orient="records"))

    if not rows:
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])
    out = pd.DataFrame(rows)
    demand = out.groupby("date", as_index=False)["value"].sum(min_count=1).sort_values("date")
    out["source"] = "MAPA"
    out["dataset"] = "CONSUMPTION_MAPA_PANEL"
    out["unit"] = "kg"
    return out[long_cols], demand


def ingest_ine_cpi(
    session: requests.Session,
    *,
    raw_dir: Path,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    force_download: bool,
    retries: int,
    timeout: int,
    backoff: float,
    cpi_codes: list[str],
    registry: list[dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_logger = logger or LOGGER
    csv_path = _download_file(
        session,
        INE_CPI_76128_URL,
        raw_dir / "ine_ipc" / "76128.csv",
        source="INE_CPI",
        registry=registry,
        force_download=force_download,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        logger=active_logger,
    )
    df = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=enc, dtype=str)
            break
        except Exception:
            continue
    long_cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]
    if df is None or df.empty:
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])
    df.columns = [str(c).strip() for c in df.columns]
    eco_col = _find_keyword_column(df.columns.tolist(), ["ecoicop", "coicop", "clase"])
    type_col = _find_keyword_column(df.columns.tolist(), ["tipo de dato", "tipo"])
    period_col = _find_keyword_column(df.columns.tolist(), ["periodo", "period"])
    num_candidates = []
    for col in df.columns:
        if col in {eco_col, type_col, period_col}:
            continue
        num_candidates.append((df[col].map(_parse_float).notna().mean(), col))
    num_candidates.sort(reverse=True)
    value_col = num_candidates[0][1] if num_candidates else None
    if eco_col is None or period_col is None or value_col is None:
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    mask = pd.Series(False, index=df.index)
    eco = df[eco_col].astype(str)
    for code in cpi_codes:
        mask = mask | eco.str.contains(re.escape(code), na=False)
    df = df[mask].copy()
    if type_col is not None:
        idx_mask = df[type_col].astype(str).str.contains("indice", case=False, na=False)
        if idx_mask.any():
            df = df[idx_mask].copy()
    df["date"] = df[period_col].map(_parse_period_to_month)
    df["value"] = df[value_col].map(_parse_float)
    df = df[df["date"].notna() & df["value"].notna()]
    df = df[(df["date"] >= start_month) & (df["date"] <= end_month)]
    if df.empty:
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    df["subseries"] = df[eco_col].astype(str).str.extract(r"(\d+(?:\.\d+)+)", expand=False).fillna(df[eco_col].astype(str))
    demand = df.groupby("date", as_index=False)["value"].mean().sort_values("date")
    demand["value"] = _to_base100(demand["value"])
    out = pd.DataFrame(
        {
            "date": df["date"],
            "source": "INE",
            "dataset": "CPI",
            "subseries": df["subseries"],
            "value": pd.to_numeric(df["value"], errors="coerce"),
            "unit": "index",
            "notes": "table_id=76128",
        }
    )
    return out[long_cols], demand


def ingest_datacomex_optional(
    session: requests.Session,
    *,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    retries: int,
    timeout: int,
    backoff: float,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_logger = logger or LOGGER
    base = os.getenv("DATACOMEX_API_BASE")
    key = os.getenv("DATACOMEX_API_KEY")
    endpoint = os.getenv("DATACOMEX_API_ENDPOINT", "")
    long_cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]
    if not base or not key:
        active_logger.warning("DataComex credentials not configured; skipping trade pressure")
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    url = base.rstrip("/")
    if endpoint:
        url = f"{url}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {key}", "X-API-Key": key}
    params = {
        "start_date": str(start_month.date()),
        "end_date": str(end_month.date()),
        "geo": "ES",
        "hs_codes": "1601,1602,0201,0202,0203,0204,0205,0206",
    }
    try:
        payload = _http_get(
            session,
            url,
            params=params,
            headers=headers,
            retries=retries,
            timeout=timeout,
            backoff=backoff,
            logger=active_logger,
        ).json()
    except Exception as exc:
        active_logger.warning("DataComex request failed error=%s", exc)
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("data") or payload.get("results") or []
    else:
        records = []

    rows: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        date = _parse_period_to_month(item.get("date") or item.get("period") or item.get("time"))
        value = _parse_float(item.get("value") or item.get("trade_pressure") or item.get("importe") or item.get("valor"))
        if pd.isna(date) or value is None:
            continue
        rows.append(
            {
                "date": date,
                "source": "DATACOMEX",
                "dataset": "TRADE_PRESSURE",
                "subseries": str(item.get("hs_code") or item.get("code") or item.get("producto") or "trade"),
                "value": float(value),
                "unit": str(item.get("unit") or ""),
                "notes": "optional_datacomex_api",
            }
        )
    if not rows:
        return pd.DataFrame(columns=long_cols), pd.DataFrame(columns=["date", "value"])
    out = pd.DataFrame(rows)
    out = out[(out["date"] >= start_month) & (out["date"] <= end_month)]
    index = out.groupby("date", as_index=False)["value"].mean().sort_values("date")
    index["value"] = _to_base100(index["value"])
    return out[long_cols], index


def run_cu04_external_pipeline(
    *,
    start_date: str = "2004-01-01",
    end_date: str = "today",
    force_download: bool = False,
    raw_dir: Path = Path("data/raw/external"),
    proc_dir: Path = Path("data/processed/external/context"),
    registry_path: Optional[Path] = None,
    mapa_years: Optional[list[int]] = None,
    species_keywords: Optional[list[str]] = None,
    price_products_keywords: Optional[list[str]] = None,
    cpi_ecoicop_codes: Optional[list[str]] = None,
    eurostat_dataset: str = "apro_mt_pwgtm",
    retries: int = 3,
    timeout: int = 60,
    backoff: float = 1.6,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    active_logger = logger or LOGGER
    if not active_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    repo_root = find_repo_root(proc_dir.parent if proc_dir.is_absolute() else Path.cwd())
    registry_path = registry_path or (proc_dir / "download_registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else []
    if not isinstance(registry, list):
        registry = []

    start_raw = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(start_raw):
        raise ValueError(f"Invalid start_date: {start_date}")
    if end_date is None or str(end_date).strip().lower() == "today":
        end_raw = pd.Timestamp.utcnow().tz_localize(None).normalize()
    else:
        end_raw = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(end_raw):
        raise ValueError(f"Invalid end_date: {end_date}")
    start_month = pd.Timestamp(start_raw).to_period("M").to_timestamp(how="start")
    end_month = pd.Timestamp(end_raw).to_period("M").to_timestamp(how="start")

    years = mapa_years or list(range(2015, pd.Timestamp.utcnow().year + 1))
    species = species_keywords or ["bov", "porc", "ovin", "capr", "aviar", "cun"]
    price_kw = price_products_keywords or ["vacuno", "ternera", "porcino", "cerdo", "pollo", "pavo", "elaborados"]
    cpi_codes = cpi_ecoicop_codes or ["01.1.2.3", "01.1.2.5"]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) External-Context-Pipeline/1.0",
            "Accept": "*/*",
        }
    )

    all_long: list[pd.DataFrame] = []
    monthly: dict[str, pd.DataFrame] = {}
    weekly: dict[str, pd.DataFrame] = {}

    mapa_long, supply_mapa = ingest_mapa_slaughter(
        session,
        raw_dir=raw_dir,
        start_month=start_month,
        end_month=end_month,
        force_download=force_download,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        species_keywords=species,
        registry=registry,
        logger=active_logger,
    )
    if not mapa_long.empty:
        all_long.append(mapa_long)
    if not supply_mapa.empty:
        monthly["supply_index_mapa"] = supply_mapa

    euro_long, supply_euro = ingest_eurostat_slaughter(
        session,
        dataset_id=eurostat_dataset,
        start_month=start_month,
        end_month=end_month,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        logger=active_logger,
    )
    if not euro_long.empty:
        all_long.append(euro_long)
    if not supply_euro.empty:
        monthly["supply_index_eurostat"] = supply_euro

    om_long, price_weekly = ingest_mapa_prices_om(
        session,
        raw_dir=raw_dir,
        start_month=start_month,
        end_raw=pd.Timestamp(end_raw),
        force_download=force_download,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        price_keywords=price_kw,
        registry=registry,
        logger=active_logger,
    )
    if not om_long.empty:
        all_long.append(om_long)
    if not price_weekly.empty:
        weekly["purchase_price_index_om"] = price_weekly
    else:
        active_logger.warning(
            "No se pudo extraer PRICES_OM del XLSX. Usando fallback purchase_price_index=100 constante."
        )
        weekly_start = start_month - pd.to_timedelta(start_month.weekday(), unit="D")
        weekly_end = pd.Timestamp(end_raw) - pd.to_timedelta(pd.Timestamp(end_raw).weekday(), unit="D")
        fallback_idx = pd.date_range(start=weekly_start, end=weekly_end, freq="W-MON")
        fallback_weekly = pd.DataFrame({"date": fallback_idx, "value": 100.0})
        weekly["purchase_price_index_om"] = fallback_weekly
        all_long.append(
            pd.DataFrame(
                {
                    "date": fallback_idx,
                    "source": "MAPA",
                    "dataset": "PRICES_OM",
                    "subseries": "fallback_constant",
                    "value": 100.0,
                    "unit": "index",
                    "notes": "fallback_no_prices_extracted",
                }
            )
        )

    cons_long, demand_mapa = ingest_mapa_consumption_panel(
        session,
        raw_dir=raw_dir,
        start_month=start_month,
        end_month=end_month,
        force_download=force_download,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        mapa_years=years,
        registry=registry,
        logger=active_logger,
    )
    if not cons_long.empty:
        all_long.append(cons_long)
    if not demand_mapa.empty:
        monthly["demand_index_mapa_consumption"] = demand_mapa

    cpi_long, demand_cpi = ingest_ine_cpi(
        session,
        raw_dir=raw_dir,
        start_month=start_month,
        end_month=end_month,
        force_download=force_download,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        cpi_codes=cpi_codes,
        registry=registry,
        logger=active_logger,
    )
    if not cpi_long.empty:
        all_long.append(cpi_long)
    if not demand_cpi.empty:
        monthly["demand_index_cpi_processed_meat"] = demand_cpi

    trade_long, trade_index = ingest_datacomex_optional(
        session,
        start_month=start_month,
        end_month=end_month,
        retries=retries,
        timeout=timeout,
        backoff=backoff,
        logger=active_logger,
    )
    if not trade_long.empty:
        all_long.append(trade_long)
    if not trade_index.empty:
        monthly["trade_pressure_index"] = trade_index

    cols = ["date", "source", "dataset", "subseries", "value", "unit", "notes"]
    if all_long:
        external_long = pd.concat(all_long, ignore_index=True)
    else:
        external_long = pd.DataFrame(columns=cols)
    for col in cols:
        if col not in external_long.columns:
            external_long[col] = pd.NA
    external_long["date"] = external_long["date"].map(parse_any_date)
    external_long["date"] = external_long["date"].dt.to_period("M").dt.to_timestamp(how="start")
    external_long["value"] = pd.to_numeric(external_long["value"], errors="coerce")
    external_long = external_long[external_long["date"].notna()]
    external_long = external_long[(external_long["date"] >= start_month) & (external_long["date"] <= end_month)]
    external_long = (
        external_long.groupby(["date", "source", "dataset", "subseries"], as_index=False)
        .agg(value=("value", "mean"), unit=("unit", "first"), notes=("notes", "first"))
        [cols]
        .sort_values(["date", "source", "dataset", "subseries"])
        .reset_index(drop=True)
    )

    monthly_idx = pd.date_range(start=start_month, end=end_month, freq="MS")
    wide_monthly = pd.DataFrame({"date": monthly_idx})
    for name, df in monthly.items():
        series_df = df.copy()
        series_df["date"] = series_df["date"].map(parse_any_date)
        series_df["date"] = series_df["date"].dt.to_period("M").dt.to_timestamp(how="start")
        series_df["value"] = pd.to_numeric(series_df["value"], errors="coerce")
        ser = series_df.groupby("date", as_index=True)["value"].mean().sort_index().reindex(monthly_idx).ffill()
        wide_monthly[name] = ser.values
    if "purchase_price_index_om" in weekly:
        wk = weekly["purchase_price_index_om"].copy()
        wk["date"] = _to_week_monday(wk["date"])
        wk["value"] = pd.to_numeric(wk["value"], errors="coerce")
        wk = wk.dropna(subset=["date"]).groupby("date", as_index=False)["value"].mean()
        wk = _drop_nat_and_assert_unique_date(wk, context="purchase_price_index_om_monthly_resample")
        m_price = wk.set_index("date")["value"].resample("MS").mean().reindex(monthly_idx).ffill()
        wide_monthly["purchase_price_index_om"] = m_price.values

    assert pd.notna(start_month), "start_month invalido para construir indice semanal"
    assert pd.notna(end_raw), "end_raw invalido para construir indice semanal"
    weekly_start = start_month - pd.to_timedelta(start_month.weekday(), unit="D")
    weekly_end = pd.Timestamp(end_raw) - pd.to_timedelta(pd.Timestamp(end_raw).weekday(), unit="D")
    weekly_idx = pd.date_range(start=weekly_start, end=weekly_end, freq="W-MON")
    wide_weekly = pd.DataFrame({"date": weekly_idx})
    for col in [c for c in wide_monthly.columns if c != "date"]:
        m = wide_monthly.set_index("date")[col]
        w = m.reindex(m.index.union(weekly_idx)).sort_index().ffill().reindex(weekly_idx)
        wide_weekly[col] = w.values
    if "purchase_price_index_om" in weekly:
        wk = weekly["purchase_price_index_om"].copy()
        wk["date"] = _to_week_monday(wk["date"])
        wk["value"] = pd.to_numeric(wk["value"], errors="coerce")
        wk = wk.dropna(subset=["date"]).groupby("date", as_index=False)["value"].mean()
        wk = _drop_nat_and_assert_unique_date(wk, context="purchase_price_index_om_weekly_align")
        ser = wk.set_index("date")["value"].sort_index().reindex(weekly_idx).ffill()
        wide_weekly["purchase_price_index_om"] = ser.values

    context = pd.DataFrame({"date": wide_weekly["date"]})
    supply = None
    if "supply_index_eurostat" in wide_weekly.columns:
        supply = wide_weekly["supply_index_eurostat"]
    if "supply_index_mapa" in wide_weekly.columns:
        supply = wide_weekly["supply_index_mapa"] if supply is None else supply.combine_first(wide_weekly["supply_index_mapa"])
    context["supply_index"] = _to_base100(supply) if supply is not None else pd.NA
    context["purchase_price_index"] = (
        _to_base100(wide_weekly["purchase_price_index_om"]) if "purchase_price_index_om" in wide_weekly.columns else pd.NA
    )
    demand = None
    if "demand_index_cpi_processed_meat" in wide_weekly.columns:
        demand = wide_weekly["demand_index_cpi_processed_meat"]
    if "demand_index_mapa_consumption" in wide_weekly.columns:
        demand = wide_weekly["demand_index_mapa_consumption"] if demand is None else demand.combine_first(wide_weekly["demand_index_mapa_consumption"])
    context["demand_index"] = _to_base100(demand) if demand is not None else pd.NA
    if "trade_pressure_index" in wide_weekly.columns:
        context["trade_pressure"] = _to_base100(wide_weekly["trade_pressure_index"])
    context = _drop_nat_and_assert_unique_date(context, context="context_weekly_for_simulation")

    long_csv = proc_dir / "external_long.csv"
    long_parquet = proc_dir / "external_long.parquet"
    monthly_csv = proc_dir / "external_wide_monthly.csv"
    monthly_parquet = proc_dir / "external_wide_monthly.parquet"
    weekly_csv = proc_dir / "external_wide_weekly.csv"
    weekly_parquet = proc_dir / "external_wide_weekly.parquet"
    context_csv = proc_dir / "context_weekly_for_simulation.csv"
    context_parquet = proc_dir / "context_weekly_for_simulation.parquet"

    external_long.to_csv(long_csv, index=False)
    wide_monthly.to_csv(monthly_csv, index=False)
    wide_weekly.to_csv(weekly_csv, index=False)
    context.to_csv(context_csv, index=False)
    try:
        external_long.to_parquet(long_parquet, index=False)
        wide_monthly.to_parquet(monthly_parquet, index=False)
        wide_weekly.to_parquet(weekly_parquet, index=False)
        context.to_parquet(context_parquet, index=False)
    except Exception as exc:
        active_logger.warning("Could not write parquet outputs: %s", exc)

    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = (
        external_long.groupby(["dataset", "subseries"], as_index=False)
        .agg(n_points=("value", "count"), min_date=("date", "min"), max_date=("date", "max"))
        .sort_values(["dataset", "subseries"])
    )
    null_pct = (wide_weekly.isna().mean() * 100).round(2).reset_index().rename(columns={"index": "column", 0: "null_pct"})
    context_null_pct = (context.isna().mean() * 100).round(2).reset_index().rename(columns={"index": "column", 0: "null_pct"})
    context_dup_count = int(context.duplicated(subset=["date"]).sum())
    context_range = {
        "min_date": str(context["date"].min()) if not context.empty else None,
        "max_date": str(context["date"].max()) if not context.empty else None,
        "rows": int(len(context)),
    }
    dup_count = int(wide_weekly.melt(id_vars=["date"], var_name="variable", value_name="value").duplicated(["date", "variable"]).sum())

    has_supply = any(col in wide_monthly.columns and wide_monthly[col].notna().any() for col in ["supply_index_eurostat", "supply_index_mapa"])
    has_price = "purchase_price_index_om" in wide_weekly.columns and wide_weekly["purchase_price_index_om"].notna().any()
    has_demand = "demand_index_cpi_processed_meat" in wide_monthly.columns and wide_monthly["demand_index_cpi_processed_meat"].notna().any()

    assert dup_count == 0, "Duplicados por (date, variable) detectados en wide semanal."
    assert has_supply, "No hay supply_index mensual (Eurostat o MAPA)."
    assert has_price, "No hay price_index semanal (MAPA O-M)."
    assert has_demand, "No hay demand_index mensual (INE CPI)."
    assert context_dup_count == 0, "Duplicados por date detectados en context_weekly_for_simulation."
    assert "supply_index" in context.columns and context["supply_index"].notna().any(), "No existe supply_index usable en contexto semanal."
    assert "purchase_price_index" in context.columns and context["purchase_price_index"].notna().any(), "No existe purchase_price_index usable en contexto semanal."
    assert "demand_index" in context.columns and context["demand_index"].notna().any(), "No existe demand_index usable en contexto semanal."

    active_logger.info("Context range=%s", context_range)
    active_logger.info("Context null%% by column:\\n%s", context_null_pct.to_string(index=False))

    result = {
        "external_long": external_long,
        "external_wide_monthly": wide_monthly,
        "external_wide_weekly": wide_weekly,
        "context_weekly_for_simulation": context,
        "download_registry": registry,
        "summary_points": summary,
        "null_pct_weekly": null_pct,
        "context_null_pct": context_null_pct,
        "context_range": context_range,
        "context_dup_count": context_dup_count,
        "dup_count_weekly": dup_count,
        "paths": {
            "long_csv": _portable_path(long_csv, repo_root),
            "long_parquet": _portable_path(long_parquet, repo_root),
            "monthly_csv": _portable_path(monthly_csv, repo_root),
            "monthly_parquet": _portable_path(monthly_parquet, repo_root),
            "weekly_csv": _portable_path(weekly_csv, repo_root),
            "weekly_parquet": _portable_path(weekly_parquet, repo_root),
            "context_csv": _portable_path(context_csv, repo_root),
            "context_parquet": _portable_path(context_parquet, repo_root),
            "registry_json": _portable_path(registry_path, repo_root),
        },
    }
    result["cu04_external_long"] = external_long
    result["cu04_external_wide_monthly"] = wide_monthly
    result["cu04_external_wide_weekly"] = wide_weekly
    result["cu04_context_weekly_for_simulation"] = context
    return result

run_external_context_pipeline = run_cu04_external_pipeline

__all__ = ["run_cu04_external_pipeline", "run_external_context_pipeline"]
