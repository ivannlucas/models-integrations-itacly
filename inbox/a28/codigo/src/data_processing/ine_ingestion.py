"""Utilities for INE external data ingestion (CU04)."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 2
DEFAULT_SLEEP_SECONDS = 1.5
DEFAULT_MAX_OPERATION_PAGES = 10
DEFAULT_INE_LANG = "ES"

LOGGER = logging.getLogger(__name__)


def build_base_url(ine_lang: str = DEFAULT_INE_LANG) -> str:
    return f"https://servicios.ine.es/wstempus/js/{ine_lang}"


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path.cwd()).resolve()
    candidates = [current] + list(current.parents)
    for candidate in candidates:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    return current


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def safe_slug(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def extract_name(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("Nombre", "nombre", "NOMBRE", "Desc", "Descripcion", "DESCRIPCION"):
        if key in item and item[key] not in (None, ""):
            return str(item[key])
    return ""


def extract_id(item: Any) -> Optional[Any]:
    if not isinstance(item, dict):
        return None
    for key in ("Id", "id", "ID", "Codigo", "codigo", "COD", "cod"):
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def is_json_response(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type.lower():
        return True
    body = response.text.strip()
    return body.startswith("{") or body.startswith("[")


def _parse_trust_env(value: Any) -> Optional[bool]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _build_trust_env_plan(trust_env: Optional[bool]) -> list[bool]:
    if trust_env is not None:
        return [trust_env]

    env_override = _parse_trust_env(os.getenv("INE_TRUST_ENV"))
    if env_override is not None:
        return [env_override]

    # Auto mode: prefer environment settings (proxy/certs), then retry direct.
    return [True, False]


def ine_get(
    path: str,
    params: Optional[dict[str, Any]] = None,
    *,
    base_url: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    trust_env: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    active_logger = logger or LOGGER
    endpoint = path.lstrip("/")
    url = f"{(base_url or build_base_url()).rstrip('/')}/{endpoint}"
    last_error: Optional[Exception] = None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-requests INE-ingestion/1.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Connection": "close",   # importante para evitar problemas de keep-alive
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    trust_env_plan = _build_trust_env_plan(trust_env)
    total_attempts = (retries + 1) * len(trust_env_plan)
    current_attempt = 0

    for trust_env_value in trust_env_plan:
        for mode_attempt in range(retries + 1):
            current_attempt += 1
            try:
                with requests.Session() as session:
                    session.trust_env = trust_env_value
                    response = session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=timeout,
                        allow_redirects=True,
                    )

                response.raise_for_status()

                if not is_json_response(response):
                    try:
                        return response.json()
                    except Exception:
                        snippet = response.text[:300].replace("\n", " ")
                        raise ValueError(
                            f"Non-JSON response for {url} | content-type={response.headers.get('content-type')} | body={snippet}"
                        )

                return response.json()

            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                active_logger.warning(
                    "INE GET failed (%s/%s) url=%s params=%s trust_env=%s error=%r",
                    current_attempt,
                    total_attempts,
                    url,
                    params,
                    trust_env_value,
                    exc,
                )

                if current_attempt < total_attempts:
                    # Reset backoff for each trust_env mode and retry sequence.
                    time.sleep(sleep_seconds * (2 ** mode_attempt))
                else:
                    raise RuntimeError(f"Could not query INE endpoint: {url}") from last_error

    raise RuntimeError(f"Could not query INE endpoint: {url}")


def search_operations(
    query: str,
    *,
    base_url: Optional[str] = None,
    max_pages: int = DEFAULT_MAX_OPERATION_PAGES,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> list[dict[str, Any]]:
    query_norm = normalize_text(query)
    results: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        payload = ine_get(
            "OPERACIONES_DISPONIBLES",
            params={"page": page, "det": 2},
            base_url=base_url,
            retries=retries,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            logger=logger,
        )
        if not isinstance(payload, list) or not payload:
            break

        if not query_norm:
            results.extend(item for item in payload if isinstance(item, dict))
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            name = extract_name(item)
            if query_norm in normalize_text(name):
                results.append(item)

    return results


def get_tables_for_operation(
    operation_id: Any,
    *,
    base_url: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> list[dict[str, Any]]:
    payload = ine_get(
        f"TABLAS_OPERACION/{operation_id}",
        params={"det": 2},
        base_url=base_url,
        retries=retries,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        logger=logger,
    )
    return payload if isinstance(payload, list) else []


def get_table_groups(
    table_id: Any,
    *,
    base_url: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> list[dict[str, Any]]:
    payload = ine_get(
        f"GRUPOS_TABLA/{table_id}",
        params={"det": 2},
        base_url=base_url,
        retries=retries,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        logger=logger,
    )
    return payload if isinstance(payload, list) else []


def get_table_values(
    table_id: Any,
    group_id: Any,
    *,
    base_url: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> list[dict[str, Any]]:
    payload = ine_get(
        f"VALORES_GRUPO/{table_id}/{group_id}",
        params={"det": 2},
        base_url=base_url,
        retries=retries,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        logger=logger,
    )
    return payload if isinstance(payload, list) else []


def download_table_data(
    table_id: Any,
    dataset: str,
    *,
    raw_dir: Path,
    params: Optional[dict[str, Any]] = None,
    force_download: bool = False,
    base_url: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> tuple[Any, dict[str, Any]]:
    active_logger = logger or LOGGER
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    slug = f"{safe_slug(dataset)}_table_{table_id}"
    raw_json_path = raw_dir / f"{slug}.json"
    raw_csv_path = raw_dir / f"{slug}.csv"
    metadata_path = raw_dir / f"{slug}.meta.json"

    endpoint = f"DATOS_TABLA/{table_id}"
    request_params = params or {"tip": "AM", "det": 2}

    if raw_json_path.exists() and not force_download:
        active_logger.info("Using local cache for %s (table_id=%s)", dataset, table_id)
        payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
        io_meta = {
            "downloaded": False,
            "raw_json_path": str(raw_json_path),
            "raw_csv_path": str(raw_csv_path),
            "metadata_path": str(metadata_path),
            "endpoint": endpoint,
            "params": request_params,
        }
        return payload, io_meta

    active_logger.info("Downloading %s (table_id=%s)", dataset, table_id)
    payload = ine_get(
        endpoint,
        params=request_params,
        base_url=base_url,
        retries=retries,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        logger=active_logger,
    )

    if not isinstance(payload, (list, dict)):
        raise ValueError(f"Unexpected format from DATOS_TABLA/{table_id}: {type(payload)}")

    raw_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        if isinstance(payload, list):
            pd.json_normalize(payload, sep="_").to_csv(raw_csv_path, index=False)
        else:
            pd.json_normalize([payload], sep="_").to_csv(raw_csv_path, index=False)
    except Exception as exc:  # pragma: no cover - defensive IO path
        active_logger.warning("Could not write raw CSV for table_id=%s: %s", table_id, exc)

    metadata = {
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "INE",
        "dataset": dataset,
        "table_id": table_id,
        "endpoint": endpoint,
        "params": request_params,
        "record_type": type(payload).__name__,
        "record_count": len(payload) if isinstance(payload, list) else 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    io_meta = {
        "downloaded": True,
        "raw_json_path": str(raw_json_path),
        "raw_csv_path": str(raw_csv_path),
        "metadata_path": str(metadata_path),
        "endpoint": endpoint,
        "params": request_params,
    }
    return payload, io_meta


def _get_first_value(point: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in point and point[key] not in (None, ""):
            return point[key]

    normalized = {normalize_text(k): v for k, v in point.items()}
    for key in keys:
        norm_key = normalize_text(key)
        value = normalized.get(norm_key)
        if value not in (None, ""):
            return value
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace("\xa0", "").replace(" ", "")
    if text in ("", "-", "..."):
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


def _parse_ine_date_and_frequency(point: dict[str, Any]) -> tuple[Optional[pd.Timestamp], Optional[str]]:
    year = _get_first_value(point, ("Anyo", "A\u00f1o", "anyo", "year"))
    period = _get_first_value(point, ("FK_Periodo", "Periodo", "periodo"))

    if year is not None and period is not None:
        year = int(str(year)[:4])
        period_clean = str(period).upper().strip().replace(" ", "")

        month_match = re.match(r"^M(\d{1,2})$", period_clean)
        if month_match:
            month = int(month_match.group(1))
            if 1 <= month <= 12:
                return pd.Timestamp(year=year, month=month, day=1), "M"

        quarter_match = re.match(r"^T([1-4])$", period_clean)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            month = (quarter - 1) * 3 + 1
            return pd.Timestamp(year=year, month=month, day=1), "Q"

        if period_clean.startswith("A"):
            return pd.Timestamp(year=year, month=1, day=1), "A"

    candidates = [
        _get_first_value(point, ("Periodo", "periodo")),
        _get_first_value(point, ("Fecha", "fecha")),
    ]
    for value in candidates:
        if value in (None, ""):
            continue

        text = str(value).strip()
        text_upper = text.upper().replace(" ", "")

        month_match = re.search(r"((?:19|20)\d{2})[-/]?M(\d{1,2})", text_upper)
        if month_match:
            year = int(month_match.group(1))
            month = int(month_match.group(2))
            if 1 <= month <= 12:
                return pd.Timestamp(year=year, month=month, day=1), "M"

        annual_match = re.search(r"((?:19|20)\d{2})$", text_upper)
        if annual_match:
            year = int(annual_match.group(1))
            return pd.Timestamp(year=year, month=1, day=1), "A"

        timestamp_match = re.search(r"/Date\((\d+)\)/", text)
        if timestamp_match:
            dt = pd.to_datetime(int(timestamp_match.group(1)), unit="ms", utc=True).tz_convert(None)
            return pd.Timestamp(dt.date()), None

        dt = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return pd.Timestamp(dt.year, dt.month, 1), None

    return None, None


def _iter_series_from_payload(payload: Any):
    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list):
        return

    for item in payload:
        if not isinstance(item, dict):
            continue

        for key in ("Data", "Datos", "data"):
            if key in item and isinstance(item[key], list):
                yield item, item[key]
                break
        else:
            if any(
                key in item
                for key in ("Valor", "valor", "Value", "Anyo", "A\u00f1o", "Periodo", "FK_Periodo", "Fecha")
            ):
                yield {"Nombre": item.get("Nombre") or item.get("COD") or "serie_unica"}, [item]


def normalize_ine_series(
    payload: Any,
    dataset: str,
    table_id: Any,
    *,
    source: str = "INE",
    notes: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for idx, (series_meta, observations) in enumerate(_iter_series_from_payload(payload), start=1):
        subseries = (
            series_meta.get("Nombre")
            or series_meta.get("nombre")
            or series_meta.get("COD")
            or f"series_{idx}"
        )
        unit = (
            series_meta.get("Unidad")
            or series_meta.get("NombreUnidad")
            or series_meta.get("unit")
        )
        series_note = (
            series_meta.get("Notas")
            or series_meta.get("Nota")
            or notes
        )

        for point in observations:
            if not isinstance(point, dict):
                continue

            value = _to_float(
                point.get("Valor")
                if "Valor" in point
                else point.get("valor", point.get("Value", point.get("value")))
            )

            date, frequency = _parse_ine_date_and_frequency(point)
            if date is None:
                continue

            rows.append(
                {
                    "date": pd.Timestamp(date.year, date.month, 1),
                    "source": source,
                    "dataset": dataset,
                    "subseries": str(subseries),
                    "value": value,
                    "frequency": frequency or "",
                    "unit": point.get("Unidad") or point.get("NombreUnidad") or unit or "",
                    "table_id": str(table_id),
                    "notes": point.get("Notas") or point.get("Nota") or series_note or "",
                }
            )

    columns = ["date", "source", "dataset", "subseries", "value", "frequency", "unit", "table_id", "notes"]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df = df.sort_values(["dataset", "subseries", "date"]).reset_index(drop=True)
    return df[columns]


def monthly_to_weekly_ffill(df_wide: pd.DataFrame) -> pd.DataFrame:
    if df_wide.empty:
        return df_wide.copy()

    value_cols = [col for col in df_wide.columns if col != "date"]
    if not value_cols:
        return df_wide.copy()

    base = df_wide.copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values("date").drop_duplicates("date", keep="last").set_index("date")

    weekly_index = pd.date_range(start=base.index.min(), end=base.index.max(), freq="W-MON")
    expanded = base.reindex(base.index.union(weekly_index)).sort_index()
    expanded[value_cols] = expanded[value_cols].ffill()

    return expanded.loc[weekly_index].reset_index().rename(columns={"index": "date"})


def build_default_dataset_config(include_optional_eip: bool = False) -> dict[str, dict[str, Any]]:
    config = {
        "IPI": {
            "operation_code": "IPI",  # lookup directo por código (evita OPERACIONES_DISPONIBLES)
            "operation_queries": ["indice de produccion industrial"],
            "operation_keywords": ["produccion", "industrial", "ipi"],
            "table_keywords": ["cnae", "10.1", "101", "carne", "alimentacion", "industria alimentaria"],
            "series_keywords": ["10.1", "101", "carne", "carnic", "alimentacion"],
            "target_frequency": "M",
            "wide_column": "ine_ipi_cnae101",
        },
        "IPRI": {
            "operation_code": "IPRI",  # lookup directo por código
            "operation_queries": ["indice de precios industriales"],
            "operation_keywords": ["precios", "industrial", "ipri"],
            "table_keywords": ["alimentacion", "cnae", "10", "carne", "industria alimentaria"],
            "series_keywords": ["10", "alimentacion", "carne", "carnic"],
            "target_frequency": "M",
            "wide_column": "ine_ipri_alimentacion",
        },
        "ICN": {
            "operation_code": "ICN",  # lookup directo por código
            "operation_queries": ["indices de cifra de negocios en la industria"],
            "operation_keywords": ["cifra", "negocios", "industria", "icn"],
            "table_keywords": ["division 10", "cnae", "10", "alimentacion", "carne"],
            "series_keywords": ["division 10", "10", "alimentacion", "carne", "carnic"],
            "target_frequency": "M",
            "wide_column": "ine_icn_div10",
        },
    }

    if include_optional_eip:
        config["EIP"] = {
            # Si conoces el código real de la operación en Tempus3, ponlo aquí.
            # Si no, se mantiene búsqueda por texto como fallback.
            # "operation_code": "EIP",  # <- solo si confirmas que existe así en la API
            "operation_queries": ["encuesta industrial de productos"],
            "operation_keywords": ["encuesta", "industrial", "productos", "eip", "eiap"],
            "table_keywords": ["carne", "carnicos", "alimentacion", "anual"],
            "series_keywords": ["carne", "carnicos", "alimentacion"],
            "target_frequency": "A",
            "wide_column": "ine_eip_carnicos",
        }

    return config


def build_default_stable_table_fallbacks(include_optional_eip: bool = False) -> dict[str, Optional[Any]]:
    fallbacks: dict[str, Optional[Any]] = {
        "IPI": None,
        "IPRI": None,
        "ICN": None,
    }
    if include_optional_eip:
        fallbacks["EIP"] = None
    return fallbacks


def _keyword_score(text: str, keywords: list[str]) -> int:
    text_norm = normalize_text(text)
    return sum(1 for keyword in keywords if normalize_text(keyword) in text_norm)


def _deduplicate_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = extract_id(item)
        if item_id is not None:
            dedup[str(item_id)] = item
    return list(dedup.values())


def _pick_best_item(
    items: list[dict[str, Any]],
    keywords: list[str],
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict[str, Any]]:
    active_logger = logger or LOGGER
    if not items:
        return None

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for item in items:
        item_name = extract_name(item)
        score = _keyword_score(item_name, keywords)
        scored.append((score, item_name, item))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, _, best_item = scored[0]

    if best_score == 0:
        active_logger.warning("No exact keyword match found. Using closest available option.")

    return best_item


def discover_operation(
    config: dict[str, Any],
    *,
    base_url: Optional[str] = None,
    max_pages: int = DEFAULT_MAX_OPERATION_PAGES,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    active_logger = logger or LOGGER
    candidates: list[dict[str, Any]] = []

    # 1) Vía preferente: código de operación fijo (evita OPERACIONES_DISPONIBLES)
    operation_code = config.get("operation_code")
    if operation_code:
        try:
            op = ine_get(
                f"OPERACION/{operation_code}",
                base_url=base_url,
                params={"det": 2},
                retries=retries,
                timeout=timeout,
                sleep_seconds=sleep_seconds,
                logger=active_logger,
            )

            # Normalizamos a lista de candidatos para mantener compatibilidad aguas abajo
            if isinstance(op, dict):
                candidates.append(op)
            elif isinstance(op, list):
                candidates.extend([item for item in op if isinstance(item, dict)])

            if candidates:
                candidates = _deduplicate_by_id(candidates)
                best = _pick_best_item(candidates, config.get("operation_keywords", []), logger=active_logger)
                active_logger.info("Operation resolved by code=%s", operation_code)
                return best, candidates

        except Exception as exc:  # pragma: no cover - external API path
            active_logger.warning(
                "Direct operation lookup failed operation_code=%s error=%s. Falling back to text search.",
                operation_code,
                exc,
            )

    # 2) Fallback: búsqueda por texto (tu comportamiento actual)
    for query in config.get("operation_queries", []):
        try:
            matches = search_operations(
                query,
                base_url=base_url,
                max_pages=max_pages,
                retries=retries,
                timeout=timeout,
                sleep_seconds=sleep_seconds,
                logger=active_logger,
            )
            candidates.extend(matches)
        except Exception as exc:  # pragma: no cover - external API path
            active_logger.warning("Could not search operations query=%s error=%s", query, exc)

    candidates = _deduplicate_by_id(candidates)
    if not candidates:
        return None, []

    best = _pick_best_item(candidates, config.get("operation_keywords", []), logger=active_logger)
    return best, candidates


def discover_table(
    operation_id: Any,
    config: dict[str, Any],
    *,
    base_url: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    active_logger = logger or LOGGER
    tables = get_tables_for_operation(
        operation_id,
        base_url=base_url,
        retries=retries,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        logger=active_logger,
    )
    if not tables:
        return None, []
    best = _pick_best_item(tables, config.get("table_keywords", []), logger=active_logger)
    return best, tables


def run_ine_ingestion_pipeline(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_download: bool = False,
    ine_lang: str = DEFAULT_INE_LANG,
    save_parquet: bool = True,
    include_optional_eip: bool = False,
    dataset_config: Optional[dict[str, dict[str, Any]]] = None,
    stable_table_fallbacks: Optional[dict[str, Optional[Any]]] = None,
    repo_root: Optional[Path] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_operation_pages: int = DEFAULT_MAX_OPERATION_PAGES,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    active_logger = logger or LOGGER

    if not active_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    repository_root = find_repo_root(repo_root)
    raw_ine_dir = repository_root / "data" / "raw" / "external" / "ine"
    processed_ine_dir = repository_root / "data" / "processed" / "external" / "ine"
    raw_ine_dir.mkdir(parents=True, exist_ok=True)
    processed_ine_dir.mkdir(parents=True, exist_ok=True)

    base_url = build_base_url(ine_lang)
    dataset_config = dataset_config or build_default_dataset_config(include_optional_eip=include_optional_eip)
    stable_table_fallbacks = stable_table_fallbacks or build_default_stable_table_fallbacks(
        include_optional_eip=include_optional_eip
    )

    download_registry: dict[str, dict[str, Any]] = {}
    all_long_frames: list[pd.DataFrame] = []
    wide_frames: list[pd.DataFrame] = []

    for dataset, config in dataset_config.items():
        active_logger.info("")
        active_logger.info("=" * 90)
        active_logger.info("Processing INE dataset: %s", dataset)

        operation, operation_candidates = discover_operation(
            config,
            base_url=base_url,
            max_pages=max_operation_pages,
            retries=retries,
            timeout=timeout_seconds,
            sleep_seconds=sleep_seconds,
            logger=active_logger,
        )
        if operation is None:
            active_logger.warning("No operation found for %s. Skipping dataset.", dataset)
            continue

        operation_id = extract_id(operation)
        operation_name = extract_name(operation)
        active_logger.info("Selected operation: %s (%s)", operation_name, operation_id)

        fallback_table_id = stable_table_fallbacks.get(dataset)
        if fallback_table_id is not None:
            table = {"Id": fallback_table_id, "Nombre": f"fallback_table_{fallback_table_id}"}
            table_candidates = [table]
            active_logger.info("Using stable fallback for %s table_id=%s", dataset, fallback_table_id)
        else:
            table, table_candidates = discover_table(
                operation_id,
                config,
                base_url=base_url,
                retries=retries,
                timeout=timeout_seconds,
                sleep_seconds=sleep_seconds,
                logger=active_logger,
            )

        if table is None:
            active_logger.warning("No table found for %s. Skipping dataset.", dataset)
            continue

        table_id = extract_id(table)
        table_name = extract_name(table)
        active_logger.info("Selected table: %s (%s)", table_name, table_id)

        try:
            groups = get_table_groups(
                table_id,
                base_url=base_url,
                retries=retries,
                timeout=timeout_seconds,
                sleep_seconds=sleep_seconds,
                logger=active_logger,
            )
            active_logger.info("Detected groups in table %s: %s", table_id, len(groups))
            if groups:
                first_group_id = extract_id(groups[0])
                if first_group_id is not None:
                    values = get_table_values(
                        table_id,
                        first_group_id,
                        base_url=base_url,
                        retries=retries,
                        timeout=timeout_seconds,
                        sleep_seconds=sleep_seconds,
                        logger=active_logger,
                    )
                    active_logger.info("Values in first group (%s): %s", first_group_id, len(values))
        except Exception as exc:  # pragma: no cover - external API path
            active_logger.warning("Could not inspect groups/values for table_id=%s error=%s", table_id, exc)

        payload, io_meta = download_table_data(
            table_id=table_id,
            dataset=dataset,
            raw_dir=raw_ine_dir,
            params={"tip": "AM", "det": 2},
            force_download=force_download,
            base_url=base_url,
            retries=retries,
            timeout=timeout_seconds,
            sleep_seconds=sleep_seconds,
            logger=active_logger,
        )

        df_long = normalize_ine_series(
            payload=payload,
            dataset=dataset,
            table_id=table_id,
            source="INE",
            notes=f"operation={operation_name}; table={table_name}",
            start_date=start_date,
            end_date=end_date,
        )
        if df_long.empty:
            active_logger.warning("No normalized data for %s.", dataset)
            continue

        series_keywords = config.get("series_keywords", [])
        if series_keywords:
            normalized_keywords = [normalize_text(keyword) for keyword in series_keywords]
            mask = df_long["subseries"].fillna("").map(normalize_text).apply(
                lambda text: any(keyword in text for keyword in normalized_keywords)
            )
        else:
            mask = pd.Series([True] * len(df_long), index=df_long.index)

        df_focus = df_long[mask].copy()
        if df_focus.empty:
            active_logger.warning(
                "No exact cnae/meat match for %s. Using closest available breakdown.",
                dataset,
            )
            best_subseries = (
                df_long.groupby("subseries", dropna=False)["value"]
                .count()
                .sort_values(ascending=False)
                .index[0]
            )
            df_focus = df_long[df_long["subseries"] == best_subseries].copy()

        target_freq = config.get("target_frequency", "M")
        df_focus_target = df_focus[df_focus["frequency"] == target_freq].copy()
        if df_focus_target.empty:
            active_logger.warning(
                "No frequency %s for %s after filters. Using available frequency.",
                target_freq,
                dataset,
            )
            df_focus_target = df_focus.copy()

        selected_subseries = (
            df_focus_target.groupby("subseries", dropna=False)["value"]
            .count()
            .sort_values(ascending=False)
            .index[0]
        )
        df_primary = df_focus_target[df_focus_target["subseries"] == selected_subseries][["date", "value"]].copy()
        wide_col = config["wide_column"]
        df_primary = (
            df_primary.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .rename(columns={"value": wide_col})
        )

        dataset_long_csv = processed_ine_dir / f"ine_{dataset.lower()}_long.csv"
        df_long.to_csv(dataset_long_csv, index=False)

        if save_parquet:
            dataset_long_parquet = processed_ine_dir / f"ine_{dataset.lower()}_long.parquet"
            try:
                df_long.to_parquet(dataset_long_parquet, index=False)
            except Exception as exc:  # pragma: no cover - optional dependency path
                active_logger.warning("Could not write parquet for %s: %s", dataset, exc)

        download_registry[dataset] = {
            "operation_id": str(operation_id),
            "operation_name": operation_name,
            "table_id": str(table_id),
            "table_name": table_name,
            "operation_candidates": len(operation_candidates),
            "table_candidates": len(table_candidates),
            "rows_long": int(len(df_long)),
            "rows_focus_target": int(len(df_focus_target)),
            "selected_subseries_for_wide": str(selected_subseries),
            "raw_json_path": io_meta["raw_json_path"],
            "raw_csv_path": io_meta["raw_csv_path"],
            "metadata_path": io_meta["metadata_path"],
        }

        all_long_frames.append(df_long)
        wide_frames.append(df_primary)

    if all_long_frames:
        ine_long_all = pd.concat(all_long_frames, ignore_index=True)
        ine_long_all = ine_long_all.sort_values(["date", "dataset", "subseries"]).reset_index(drop=True)
    else:
        ine_long_all = pd.DataFrame(
            columns=["date", "source", "dataset", "subseries", "value", "frequency", "unit", "table_id", "notes"]
        )

    if wide_frames:
        ine_wide = wide_frames[0]
        for frame in wide_frames[1:]:
            ine_wide = ine_wide.merge(frame, on="date", how="outer")
        ine_wide = ine_wide.sort_values("date").reset_index(drop=True)
    else:
        ine_wide = pd.DataFrame(columns=["date"])

    ine_weekly_aux = monthly_to_weekly_ffill(ine_wide)

    consolidated_long_csv = processed_ine_dir / "ine_consolidated_long.csv"
    consolidated_wide_csv = processed_ine_dir / "ine_consolidated_wide.csv"
    consolidated_weekly_csv = processed_ine_dir / "ine_consolidated_wide_weekly_ffill.csv"
    download_registry_json = processed_ine_dir / "ine_download_registry.json"

    ine_long_all.to_csv(consolidated_long_csv, index=False)
    ine_wide.to_csv(consolidated_wide_csv, index=False)
    ine_weekly_aux.to_csv(consolidated_weekly_csv, index=False)

    download_registry_json.write_text(
        json.dumps(download_registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if save_parquet:
        try:
            ine_long_all.to_parquet(processed_ine_dir / "ine_consolidated_long.parquet", index=False)
            ine_wide.to_parquet(processed_ine_dir / "ine_consolidated_wide.parquet", index=False)
        except Exception as exc:  # pragma: no cover - optional dependency path
            active_logger.warning("Could not write consolidated parquet files: %s", exc)

    paths = {
        "raw_ine_dir": str(raw_ine_dir),
        "processed_ine_dir": str(processed_ine_dir),
        "consolidated_long_csv": str(consolidated_long_csv),
        "consolidated_wide_csv": str(consolidated_wide_csv),
        "consolidated_weekly_csv": str(consolidated_weekly_csv),
        "download_registry_json": str(download_registry_json),
    }

    return {
        "ine_long_all": ine_long_all,
        "ine_wide": ine_wide,
        "ine_weekly_aux": ine_weekly_aux,
        "download_registry": download_registry,
        "paths": paths,
        "base_url": base_url,
        "repo_root": str(repository_root),
    }


__all__ = [
    "build_base_url",
    "find_repo_root",
    "normalize_text",
    "safe_slug",
    "extract_name",
    "extract_id",
    "ine_get",
    "search_operations",
    "get_tables_for_operation",
    "get_table_groups",
    "get_table_values",
    "download_table_data",
    "normalize_ine_series",
    "monthly_to_weekly_ffill",
    "build_default_dataset_config",
    "build_default_stable_table_fallbacks",
    "discover_operation",
    "discover_table",
    "run_ine_ingestion_pipeline",
]
