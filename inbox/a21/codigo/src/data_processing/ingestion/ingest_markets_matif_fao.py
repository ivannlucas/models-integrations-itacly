"""Ingesta de señales de mercados internacionales (MATIF, CBOT, FAO).

Descarga precios semanales de los principales mercados de futuros de cereales
y el índice de precios FAO, y los consolida en un único CSV semanal alineado
aplicando fallback a yahooquery si yfinance devuelve pocos datos.

Entradas (APIs externas):
  - yfinance / yahooquery: futuros de trigo MATIF (EBM=F) y maíz CBOT (ZC=F)
  - FAO FPMA: índice mensual de precios de cereales (descargado como Excel/CSV)
  - config/config.yaml (claves: paths.data_external, weather.start_date/end_date)

Salidas:
  - data/external/mercados_internacionales.csv
    (columnas: week_start, EBM_EUR, ZC_USD, Cereals, Cereals Price Index;
     una fila por semana ISO, inicio en lunes)

Uso:
  python src/data_processing/ingestion/ingest_markets_matif_fao.py
  # o desde código:
  from scripts.ingest_markets_matif_fao import MarketIngestor
  MarketIngestor(config_path='config/config.yaml').run()
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yaml

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class MarketIngestor:
    config_path: str = str(PROJECT_ROOT / "config" / "config.yaml")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    data_external: str = str(PROJECT_ROOT / "data" / "external")

    def __post_init__(self):
        cfg_path = Path(self.config_path)
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path
        self.config_path = str(cfg_path)
        self._load_config()

    def _load_config(self) -> None:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = {}
        wcfg = cfg.get("weather", {})
        start = wcfg.get("start_date", "2018-01-01")
        end = wcfg.get("end_date", "today")
        if isinstance(start, str):
            start = pd.to_datetime(start).date()
        if isinstance(end, str) and str(end).lower() == "today":
            end = date.today()
        else:
            end = pd.to_datetime(end).date()

        self.start_date = self.start_date or start
        self.end_date = self.end_date or end

        paths = cfg.get("paths", {})
        data_external = Path(paths.get("data_external", self.data_external))
        if not data_external.is_absolute():
            data_external = PROJECT_ROOT / data_external
        self.data_external = str(data_external)

    def _fetch_yahoo(self, ticker: str) -> pd.DataFrame:
        """Fetch daily OHLC from Yahoo via yfinance and return DataFrame with 'date' and 'Close'.

        Returns daily DataFrame indexed by date with 'Close' column.
        """
        # Prefer yahooquery for downloads (more reliable for EU tickers), fallback to yfinance
        start_str = pd.to_datetime(self.start_date).strftime("%Y-%m-%d")
        end_str = (pd.to_datetime(self.end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("Downloading %s from Yahoo sources (%s - %s)", ticker, start_str, pd.to_datetime(end_str).date())

        df = None
        # try yahooquery first (non-blocking) to get a nicer structure; but prefer yfinance.history
        try:
            from yahooquery import Ticker as YQTicker
            yq = YQTicker(ticker)
            hist = yq.history(start=start_str, end=end_str)
            if isinstance(hist, dict):
                hist = hist.get(ticker, hist)
            df = pd.DataFrame(hist)
        except Exception:
            logger.debug("yahooquery not available/failed for %s; will try yfinance", ticker)

        # Use yfinance.Ticker.history as the authoritative downloader (avoids yf.download quirks)
        if df is None or getattr(df, 'empty', True):
            try:
                import yfinance as yf
                tk = yf.Ticker(ticker)
                df = tk.history(start=start_str, end=end_str, interval="1d", auto_adjust=False)
            except Exception as e:
                logger.debug("yfinance.Ticker.history failed for %s: %s", ticker, e)

        if df is None or getattr(df, 'empty', True):
            logger.warning("No data returned for %s", ticker)
            return pd.DataFrame(columns=["date", "Close"])

        # normalize dataframe
        # If the DataFrame uses a DatetimeIndex (common from yfinance), reset it to a column
        # If the DataFrame uses a DatetimeIndex (common from yfinance), reset it to a column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # first column after reset is the datetime index
            df = df.rename(columns={df.columns[0]: "date"})
        # If the DataFrame uses a MultiIndex (yahooquery often returns symbol,date), reset to get date
        elif isinstance(df.index, pd.MultiIndex):
            try:
                df = df.reset_index()
            except Exception:
                df = df.copy()

        # make column name detection case-insensitive
        cols_lower = {c.lower(): c for c in df.columns}

        # find date column
        if "date" in cols_lower:
            date_col = cols_lower["date"]
        elif "date_dt" in cols_lower:
            date_col = cols_lower["date_dt"]
        elif df.columns.size > 0:
            date_col = df.columns[0]
        else:
            raise RuntimeError(f"Unable to determine date column in Yahoo data for {ticker}")

        # find close column (prioritize exact 'Close')
        close_col = None
        if "close" in cols_lower:
            close_col = cols_lower["close"]
        else:
            for c in df.columns:
                if "close" in c.lower():
                    close_col = c
                    break

        # ensure standardized columns
        df = df.rename(columns={date_col: "date"})
        if close_col is not None:
            df = df.rename(columns={close_col: "Close"})
        else:
            df["Close"] = np.nan

        # parse date robustly (handle tz-aware and naive timestamps)
        try:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.date
        except Exception:
            try:
                df["date"] = pd.to_datetime(df["date"])
                if getattr(df["date"].dt, "tz", None) is not None:
                    df["date"] = df["date"].dt.tz_localize(None)
                df["date"] = df["date"].dt.date
            except Exception:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        df["Close"] = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce")

        # If the ticker returns too few observations, warn but return what we have
        try:
            n_obs = int(df["Close"].count())
            date_min = pd.to_datetime(df["date"]).min()
            date_max = pd.to_datetime(df["date"]).max()
            span_days = (pd.to_datetime(date_max) - pd.to_datetime(date_min)).days if pd.notna(date_min) and pd.notna(date_max) else 0
        except Exception:
            n_obs = 0
            span_days = 0

        # If the ticker provides insufficient historical rows, treat it as unavailable so caller can try fallbacks
        if n_obs < 100 or span_days < 365:
            logger.warning("⚠️ Ticker %s returned insufficient history (rows=%s, span_days=%s); marking as insufficient.", ticker, n_obs, span_days)
            return pd.DataFrame(columns=["date", "Close"])  # signal caller to try alternative

        return df

    def _fetch_yahoo_try_list(self, tickers: list[str], label: str | None = None, min_rows: int = 100) -> pd.DataFrame:
        """Try multiple tickers in order until one returns a reasonable series.

        Logs fallbacks with warnings but ultimately returns the best available (even partial).
        """
        last_df = pd.DataFrame(columns=["date", "Close"])
        prev = None
        for i, t in enumerate(tickers):
            try:
                df = self._fetch_yahoo(t)
            except Exception as e:
                logger.warning("⚠️ Error descargando %s: %s", t, e)
                df = pd.DataFrame(columns=["date", "Close"])

            # if this ticker returned nothing, and there is a next one, log fallback
            if (df is None or df.empty) and i < len(tickers) - 1:
                logger.warning("⚠️ Falló Ticker %s, intentando Ticker %s...", t, tickers[i + 1])
                last_df = df
                continue

            # if df has reasonable history (>= min_rows) accept it
            try:
                if int(df["Close"].count()) >= int(min_rows):
                    return df
            except Exception:
                pass

            # keep as candidate and continue trying fallbacks
            last_df = df

        # If none of the tickers met min_rows, do NOT return a single-row/partial series; return empty so caller can handle
        try:
            n_obs = int(last_df["Close"].count())
        except Exception:
            n_obs = 0
        if n_obs == 0:
            logger.warning("⚠️ Ningún ticker de la lista %s devolvió datos históricos suficientes para %s.", tickers, label or "ticker group")
            return pd.DataFrame(columns=["date", "Close"])
        else:
            logger.warning("⚠️ Se obtuvo solo una serie parcial (%d filas) de la lista %s para %s; marcando como insuficiente.", n_obs, tickers, label or "ticker group")
            return pd.DataFrame(columns=["date", "Close"])  # treat as unavailable

    def _fetch_fao(self) -> pd.DataFrame:
        """Download FAO Food Price Index CSV and return monthly 'Cereals' series.

        This function tries a few well-known FAO CSV endpoints; if none succeed
        it will attempt to find a CSV link on the FAO page.
        The returned DataFrame has columns ['date','Cereals'] with monthly dates.
        """
        page_url = "https://www.fao.org/worldfoodsituation/foodpricesindex/en/"
        preferred_excel = "https://www.fao.org/fileadmin/templates/worldfood/Files/food_prices_index/Food_Price_Index.xlsx"

        session = requests.Session()
        df = None

        # 1) Try direct Excel URL with User-Agent to avoid bot blocking
        from io import BytesIO
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            logger.info("Attempting to read FAO Excel: %s", preferred_excel)
            r = session.get(preferred_excel, timeout=30, headers=headers)
            if r.status_code == 200:
                df_x = pd.read_excel(BytesIO(r.content), skiprows=2)
                df = df_x
            else:
                logger.debug("FAO Excel returned status %s", r.status_code)
        except Exception:
            logger.debug("Direct FAO Excel fetch failed; will try discovery")

        # 2) Discover .xlsx link on FAO page and try
        if df is None:
            try:
                r = session.get(page_url, timeout=30, headers=headers)
                if r.status_code == 200:
                    import re

                    # look for a link that mentions Food Price Index and ends with xlsx
                    m = re.search(r'href=["\']([^"\']*Food[_ ]?Price[_ ]?Index[^"\']*\.xlsx)["\']', r.text, flags=re.I)
                    if not m:
                        # fallback: any xlsx link
                        m = re.search(r'href=["\']([^"\']+\.xlsx)["\']', r.text, flags=re.I)
                    if m:
                        excel_link = m.group(1)
                        if excel_link.startswith('/'):
                            excel_link = 'https://www.fao.org' + excel_link
                        logger.info("Found FAO Excel candidate: %s", excel_link)
                        try:
                            r2 = session.get(excel_link, timeout=30, headers=headers)
                            if r2.status_code == 200:
                                df_x = pd.read_excel(BytesIO(r2.content), skiprows=2)
                                df = df_x
                        except Exception:
                            logger.debug("Failed to read discovered FAO Excel link")
            except Exception:
                logger.debug("FAO page discovery failed")

        # 3) Try legacy CSVs as last resort
        if df is None:
            legacy_csvs = [
                "https://www.fao.org/fileadmin/templates/worldfood/Files/food_prices_index/Cereal_price_index.csv",
                "https://www.fao.org/fileadmin/templates/worldfood/Files/food_prices_index/FoodPriceIndex.csv",
                "https://www.fao.org/fileadmin/templates/worldfood/Images/food_prices_index/Food_Price_Index.csv",
            ]
            from io import StringIO

            for url in legacy_csvs:
                try:
                    logger.debug("Trying legacy FAO CSV: %s", url)
                    r = session.get(url, timeout=30, headers=headers)
                    if r.status_code == 200 and "Cereals" in r.text:
                        df = pd.read_csv(StringIO(r.text))
                        break
                except Exception:
                    continue

        # 4) Local manual file fallback
        if df is None:
            local_xlsx = os.path.join(self.data_external, "fao_manual.xlsx")
            if os.path.exists(local_xlsx):
                try:
                    logger.info("Reading local FAO file: %s", local_xlsx)
                    df = pd.read_excel(local_xlsx, skiprows=2)
                except Exception:
                    logger.debug("Failed to read local FAO file %s", local_xlsx)

        # If still no DF, return an empty DataFrame (don't raise) so the pipeline can continue
        if df is None:
            logger.warning("Could not locate FAO Food_Price_Index automatically; returning empty DataFrame. Visit %s to download manually.", page_url)
            return pd.DataFrame(columns=["date", "Cereals"]) 

        # normalize dataframe: find 'Cereals' column and a date-like column
        cereal_candidates = [c for c in df.columns if 'cereal' in c.lower()]
        if not cereal_candidates:
            # try broader names
            cereal_candidates = [c for c in df.columns if 'cereals' in c.lower() or 'cereals index' in c.lower()]
        if not cereal_candidates:
            raise RuntimeError("FAO file loaded but 'Cereals' column not found; inspect columns: %s" % list(df.columns))

        cereal_col = cereal_candidates[0]

        # detect date/month column
        date_col = None
        for c in df.columns:
            if any(k in c.lower() for k in ('date', 'month', 'period', 'time', 'year')):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]

        df = df[[date_col, cereal_col]].rename(columns={date_col: 'date', cereal_col: 'Cereals'})

        # parse dates; FAO loads months, ensure month start
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().all():
            # try YYYY-MM formats
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y-%m', errors='coerce')
        df = df.dropna(subset=['date'])
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df = df.sort_values('date').reset_index(drop=True)

        df = df[['date', 'Cereals']]
        return df

    def _resample_to_weekly(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Take a DataFrame with a 'date' column and a value column and return weekly values aligned to Monday.

        Steps:
        - ensure full daily index from start_date to end_date
        - interpolate/ffill missing daily values
        - compute week_start (Monday) and aggregate (mean)
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["week_start", value_col])

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        # drop any duplicate date labels (some tickers return duplicated dates)
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")]

        # daily index
        full_idx = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        df = df.reindex(full_idx)

        # ensure value column exists
        if value_col not in df.columns:
            df[value_col] = np.nan

        # forward-fill then linear interpolate for any gaps
        df[value_col] = df[value_col].ffill().interpolate(method="linear", limit_direction="both")

        df = df.reset_index().rename(columns={"index": "date_dt"})
        df["date_dt"] = pd.to_datetime(df["date_dt"])
        df["week_start"] = (df["date_dt"] - pd.to_timedelta(df["date_dt"].dt.weekday, unit="D")).dt.date

        weekly = df.groupby("week_start")[value_col].mean().reset_index()
        return weekly

    def run(self) -> None:
        """Fetch all sources, align to weekly, and save CSV to `data/raw/mercados_internacionales.csv`.

        Behavior:
        - Use ZW=F as default wheat proxy (column `Wheat_Global_Proxy_USD`).
        - Try W.PA before falling back to ZW=F.
        - Try ZC=F then C=F for corn and record used tickers.
        - Do not include FAO in CSV, but report its status.
        """

        # Ensure output dir exists
        out_dir = str(PROJECT_ROOT / "data" / "raw")
        os.makedirs(out_dir, exist_ok=True)

        # 1) Corn selection (ZC then C)
        corn_ticker_used = None
        zc = pd.DataFrame(columns=["date", "Close"])
        for ct in ["ZC=F", "C=F"]:
            try:
                cand = self._fetch_yahoo(ct)
            except Exception:
                cand = pd.DataFrame(columns=["date", "Close"])
            if cand is not None and not cand.empty:
                zc = cand
                corn_ticker_used = ct
                break
        if corn_ticker_used is None:
            logger.warning("No corn ticker returned sufficient data; ZC/C=F unavailable")

        # 2) Wheat selection: try W.PA, then default to ZW=F
        wheat_ticker_used = "ZW=F"
        wheat_df = pd.DataFrame(columns=["date", "Close"])
        try:
            wp = self._fetch_yahoo("W.PA")
        except Exception:
            wp = pd.DataFrame(columns=["date", "Close"])
        if wp is not None and not wp.empty:
            try:
                cnt_wp = int(wp["Close"].count())
                std_wp = float(wp["Close"].std())
            except Exception:
                cnt_wp = 0
                std_wp = 0.0
            if cnt_wp > 100 and std_wp > 0.0:
                wheat_df = wp
                wheat_ticker_used = "W.PA"
                logger.info("Using Paris ticker W.PA for wheat (rows=%d, std=%.4f)", cnt_wp, std_wp)
            else:
                logger.info("W.PA failed quality checks; falling back to Chicago proxy ZW=F")

        if wheat_df.empty:
            try:
                zw = self._fetch_yahoo("ZW=F")
            except Exception:
                zw = pd.DataFrame(columns=["date", "Close"])
            if zw is not None and not zw.empty:
                wheat_df = zw
                wheat_ticker_used = "ZW=F"
                logger.info("Using Chicago wheat proxy ZW=F (rows=%d)", int(wheat_df.get("Close", pd.Series(dtype=float)).count()))
            else:
                logger.warning("ZW=F returned no usable data; wheat series will be empty")

        # 3) Resample to weekly and merge
        wheat_col = "Wheat_Global_Proxy_USD"
        ebm_w = self._resample_to_weekly(wheat_df.rename(columns={"Close": wheat_col}), wheat_col)
        zc_w = self._resample_to_weekly(zc.rename(columns={"Close": "ZC_USD"}), "ZC_USD")
        df = ebm_w.merge(zc_w, on="week_start", how="outer")

        # 4) Ensure week_start format is ISO YYYY-MM-DD and aligned to Mondays
        df = df.sort_values("week_start").reset_index(drop=True)
        df["week_start"] = pd.to_datetime(df["week_start"]).dt.to_period("W-MON").dt.start_time.dt.date
        df["week_start"] = df["week_start"].astype(str)

        # 5) Prevent flatline: invalidate sources with too few weekly rows
        MIN_WEEKLY_ROWS = 100
        for col in [wheat_col, "ZC_USD"]:
            if col in df.columns:
                non_null = df[col].dropna().shape[0]
                if non_null < MIN_WEEKLY_ROWS:
                    logger.warning("%s has only %d weekly observations (< %d); marking as missing to avoid flatline.", col, non_null, MIN_WEEKLY_ROWS)
                    df[col] = np.nan

        # 6) Fill remaining gaps (only for columns with sufficient data)
        df = df.ffill().bfill()

        # 7) FAO status (do not include in CSV)
        try:
            fao_df = self._fetch_fao()
            fao_status = "Loaded" if (fao_df is not None and not fao_df.empty) else "Missing"
        except Exception:
            fao_status = "Missing"

        # 8) DATAGIA report (log to console)
        print(f"DATAGIA REPORT: Wheat Signal = {wheat_ticker_used}, Corn Signal = {corn_ticker_used or 'Missing'}, FAO Status = {fao_status}")

        # 9) Save CSV to data/raw
        out_path = os.path.join(out_dir, "mercados_internacionales.csv")
        df.to_csv(out_path, index=False)
        logger.info("Saved market signals to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mi = MarketIngestor(start_date=pd.to_datetime('2003-01-01').date(), end_date=date.today())
    mi.run()
