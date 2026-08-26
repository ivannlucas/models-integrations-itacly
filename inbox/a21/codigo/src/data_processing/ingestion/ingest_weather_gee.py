"""Ingesta de datos climáticos provinciales desde Google Earth Engine (GEE).

Descarga datos diarios de temperatura y precipitación del dataset ERA5-Land
para las 52 provincias españolas, iterando por año y mes. Soporta autenticación
mediante cuenta de usuario (`earthengine authenticate`) o cuenta de servicio
(variable de entorno GOOGLE_APPLICATION_CREDENTIALS).

Entradas:
  - Google Earth Engine API (ECMWF/ERA5_LAND/DAILY_AGGR)
  - config/config.yaml  (clave gee_project_id)

Salidas:
  - data/processed/climate_provinces_GEE_by_year/climate_provinces_GEE_YYYY_MM.csv
    (una fila por provincia×día; columnas: date_ms, province_name, temp_k,
     temp_std_k, precip_m, precip_std_m)

Uso:
  python src/data_processing/ingestion/ingest_weather_gee.py
"""
import ee
import pandas as pd
import os
import yaml
import logging
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

class GEEWeatherIngestor:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = PROJECT_ROOT / 'config' / 'config.yaml'
        self._load_config(config_path)
        # Inicializar Earth Engine
        try:
            gee_project = self.config.get('gee_project_id')
            # If the placeholder value is present, treat it as missing and initialize without project
            if gee_project and gee_project != 'tu-proyecto-gee':
                ee.Initialize(project=gee_project)
                logger.info(f"GEE inicializado con el proyecto: {gee_project}")
            else:
                # Intentar inicializar sin pasar project (usa el proyecto por defecto del usuario autenticado)
                ee.Initialize()
                logger.info("GEE inicializado sin argumento de proyecto (usar proyecto por defecto autenticado).")
        except Exception as e:
            # Intentar inicializar mediante cuenta de servicio si está proporcionada
            sa_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if sa_key and os.path.exists(sa_key):
                try:
                    import json
                    with open(sa_key, 'r', encoding='utf-8') as _f:
                        sa_json = json.load(_f)
                    service_account_email = sa_json.get('client_email')
                    if service_account_email:
                        creds = ee.ServiceAccountCredentials(service_account_email, sa_key)
                        # Si hay un proyecto válido, pásalo; si no, intenta sin project
                        try:
                            if gee_project and gee_project != 'tu-proyecto-gee':
                                ee.Initialize(credentials=creds, project=gee_project)
                                logger.info(f"GEE inicializado con cuenta de servicio y proyecto: {gee_project}")
                            else:
                                ee.Initialize(credentials=creds)
                                logger.info("GEE inicializado con cuenta de servicio (sin project especificado).")
                        except Exception as e2:
                            logger.error("Error inicializando GEE con cuenta de servicio: %s", e2)
                            logger.error("Si no usas cuenta de servicio, ejecuta 'earthengine authenticate' en este equipo y/o configura 'gee_project_id' en config/config.yaml con un proyecto válido.")
                            raise
                    else:
                        logger.error("El archivo de clave de servicio no contiene 'client_email'. No se puede inicializar con cuenta de servicio.")
                        logger.error("Detalles del error original: %s", e)
                        raise
                except Exception:
                    # Si falla la cuenta de servicio, mostrar instrucciones útiles
                    logger.error("Fallo al intentar inicializar con cuenta de servicio. Por favor, verifica 'GOOGLE_APPLICATION_CREDENTIALS' y permisos.")
                    logger.error("Detalles del error original: %s", e)
                    raise
            else:
                logger.error("Error al inicializar GEE. ¿Has ejecutado 'earthengine authenticate'? Detalles: %s", e)
                logger.error("Opciones:")
                logger.error(" - Ejecuta 'earthengine authenticate' en la terminal y reintenta.")
                logger.error(" - Proporciona un proyecto GCP con Earth Engine habilitado y pon su id en 'gee_project_id' en config/config.yaml.")
                logger.error(" - O configura una cuenta de servicio y exporta GOOGLE_APPLICATION_CREDENTIALS apuntando al JSON.")
                raise

    def _load_config(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        # Cargar configuración y asegurar que existe gee_project_id
        with open(path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}

        # Añadir clave por defecto si falta
        if 'gee_project_id' not in self.config:
            self.config['gee_project_id'] = 'tu-proyecto-gee'
            try:
                with open(path, 'w', encoding='utf-8') as wf:
                    yaml.safe_dump(self.config, wf, sort_keys=False)
                logger.info(f"Se añadió 'gee_project_id' a {path} con placeholder.")
            except Exception as e:
                logger.warning(f"No se pudo escribir en {path}: {e}")

        self.start_date = self.config.get('weather', {}).get('start_date', '2018-01-01')
        self.end_date = self.config.get('weather', {}).get('end_date', datetime.now().strftime('%Y-%m-%d'))
        output_path = Path(self.config.get('paths', {}).get('data_external', 'data/external'))
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        self.output_path = str(output_path)

    def get_spain_provinces(self):
        """Obtiene polígonos de las provincias de España desde GAUL."""
        return ee.FeatureCollection("FAO/GAUL/2015/level2") \
                 .filter(ee.Filter.eq('ADM0_NAME', 'Spain'))

    def fetch_data(self):
        provinces = self.get_spain_provinces()
        # Dataset base (sin filtrar por fecha aún)
        base_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

        # Detectar nombres de banda disponibles en la primera imagen para elegir precipitación correctamente
        try:
            first_img = base_collection.first()
            available_bands = first_img.bandNames().getInfo() if first_img else []
        except Exception:
            available_bands = []

        # Prefer temperature_2m and look for any precip-related band
        temp_band = 'temperature_2m' if 'temperature_2m' in available_bands else (available_bands[0] if available_bands else None)
        precip_candidates = [b for b in available_bands if 'precip' in b or 'precipitation' in b]
        precip_band = precip_candidates[0] if precip_candidates else None

        select_bands = [b for b in (temp_band, precip_band) if b]
        if select_bands:
            base_collection = base_collection.select(select_bands)

        logger.info(f"Iniciando extracción GEE desde {self.start_date} hasta {self.end_date} por años (chunking)...")

        # Función para procesar cada imagen (día) y reducir por provincia
        def reduce_spatial(image):
            # OJO: 'yyyy-MM-dd' en minúsculas. Con 'YYYY-MM-DD' (mayúsculas), la
            # librería de fechas de Earth Engine interpreta 'DD' como día del año
            # (no día del mes), lo que corrompe la fecha a partir de febrero
            # (ej. 10 de abril -> "2026-04-100" en vez de "2026-04-10").
            date = image.date().format('yyyy-MM-dd')
            return image.reduceRegions(
                collection=provinces,
                reducer=ee.Reducer.mean(),
                scale=9000
            ).map(lambda f: f.set('date', date))

        # Preparar rango por meses para mejor monitoreo (fallback semanal si falla)
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        month_starts = pd.date_range(start=start_dt.replace(day=1), end=end_dt.replace(day=1), freq='MS')
        month_ranges = []
        for ms in month_starts:
            me = (ms + pd.offsets.MonthEnd(1)).to_pydatetime()
            ms_dt = ms.to_pydatetime()
            if me < start_dt or ms_dt > end_dt:
                continue
            ms_clip = max(ms_dt, start_dt)
            me_clip = min(me, end_dt)
            month_ranges.append((ms_clip.strftime('%Y-%m-%d'), me_clip.strftime('%Y-%m-%d')))

        data = []
        total_months = len(month_ranges)
        processed_months = 0
        chunks_since_flush = 0

        def fetch_and_append(start_iso, end_iso, retries=3, backoff_base=2.0):
            era5_chunk = base_collection.filterDate(start_iso, end_iso)
            raw_stats_chunk = era5_chunk.map(reduce_spatial).flatten()
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    t0 = time.time()
                    info = raw_stats_chunk.getInfo()
                    elapsed = time.time() - t0
                    features_chunk = info.get('features', []) if isinstance(info, dict) else []
                    # Append features
                    for f in features_chunk:
                        props = f.get('properties', {})
                        temp_val = props.get(temp_band) if temp_band else props.get('temperature_2m')
                        precip_val = props.get(precip_band) if precip_band else (props.get('total_precipitation') or props.get('total_precipitation_sum'))
                        data.append({
                            'date': props.get('date'),
                            'provincia': props.get('ADM2_NAME'),
                            'temp_k': temp_val,
                            'precip_m': precip_val
                        })
                    logger.info(f"Chunk {start_iso} -> {end_iso} completado. features={len(features_chunk)} elapsed={elapsed:.1f}s")
                    return len(features_chunk)
                except Exception as e:
                    last_err = e
                    logger.warning(f"Error en chunk {start_iso} -> {end_iso} (intento {attempt}/{retries}): {e}")
                    time.sleep(backoff_base ** attempt)
            logger.error(f"Fallo definitivo en chunk {start_iso} -> {end_iso}: {last_err}")
            return None

        def flush_partial(tag):
            if not data:
                return
            try:
                os.makedirs(self.output_path, exist_ok=True)
                partial_file = os.path.join(self.output_path, 'clima_provincias_GEE_partial.csv')
                pd.DataFrame(data).to_csv(partial_file, index=False)
                logger.info(f"Checkpoint parcial guardado ({tag}) en {partial_file} | rows={len(data)}")
            except Exception as e:
                logger.warning(f"No se pudo guardar checkpoint parcial: {e}")

        for idx, (m_start_iso, m_end_iso) in enumerate(month_ranges, start=1):
            processed_months += 1
            percent = (processed_months / total_months) * 100 if total_months else 0
            print(f"Procesando mes {processed_months}/{total_months} ({percent:.1f}%) -> {m_start_iso} a {m_end_iso}")
            logger.info(f"Extrayendo rango {m_start_iso} -> {m_end_iso}")

            cnt_m = fetch_and_append(m_start_iso, m_end_iso)
            time.sleep(0.3)

            # If monthly chunk fails or too big, fallback to weekly
            if cnt_m is None or (isinstance(cnt_m, int) and cnt_m > 4500):
                logger.info(f"Falling back to weekly chunks for {m_start_iso} -> {m_end_iso} (count={cnt_m}).")
                w_start = pd.to_datetime(m_start_iso)
                w_end = pd.to_datetime(m_end_iso)
                weeks = pd.date_range(start=w_start, end=w_end, freq='7D')
                for wi, w in enumerate(weeks, start=1):
                    ws = w.strftime('%Y-%m-%d')
                    we = (w + timedelta(days=6)).strftime('%Y-%m-%d')
                    ws_clip = max(pd.to_datetime(ws), w_start).strftime('%Y-%m-%d')
                    we_clip = min(pd.to_datetime(we), w_end).strftime('%Y-%m-%d')
                    print(f"  Semana {wi}/{len(weeks)} -> {ws_clip} a {we_clip}")
                    fetch_and_append(ws_clip, we_clip)
                    time.sleep(0.2)

            chunks_since_flush += 1
            if chunks_since_flush >= 3:
                flush_partial(tag=f"mes_{processed_months}")
                chunks_since_flush = 0

        df = pd.DataFrame(data)
        
        # Conversiones técnicas: Kelvin -> Celsius y metros -> mm
        if not df.empty:
            # Coerce numeric
            df['temp_k'] = pd.to_numeric(df['temp_k'], errors='coerce')
            df['precip_m'] = pd.to_numeric(df['precip_m'], errors='coerce')
            df['temperature'] = df['temp_k'] - 273.15
            df['precipitation'] = df['precip_m'] * 1000
            return df[['date', 'provincia', 'temperature', 'precipitation']]
        else:
            logger.warning("No se han obtenido features desde GEE; el DataFrame resultante está vacío.")
            # Devolver dataframe vacío con columnas esperadas
            return pd.DataFrame(columns=['date', 'provincia', 'temperature', 'precipitation'])

    def process_to_weekly(self, df):
        """Agrega datos diarios a semanales alineados a Lunes."""
        # Convert dates robustly and drop invalid
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
        df = df.dropna(subset=['date']).copy()
        df['week_start'] = df['date'].dt.to_period('W-MON').apply(lambda r: r.start_time.date())
        
        weekly = df.groupby(['week_start', 'provincia']).agg({
            'temperature': 'mean',
            'precipitation': 'sum'
        }).reset_index()
        
        return weekly

    def load_partial(self):
        partial_file = os.path.join(self.output_path, 'clima_provincias_GEE_partial.csv')
        if not os.path.exists(partial_file):
            raise FileNotFoundError(f"No se encontró el archivo parcial: {partial_file}")

        df = pd.read_csv(partial_file)

        # Ensure expected columns exist
        if 'temperature' not in df.columns:
            if 'temp_k' in df.columns:
                df['temperature'] = pd.to_numeric(df['temp_k'], errors='coerce') - 273.15
        if 'precipitation' not in df.columns:
            if 'precip_m' in df.columns:
                df['precipitation'] = pd.to_numeric(df['precip_m'], errors='coerce') * 1000

        # Keep only required columns
        cols = ['date', 'provincia', 'temperature', 'precipitation']
        return df[cols]

    def run(self, use_partial=False):
        df_daily = self.load_partial() if use_partial else self.fetch_data()

        # Guardar diario (además del semanal): permite agregar a mensual por
        # mes de calendario real (merge_recent_climate.py), sin el sesgo de
        # asignar toda una semana ISO al mes de su lunes de inicio.
        os.makedirs(self.output_path, exist_ok=True)
        daily_file = os.path.join(self.output_path, 'clima_provincias_GEE_daily.csv')
        df_daily.to_csv(daily_file, index=False)
        logger.info(f"Archivo diario guardado en {daily_file}")

        df_weekly = self.process_to_weekly(df_daily)

        # Guardar Tidy
        tidy_file = os.path.join(self.output_path, 'clima_provincias_GEE.csv')
        df_weekly.to_csv(tidy_file, index=False)
        logger.info(f"Archivo Tidy guardado en {tidy_file}")

        # Crear y guardar Pivot
        precip_pivot = df_weekly.pivot(index='week_start', columns='provincia', values='precipitation')
        precip_pivot.columns = [f"Precip_{c}" for c in precip_pivot.columns]
        
        temp_pivot = df_weekly.pivot(index='week_start', columns='provincia', values='temperature')
        temp_pivot.columns = [f"Temp_{c}" for c in temp_pivot.columns]
        
        pivot_final = pd.concat([precip_pivot, temp_pivot], axis=1).reset_index()
        pivot_file = os.path.join(self.output_path, 'clima_provincias_GEE_pivot.csv')
        pivot_final.to_csv(pivot_file, index=False)
        logger.info(f"Archivo Pivot guardado en {pivot_file}")

if __name__ == "__main__":
    ingestor = GEEWeatherIngestor()
    use_partial = '--from-partial' in sys.argv
    ingestor.run(use_partial=use_partial)