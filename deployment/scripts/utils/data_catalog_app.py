import json
import os

import pandas as pd
import streamlit as st

# Constante para el nombre del archivo
CATALOG_FILE = "master_catalog.json"


@st.cache_data
def load_catalog(filename):
    """Carga el catálogo maestro desde JSON."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error al cargar {filename}: {e}")
        return None


# --- Configuración de la Página ---
st.set_page_config(
    page_title="Catálogo de Datos",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visor del Catálogo de Datos")
st.write("Explora las entidades de datos y su linaje, generados por los ETLs.")

# --- Carga de Datos ---
catalog_data = load_catalog(CATALOG_FILE)

if not catalog_data:
    st.error(f"No se pudo cargar el archivo '{CATALOG_FILE}'.")
    st.info(f"Asegúrate de que '{CATALOG_FILE}' existe en el mismo directorio que 'app.py'.")
    st.stop()  # Detiene la ejecución si no hay datos

# --- Sidebar de Navegación ---
st.sidebar.header("Navegación")
entity_names = sorted(catalog_data.keys())

# Filtro de búsqueda
search_term = st.sidebar.text_input("Buscar entidad:", "")
if search_term:
    filtered_entities = [name for name in entity_names if search_term.lower() in name.lower()]
else:
    filtered_entities = entity_names

if not filtered_entities:
    st.sidebar.warning("No se encontraron entidades.")
    st.stop()

selected_entity_name = st.sidebar.radio(
    "Selecciona una Entidad de Datos:",
    filtered_entities
)

# --- Panel Principal ---
st.header(f"Detalles de: `{selected_entity_name}`")

entity_data = catalog_data.get(selected_entity_name)

if not entity_data:
    st.error("Entidad no encontrada. Esto no debería ocurrir.")
    st.stop()

# --- Mostrar Información General ---
st.subheader("Información General")

col1, col2 = st.columns(2)
with col1:
    st.metric("Escrito por (ETL Propietario)",
              entity_data.get('written_by_etl') or "N/A (Fuente Externa)")
with col2:
    st.metric("Consumido por (Nº ETLs)",
              len(entity_data.get('consumed_by_etls', [])))

# --- Mostrar Linaje de Entidades ---
st.subheader("Linaje de Entidades")

col_up, col_down = st.columns(2)

with col_up:
    st.markdown("##### ⬆️ Fuentes (Upstream)")
    upstreams = entity_data.get('upstream_sources', [])
    if upstreams:
        for up in upstreams:
            st.code(up, language="text")
    else:
        st.info("No se registraron fuentes (es una entidad raíz o de origen).")

with col_down:
    st.markdown("##### ⬇️ Consumidores (Downstream)")
    downstreams = entity_data.get('consumed_by_etls', [])
    if downstreams:
        for down in downstreams:
            st.code(down, language="text")
    else:
        st.info("Esta entidad no está siendo consumida por otros ETLs.")

# --- Mostrar Linaje de Columnas (Esquema) ---
st.subheader("Linaje de Columnas (Esquema)")
schema_lineage = entity_data.get('schema_lineage', [])

if schema_lineage:
    # Convertir a DataFrame de Pandas para una mejor visualización
    df = pd.DataFrame(schema_lineage)

    # Reordenar columnas para mejor legibilidad
    cols_order = ['column', 'inferred_type', 'description', 'transformation_logic', 'source_columns']
    # Filtrar solo las columnas que existen
    df = df[[col for col in cols_order if col in df.columns]]
    df.rename(columns={'column': 'Column', 'inferred_type': 'Type', 'description': 'Description',
                       'transformation_logic': 'Transformation', 'source_columns': 'Source columns'},
              inplace=True)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No hay linaje de columnas disponible para esta entidad (probablemente es una fuente raíz).")

# Opcional: Mostrar el JSON crudo
with st.expander("Ver JSON crudo de la entidad"):
    st.json(entity_data)
