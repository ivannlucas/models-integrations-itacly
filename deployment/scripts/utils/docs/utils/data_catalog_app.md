# Script: app.py

## Resumen y Objetivo

Este script es una aplicación de Streamlit que actúa como un visor interactivo para un catálogo de datos. Su objetivo es
permitir a los desarrolladores y analistas explorar las entidades de datos disponibles, sus metadatos, linaje (fuentes y
consumidores) y el esquema de columnas, facilitando la comprensión del ecosistema de datos y el impacto de los procesos
ETL.

## Arquitectura y Lógica Principal

La aplicación sigue una arquitectura típica de Streamlit:

1. **Configuración Inicial:** Establece el título, ícono y layout de la página.
2. **Carga de Datos:** Intenta cargar un archivo `master_catalog.json` que contiene la información de todas las
   entidades de datos.
3. **Manejo de Errores de Carga:** Si el archivo no se encuentra o hay un error al cargarlo, muestra un mensaje de error
   y detiene la ejecución.
4. **Sidebar de Navegación:**
    * Muestra un encabezado "Navegación".
    * Presenta una lista de todas las entidades de datos disponibles.
    * Incluye un campo de texto para filtrar las entidades por nombre.
    * Utiliza `st.sidebar.radio` para permitir la selección de una entidad específica.
5. **Panel Principal:**
    * Muestra el nombre de la entidad seleccionada.
    * Presenta la información general de la entidad, incluyendo el ETL propietario y el número de ETLs que la consumen.
    * Detalla el linaje de entidades (fuentes upstream y consumidores downstream).
    * Muestra el linaje de columnas (esquema) en un formato de tabla interactiva (DataFrame de Pandas).
    * Opcionalmente, permite ver el JSON crudo de la entidad seleccionada a través de un `st.expander`.

## Configuración Requerida

* **Archivo:** `master_catalog.json` debe existir en el mismo directorio que el script `app.py`. Este archivo debe
  contener la estructura de datos esperada para el catálogo.

## Entradas y Salidas

* **Entrada:**
    * Archivo `master_catalog.json`: Contiene la definición de las entidades de datos, sus metadatos, linaje y esquema.
* **Salida:**
    * Interfaz de usuario interactiva (web) que visualiza la información del catálogo de datos.

## Ejemplos de Uso

Para ejecutar esta aplicación, asegúrate de tener `streamlit` y `pandas` instalados (`pip install streamlit pandas`).
Luego, ejecuta el script desde la terminal:

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador web, permitiéndote navegar y buscar en el catálogo de datos.

## Mantenimiento y Puntos Clave

* **Dependencia Crítica:** La aplicación depende completamente de la existencia y formato correcto del archivo
  `master_catalog.json`. Cualquier inconsistencia en este archivo puede causar errores o visualizaciones incompletas.
* **Rendimiento:** Para catálogos de datos muy grandes, la carga inicial y la búsqueda/filtrado podrían volverse lentos.
  La función `load_catalog` utiliza `@st.cache_data` para mitigar esto, pero la eficiencia del JSON y la estructura de
  datos subyacente son importantes.
* **Extensibilidad:** El código está diseñado para ser modular. Se pueden añadir nuevas visualizaciones o
  funcionalidades en el panel principal o la sidebar fácilmente.
* **Fuentes Raíz:** Las entidades que no tienen `upstream_sources` o `consumed_by_etls` se consideran entidades raíz o
  terminales, respectivamente. El script maneja estas condiciones mostrando mensajes informativos.
