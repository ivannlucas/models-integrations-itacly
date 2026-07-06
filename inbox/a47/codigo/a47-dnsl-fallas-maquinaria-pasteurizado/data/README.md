Carpeta que contiene los datos del proyecto.

### Subdirectorios

- **`raw/`**: Deben situarse aquí los datos originales en crudo. Por defecto, el sistema espera encontrar una carpeta llamada `Dataset_Hydraulic` con los archivos .txt. (Descargarlos del enlace: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems).

- **`processed/`**: Se generan los datos limpios y procesados (10Hz).

- **`predictions/`**: Ficheros de salida de las inferencias.

- **`splits/`**: Se generan las divisiones de datos para entrenamiento, validación y prueba.