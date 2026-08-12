# Contenedor de datos

Para el correcto funcionamiento del proyecto, se deben colocar en esta carpeta:

- Datos crudos del simulador (`raw`)
- Datos procesados del simulador (`processed`)
- Splits del simulador (`splits`)
- Clima histórico (`clima_real`)
- Salidas de inferencia de la IA (`predictions`)


### IMPORTANTE: El contenido de esta carpeta no puede ser subida al repositorio debido a su gran tamaño.

Para obtener estos archivos, se deben generar siguiendo los pasos de la guía de ejecución, o bien descargar los datos crudos y splits del contenedor externo.

# Estructura esperada de esta carpeta tras ejecutar el simulador y los modelos:

```
data/
├── raw/
│   └── data_vin_raw.parquet
├── processed/
│   ├── data_vin_processed.parquet
|   ├── estadisticas.csv
│   └── feature_descriptions.csv
├── splits/
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
├── clima_real/
│   ├── clean/
│   │   └── clima_<parcela>_clean.parquet
│   └── clima_<parcela>.parquet
└── predictions/
    └── inferencia_vid.csv
```