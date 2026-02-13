# 🍷 Directorio de Datos

Este directorio almacena la materia prima para el entrenamiento del modelo de predicción de precios.

## 📁 Estructura Requerida

Para que el pipeline funcione, el archivo CSV debe estar en la carpeta `raw`:

```
data/
├── raw/
│   └── mapa_wine_prices_raw.csv    # <--- ARCHIVO PRINCIPAL (Crudo)
│
└── processed/                      # Archivos intermedios generados por el sistema
    └── ...
```

> **Nota:** El sistema gestiona internamente la división de datos (Train/Val/Test) mediante validación *Walk-Forward*, por lo que **no** necesitas separar los datos en carpetas manuales.

## 📝 Formato del Archivo (`.csv`)

El archivo `mapa_wine_prices_raw.csv` debe ser un CSV separado por comas con las siguientes columnas obligatorias:

| Columna                        | Tipo      | Descripción                                      | Ejemplo         |
| :----------------------------- | :-------- | :------------------------------------------------ | :-------------- |
| **`campaign`**         | `str`   | Campaña vitivinícola (Año Inicio / Año Fin).  | `"2022/2023"` |
| **`week`**             | `int`   | Número de semana del año (formato ISO).         | `35`          |
| **`price_red`**        | `float` | Precio del vino tinto (€/hl).                    | `42.50`       |
| *(Opcional)* `price_white` | `float` | Precio del vino blanco (extraído pero no usado). | `38.20`       |

**Ejemplo de contenido:**

```csv
campaign,week,price_red,price_white,source_url,scraped_at
2022/2023,31,35.45,NaN,http://...,2023-10-01
2022/2023,32,36.10,NaN,http://...,2023-10-08
...
```

## 🔧 Obtención de Datos (Scraping)

Desde la raíz del proyecto:

`python ../scrapers/mapa_prices_scrapper.py`

Este script:

1. Descarga los últimos boletines del MAPA.
2. Parsea los PDFs para extraer precios.
3. Genera y guarda automáticamente el archivo en `data/raw/mapa_wine_prices_raw.csv`.

## ⚠️ Nota sobre Privacidad y Git

El contenido de este directorio (`*.csv`) **NO se incluye en el repositorio** (está ignorado por `.gitignore`) porque:

* Se prioriza la **reproducibilidad** a través del código (`scrapers/`) en lugar de almacenar binarios estáticos.
* Evita conflictos de fusión (merge conflicts) en archivos de datos grandes.
* Mantiene el repositorio ligero.

Para empezar, ejecuta el scraper o coloca tu propio histórico en la carpeta `raw/`.
