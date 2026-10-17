# Práctica 5 – Consumo eléctrico doméstico

Preparación de datos (normalización, Pearson, PCA) y dashboard en Streamlit.
Almacenes y Minería de Datos – Facultad de Ciencias, UNAM.

## Dataset
Individual Household Electric Power Consumption (UCI).
Descarga: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
Coloca `household_power_consumption.txt` y `household_power_consumption.csv` en `data/data-raw/`.

## Estructura del proyecto

```
Practica-streamlit-analisis/
├── config/
│   ├── __init__.py
│   └── rutas.py                   # Rutas y decisiones técnicas (umbrales, método, etc.)
│
├── data/
│   ├── data-raw/                  # Dataset original de la UCI (no se sube a GitHub)
│   └── data-processed/            # Salidas del pipeline
│       └── figures/               # Mapa de calor, scree plot, proyección 2D
│
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── run_pipeline.py        # Ejecuta los Pasos 1-6 en orden
│   │
│   ├── cleaning/
│   │   ├── __init__.py
│   │   └── preprocess.py          # Carga, inspección, duplicados, nulos y casteo (Pasos 1-2)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── normalization.py       # Normalización numérica (Paso 3)
│   │   ├── correlation.py         # Correlación de Pearson y selección de variables (Paso 4)
│   │   └── pca_model.py           # PCA, varianza explicada y loadings (Paso 5)
│   │
│   └── dashboard/
│       ├── __init__.py
│       ├── data_loader.py         # Lectura de los datos procesados
│       ├── metrics.py             # KPIs, agregaciones y tablas
│       └── charts.py              # Gráficas (Plotly)
│
├── visualization/
│   └── app.py                     # Código principal del dashboard (Streamlit)
│
├── .gitignore                     # Archivos ignorados por Git (venv, caché, dataset crudo)
├── README.md                      # Documentación y decisiones técnicas
└── requirements.txt               # Dependencias del proyecto
```

| Carpeta / archivo | Descripción |
|---|---|
| `data/data-raw/` | Dato original sin modificar. Ver sección de descarga. |
| `data/data-processed/` | Resultado final del preprocesamiento, normalización y PCA. |
| `src/cleaning/` | Módulos de preprocesamiento. |
| `src/features/` | Normalización, correlación y PCA. |
| `visualization/app.py` | Dashboard que lee el dataset procesado. |

## Instalación y ejecución
pip install -r requirements.txt
streamlit run visualization/app.py

## Decisiones técnicas


## Autor(es)

* Equipo: Erick Luis Juárez
Julio Alejandro Herrera Avalos
Luis Mario Solares Ramos
Cesar Candelario Fuentes Anica
