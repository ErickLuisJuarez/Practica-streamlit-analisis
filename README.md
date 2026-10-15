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
├── data/
│   ├── data-raw/              # Dataset original de la UCI (no se sube a GitHub)
│   └── data-processed/        # Dataset final: limpio, normalizado y con PCA
│
├── src/
│   ├── cleaning/
│   │   ├── __init__.py
│   │   └── preprocess.py      # Limpieza, duplicados, nulos y casteo de tipos
│   │
│   └── features/
│       ├── __init__.py
│       ├── normalization.py   # Normalización numérica (Min-Max / Z-score)
│       ├── correlation.py     # Correlación de Pearson y mapa de calor
│       └── pca_model.py       # Ajuste del PCA y varianza explicada
│
├── visualization/
│   └── app.py                 # Código principal del dashboard en Streamlit
│
├── .gitignore                 # Archivos ignorados por Git (venv, caché, datos)
├── README.md                  # Documentación y decisiones técnicas
└── requirements.txt           # Dependencias del proyecto
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
