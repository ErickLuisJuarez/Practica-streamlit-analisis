# Práctica 5 – Consumo eléctrico doméstico

Preparación de datos (normalización, correlación de Pearson, PCA) y dashboard en Streamlit.
Almacenes y Minería de Datos · Facultad de Ciencias, UNAM.

## Dataset

**Individual Household Electric Power Consumption** (Hebrail & Berard, 2006), UCI Machine Learning Repository.
Mediciones por minuto del consumo eléctrico de un hogar (dic 2006 – nov 2010), ~2.07 millones de registros, multivariado y de series de tiempo.

- Fuente: <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>
- El archivo original pesa ~127 MB y **no se sube al repositorio**. Descárgalo y colócalo en
  `data/data-raw/household_power_consumption.csv`.

## Estructura

```
Practica-streamlit-analisis/
├── config/
│   └── rutas.py                   # rutas de archivos estáticos y decisiones técnicas
├── data/
│   ├── data-raw/                  # datos originales, tal como se descargaron (nunca se editan)
│   ├── data-processed/            # datos limpios: sin duplicados, tipos corregidos, imputados
│   ├── data-input-model/          # datos transformados, listos como entrada de un modelo
│   │   └── figures/               # mapa de calor, scree plot, proyección 2D
│   └── data-model/                # salidas de modelos (predicciones, clusters, reglas)
├── src/
│   ├── cleaning/                  # limpieza y preprocesamiento
│   │   ├── preprocess.py          # carga, inspección, limpieza (Pasos 1-2)
│   │   └── run_pipeline.py        # ejecuta los Pasos 1-6 en orden
│   ├── models/                    # modelos de la práctica
│   │   ├── normalizacion.py       # normalización numérica (Paso 3)
│   │   ├── correlacion.py         # correlación de Pearson y selección (Paso 4)
│   │   └── pca_model.py           # PCA (Paso 5)
│   └── visualization/             # código auxiliar del dashboard: gráficas y métricas
│       ├── data_loader.py         # lectura de datos procesados
│       ├── metrics.py             # KPIs, agregaciones, tablas
│       └── charts.py              # gráficas (Plotly)
├── notebooks/                     # notebooks exploratorios (no productivos)
├── visualization/
│   └── app.py                     # código principal del dashboard (Streamlit)
├── README.md
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Siempre desde la raíz del repositorio:

```bash
python -m src.cleaning.run_pipeline    # genera data-processed/ y data-input-model/ (~1 minuto)
streamlit run visualization/app.py
```

Salidas del pipeline:

| Carpeta                     | Archivo                                                                                | Contenido                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `data-processed/`           | `household_clean.parquet`                                                              | Dataset limpio en unidades originales (kW, V, A, Wh). Lo usa el dashboard para KPIs y gráficas.                   |
| `data-processed/`           | `cleaning_report.json`                                                                 | Inspección inicial y conteos de la limpieza.                                                                      |
| `data-input-model/`         | `household_prepared.parquet`                                                           | Dataset final del Paso 6: variables normalizadas, sin redundantes, con componentes principales y columna `split`. |
| `data-input-model/`         | `scaler_params.json`, `correlation_report.json`, `pca_report.json`, `pca_loadings.csv` | Parámetros y reportes de los Pasos 3-5.                                                                           |
| `data-input-model/figures/` | `*.png`                                                                                | Gráficas para la presentación.                                                                                    |
| `data-model/`               | —                                                                                      | Vacía en esta práctica: es para las salidas de los modelos de las siguientes prácticas.                           |

## Decisiones técnicas

- **Valores faltantes:** Se aplicó un remuestreo (upsampling) para garantizar una fila por minuto. Los huecos menores o iguales a 60 minutos (`MAX_GAP_MINUTES = 60`) se rellenaron mediante interpolación lineal. Los registros en huecos mayores a 60 minutos se eliminaron para no inventar tendencias irreales.
- **Duplicados:** Se eliminaron las filas exactamente idénticas. Para las filas con el mismo timestamp (fecha y hora repetida), se conservó únicamente la primera aparición cronológica (`keep="first"`).
- **Normalización:** Se utilizó `z-score` (StandardScaler). Para respetar la naturaleza de la serie de tiempo y evitar el "data leakage" (filtrar información del futuro hacia el pasado), los parámetros de media y desviación estándar se calcularon **exclusivamente con el primer 70% cronológico de los datos** (`TRAIN_FRACTION = 0.70`).
- **Correlación de Pearson:** Se usó un umbral de redundancia de `|r| >= 0.90` porque a partir de ese valor las variables explican el mismo fenómeno. Como variable preferida a conservar se definió `Global_active_power` (típica variable objetivo). La variable descartada es `Global_intensity` (debido a su redundancia casi perfecta con la potencia).
- **PCA:** Se retuvieron los primeros componentes que logran acumular un objetivo de varianza del `90%` (`PCA_VARIANCE_TARGET = 0.90`). La interpretación de los loadings nos indica que el PC1 captura la magnitud global del consumo, mientras que los siguientes componentes capturan patrones específicos de uso en los submedidores.
