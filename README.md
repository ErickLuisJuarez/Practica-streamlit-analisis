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

# Práctica 6 – Minería de reglas de asociación

Preparación de transacciones horarias y extracción de reglas de asociación mediante los algoritmos **Apriori** y **FP-Growth**, utilizando el dataset de consumo eléctrico doméstico de la Práctica 5.

## Objetivo

Identificar relaciones frecuentes entre el día de la semana, el periodo del día y las habitaciones o áreas con actividad eléctrica. Posteriormente, comparar los resultados obtenidos con Apriori y FP-Growth mediante sus conjuntos frecuentes, reglas de asociación y tiempos de ejecución.

## Preparación de los datos

A partir del archivo original `household_power_consumption.csv`, se construyeron registros horarios. Para cada hora se calcularon agregaciones de las variables de consumo y submedición, y se generaron indicadores de actividad para:

- `Cocina`
- `Lavanderia`
- `Climatizacion`

También se agregaron las variables categóricas `Dia_Semana` y `Tiempo_Dia`.

Los periodos del día utilizados fueron:

| Periodo | Rango de horas |
|---|---|
| Madrugada | 00:00 – 05:59 |
| Mañana | 06:00 – 11:59 |
| Mediodía | 12:00 – 13:59 |
| Tarde | 14:00 – 18:59 |
| Noche | 19:00 – 23:59 |

La columna `Habitaciones_Activas` reúne las áreas que registraron actividad en cada hora y se utiliza para construir las transacciones.

Para determinar si un área estuvo activa se utilizó el criterio de consumo mayor que cero en su respectiva submedición:

- `Sub_metering_1 > 0`: Cocina
- `Sub_metering_2 > 0`: Lavanderia
- `Sub_metering_3 > 0`: Climatizacion

El archivo preparado se guarda en:

```text
data/data-processed/input_reglas_asociacion.csv
```

## Algoritmos y métricas

Se implementaron dos algoritmos para encontrar conjuntos de ítems frecuentes:

- **Apriori:** obtiene conjuntos frecuentes a partir del soporte mínimo establecido.
- **FP-Growth:** obtiene conjuntos frecuentes mediante una estructura de árbol FP, evitando generar candidatos de la misma manera que Apriori.

A partir de los conjuntos frecuentes se generaron reglas de asociación y se consideraron las siguientes métricas:

- **Soporte:** frecuencia con la que aparece conjuntamente un conjunto de ítems en las transacciones.
- **Confianza:** proporción de transacciones con el antecedente que también contienen el consecuente.
- **Lift:** mide qué tanto aumenta la aparición del consecuente cuando aparece el antecedente. Un valor mayor que 1 indica una asociación positiva.

Se utilizó un umbral mínimo de confianza configurado en el módulo de reglas y se filtraron las reglas mediante un lift mínimo de `1.10`, para conservar asociaciones positivas con mayor fuerza.

## Comparación de algoritmos

Para comparar Apriori y FP-Growth se probaron los siguientes valores de soporte mínimo:

```python
SOPORTES_MINIMOS = [
    0.01, 0.02, 0.03, 0.05,
    0.10, 0.30, 0.50, 0.70,
]
```

La comparación considera:

- Cantidad de conjuntos frecuentes encontrados.
- Coincidencia entre los conjuntos frecuentes de ambos algoritmos.
- Tiempo de ejecución de cada algoritmo.
- Cantidad de reglas de asociación generadas.

Los resultados permiten observar cómo cambia la cantidad de conjuntos frecuentes al modificar el soporte mínimo. Un soporte bajo puede producir una gran cantidad de conjuntos y reglas, mientras que un soporte alto puede descartar patrones menos frecuentes.

## Ejecución

Ejecuta los módulos desde la raíz del repositorio y en el siguiente orden:

```bash
python -m src.cleaning.preparar_reglas
python -m src.models.reglas_asociacion
python -m src.models.comparacion_asociacion
python -m src.visualization.reglas_charts
```

Descripción de cada módulo:

1. `preparar_reglas`: prepara las transacciones horarias a partir del dataset original.
2. `reglas_asociacion`: ejecuta Apriori y FP-Growth, genera las reglas y guarda los resultados.
3. `comparacion_asociacion`: compara los algoritmos utilizando distintos valores de soporte mínimo.
4. `reglas_charts`: genera las visualizaciones de los resultados obtenidos.

## Archivos generados

| Carpeta | Archivo | Contenido |
|---|---|---|
| `data-processed/` | `input_reglas_asociacion.csv` | Transacciones horarias con variables temporales y áreas activas. |
| `data-processed/` | `reglas_apriori.csv` | Reglas de asociación obtenidas con Apriori. |
| `data-processed/` | `reglas_fpgrowth.csv` | Reglas de asociación obtenidas con FP-Growth. |
| `data-model/reglas_asociacion/` | `itemsets_apriori.csv` | Conjuntos frecuentes encontrados con Apriori. |
| `data-model/reglas_asociacion/` | `itemsets_fpgrowth.csv` | Conjuntos frecuentes encontrados con FP-Growth. |
| `data-model/reglas_asociacion/` | `comparacion_algoritmos.csv` | Resumen comparativo de los algoritmos y soportes probados. |
| `data-model/reglas_asociacion/` | `detalle_comparacion.csv` | Detalle de la comparación de conjuntos frecuentes. |
| `data-input-model/figures/reglas_asociacion/` | `*.png` | Gráficas generadas para analizar los resultados de las reglas. |

## Decisiones técnicas

- **Granularidad temporal:** Se agruparon las mediciones por hora para representar el comportamiento de consumo en periodos más amplios que el registro original por minuto.
- **Construcción de transacciones:** Cada transacción representa una hora e incluye información del día de la semana, el periodo del día y las áreas que presentaron actividad.
- **Actividad de las áreas:** Se consideró activa un área cuando su submedición registró un consumo mayor que cero.
- **Comparación:** Se ejecutaron ambos algoritmos con los mismos valores de soporte mínimo para facilitar la comparación de sus resultados.
- **Selección de reglas:** Se utilizó la confianza mínima configurada en el módulo y se filtraron las reglas con `lift >= 1.10`.
- **Organización de salidas:** Los archivos de reglas se guardan en `data-processed/`; los conjuntos frecuentes y los resultados de comparación se guardan en `data-model/reglas_asociacion/`. Las gráficas se conservan en una subcarpeta de `data-input-model/figures/`.