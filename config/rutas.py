from pathlib import Path

# Rutas del proyecto
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "data-raw"   # Original, nunca se edita
DATA_PROCESSED = ROOT / "data" / "data-processed"  # Limpio: sin duplicados, tipos corregidos, imputado
DATA_INPUT_MODEL = ROOT / "data" / "data-input-model" # Transformado: listo como entrada de un modelo
DATA_MODEL = ROOT / "data" / "data-model"  # Salidas de modelos (predicciones, clusters, reglas)
FIGURES = DATA_INPUT_MODEL / "figures"

# Archivo CSV único
RAW_FILE = DATA_RAW / "household_power_consumption.csv"

#Columnas
DATE_COL = "Date"
TIME_COL = "Time"
DATETIME_COL = "datetime"
NUMERIC_COLS = [
    "Global_active_power", #kW
    "Global_reactive_power", #kW
    "Voltage", #V
    "Global_intensity", #A
    "Sub_metering_1", #Wh (cocina)
    "Sub_metering_2", #Wh (lavandería)
    "Sub_metering_3", #Wh (calentador de agua y A/C)
]
NULL_TOKENS = ["?", ""]  #Marcadores de faltante del dataset

# Decisiones (Paso 2)
MAX_GAP_MINUTES = 60

#Decisiones (Paso 3)
NORMALIZATION_METHOD = "zscore" #"zscore" | "minmax"
TRAIN_FRACTION = 0.70 #Primeros 70% en el tiempo -> ajusta parámetros

#Decisiones (Paso 4)
CORR_THRESHOLD = 0.90
CORR_PREFER_KEEP = ["Global_active_power"]  #Variable objetivo típica de regresión

#Decisiones (Paso 5)
PCA_VARIANCE_TARGET = 0.90 #Varianza acumulada mínima a conservar
PCA_PLOT_SAMPLE = 20_000  #Puntos para la proyección 2D  
RANDOM_STATE = 42
