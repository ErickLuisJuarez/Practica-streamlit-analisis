
from pathlib import Path

# Rutas principales del proyecto
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "data-raw"
DATA_PROCESSED = ROOT / "data" / "data-processed"
DATA_INPUT_MODEL = ROOT / "data" / "data-input-model"
DATA_MODEL = ROOT / "data" / "data-model"

FIGURES = DATA_INPUT_MODEL / "figures"

# Archivo original
RAW_FILE = DATA_RAW / "household_power_consumption.csv"

# Archivo preparado para reglas de asociación
ASSOCIATION_FILE = DATA_PROCESSED / "input_reglas_asociacion.csv"

# Columnas originales
DATE_COL = "Date"
TIME_COL = "Time"
DATETIME_COL = "datetime"

NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

NULL_TOKENS = ["?", ""]

# Decisiones de la práctica anterior

MAX_GAP_MINUTES = 60

NORMALIZATION_METHOD = "zscore"
TRAIN_FRACTION = 0.70

CORR_THRESHOLD = 0.90
CORR_PREFER_KEEP = ["Global_active_power"]

PCA_VARIANCE_TARGET = 0.90
PCA_PLOT_SAMPLE = 20_000
RANDOM_STATE = 42

# Columnas para reglas de asociación

DAY_COL = "Dia_Semana"
TIME_OF_DAY_COL = "Tiempo_Dia"
ACTIVE_ROOMS_COL = "Habitaciones_Activas"

ASSOCIATION_ITEM_COLS = [
    "Cocina",
    "Lavanderia",
    "Climatizacion",
]

# Rangos horarios utilizados para clasificar el tiempo del día
TIME_RANGES = {
    "Madrugada": (0, 5),
    "Mañana": (6, 11),
    "Mediodía": (12, 13),
    "Tarde": (14, 18),
    "Noche": (19, 23),
}

