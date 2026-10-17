"""
Entrada: data/data-raw/household_power_consumption.csv
Salida:  data/data-processed/household_clean.parquet y cleaning_report.json
"""
import json
from pathlib import Path

import polars as pl

from config import rutas as cfg

# Caracteres basura que trae el archivo: comillas, saltos de línea, espacios y BOM
BASURA = '"\r\n \ufeff'

# Archivos que genera este módulo (en data-processed)
CLEAN_FILE = cfg.DATA_PROCESSED / "household_clean.parquet"
CLEANING_REPORT = cfg.DATA_PROCESSED / "cleaning_report.json"


def find_raw_file():
    """Devuelve la ruta del dataset original y avisa si no está en data-raw"""
    if not cfg.RAW_FILE.exists(): #Si el archivo no está, falla y deja un mensaje
        raise FileNotFoundError(
            f"No se encontró el dataset en {cfg.RAW_FILE}. Descárgalo de la UCI y colócalo ahí con ese nombre"
        )
    return cfg.RAW_FILE


def load_raw(path=None):
    """Lee el archivo original dejando TODAS las columnas como texto
    """
    path = Path(path) if path else find_raw_file()
    #quote_char=None porque la versión .csv trae cada línea entre comillas
    #infer_schema=False deja todo como texto
    df = pl.read_csv(path, separator=";", quote_char=None, infer_schema=False)
    return df.rename({c: c.strip(BASURA) for c in df.columns}) #Los nombres de columna también pueden traer comillas o saltos de línea entonces se limpian


def inspect_raw(df):
    """Resume el estado inicial como filas, tipos, faltantes, rango de fechas y descriptivos"""
    tmp = normalize_text_columns(df) #Los ? se cuentan como nulo después de esto
    nulos = {c: int(tmp[c].null_count()) for c in tmp.columns} #Nulos por columna

    #Se arma la fecha solo para conocer el rango temporal
    fecha = pl.concat_str([pl.col(cfg.DATE_COL), pl.col(cfg.TIME_COL)], separator=" ").str.to_datetime(
        "%d/%m/%Y %H:%M:%S", strict=False #Si una fecha falla queda nula en vez de que truene
    )
    tmp = tmp.with_columns(fecha.alias(cfg.DATETIME_COL)) #Agrega la columna datetime
    numericas = tmp.select(pl.col(cfg.NUMERIC_COLS).cast(pl.Float64, strict=False))  #Para los estadísticos descriptivos las medidas se pasan a número

    return {
        "filas": df.height, #Número de filas
        "columnas": df.width, #Número de columnas
        "tipos_detectados_en_lectura": {c: str(t) for c, t in df.schema.items()}, #Tipo de dato de cada columna
        "faltantes_por_columna": nulos, #Cuántos nulos tiene cada columna
        "faltantes_pct": {c: round(100 * n / df.height, 3) for c, n in nulos.items()}, #Porcentaje de nulos por columna
        "fecha_inicio": str(tmp[cfg.DATETIME_COL].min()), #Primera fecha del dataset
        "fecha_fin": str(tmp[cfg.DATETIME_COL].max()), #Última fecha del dataset
        "descriptivos": numericas.describe().to_dicts(), #Media, desviación, mínimo, máximo y cuartiles
    }


def normalize_text_columns(df):
    """Normaliza las columnas de texto sin espacios de más, sin acentos y en minúsculas
    """
    cols_texto = [c for c, t in df.schema.items() if t == pl.String] #Solo las columnas que son texto
    df = df.with_columns(
        pl.col(cols_texto)
        .str.strip_chars(BASURA) #Quita comillas, saltos de línea y espacios de las orillas
        .str.replace_all(r"\s+", " ") #Varios espacios seguidos pasan a uno solo
        .str.normalize("NFD") #Separa la letra de su acento
        .str.replace_all(r"\p{M}", "") #Luego quita el acento
        .str.to_lowercase() #Todo en minúsculas
    )
    #? y '' pasan a ser nulo
    return df.with_columns(
        [pl.when(pl.col(c).is_in(cfg.NULL_TOKENS)).then(None).otherwise(pl.col(c)).alias(c) for c in cols_texto]
    )


def deduplicate(df):
    """Elimina duplicados exactos y fechas repetidas y devuelve el df y cuántos quitó de cada tipo"""
    n0 = df.height #Filas antes de quitar duplicados
    df = df.unique(maintain_order=True) #Quita filas idénticas respetando el orden original
    exactos = n0 - df.height #Cuántas filas idénticas se quitaron

    #Dos filas con la misma fecha y hora son el mismo minuto registrado dos veces
    n1 = df.height #Filas después de quitar los idénticos
    df = df.unique(subset=[cfg.DATE_COL, cfg.TIME_COL], keep="first", maintain_order=True) #Si se repite la fecha y hora se queda con la primera
    return df, {"duplicados_exactos": exactos, "timestamps_repetidos": n1 - df.height}


def cast_types(df):
    """
    Convierte Date y Time a una sola columna datetime y las medidas a Float64
    """
    return (
        df.with_columns(
            pl.concat_str([pl.col(cfg.DATE_COL), pl.col(cfg.TIME_COL)], separator=" ") #Une "16/12/2006" y "17:24:00" y los convierte en una fecha-hora real
            .str.to_datetime("%d/%m/%Y %H:%M:%S", strict=False) #Convierte el texto a fecha-hora con ese formato
            .alias(cfg.DATETIME_COL), #Nombre de la columna nueva
            pl.col(cfg.NUMERIC_COLS).cast(pl.Float64, strict=False), #Las 7 medidas pasan de texto a número decimal
        )
        .drop([cfg.DATE_COL, cfg.TIME_COL])  #Ya no sirven las columnas originales
        .select([cfg.DATETIME_COL, *cfg.NUMERIC_COLS]) #datetime primero, luego las medidas
        .sort(cfg.DATETIME_COL) #En una serie de tiempo el orden importa
    )


def handle_missing(df, max_gap=cfg.MAX_GAP_MINUTES):
    """
    Trata los faltantes de la serie de tiempo
    """
    n_antes = df.height #Filas antes de revisar la rejilla de minutos
    df = df.upsample(time_column=cfg.DATETIME_COL, every="1m") #Garantiza una fila por minuto
    filas_agregadas = df.height - n_antes #Cuántas filas se agregaron para completar la rejilla

    #Para cada fila nula calculamos de cuántos minutos seguidos es su hueco
    hay_nulo = pl.any_horizontal(pl.col(cfg.NUMERIC_COLS).is_null())  #True si falta alguna medida
    id_racha = (hay_nulo != hay_nulo.shift(1)).fill_null(True).cast(pl.Int64).cum_sum() #Cambia cada vez que se pasa de "hay dato" a "no hay dato"
    largo_hueco = pl.when(hay_nulo).then(pl.len().over(id_racha)).otherwise(0) #Cuántas filas seguidas tiene la racha de nulos a la que pertenece la fila
    df = df.with_columns(largo_hueco.alias("_hueco")) #Guarda el largo del hueco en una columna auxiliar

    #Conteos para el reporte
    faltantes = int((df["_hueco"] > 0).sum()) #Minutos con algún dato faltante
    cortos = int(((df["_hueco"] > 0) & (df["_hueco"] <= max_gap)).sum()) #Minutos en huecos que sí se rellenan
    largos = int((df["_hueco"] > max_gap).sum()) #Minutos en huecos demasiado largos

    df = df.with_columns(pl.col(cfg.NUMERIC_COLS).interpolate())  #Rellena los huecos en línea recta
    df = df.filter(pl.col("_hueco") <= max_gap).drop("_hueco").drop_nulls(cfg.NUMERIC_COLS) #Se tiran las filas de huecos largos y se quita la columna auxiliar

    return df, {
        "filas_agregadas_por_rejilla": filas_agregadas,
        "minutos_faltantes_totales": faltantes,
        "interpolados_(huecos<=max_gap)": cortos,
        "eliminados_(huecos>max_gap)": largos,
        "max_gap_minutos": max_gap,
    }


def clean(raw):
    """
    Aplica todo el preprocesamiento en orden y devuelve el df limpio y un reporte
    """
    reporte = {"filas_originales": raw.height} #Empieza el reporte con las filas originales
    df = normalize_text_columns(raw)   #Texto limpio y ? se convierte en nulo
    df, rep_dup = deduplicate(df) #Sin duplicados
    df = cast_types(df) #Tipos correctos (fecha y números)
    reporte["fallos_de_casteo_fecha"] = int(df[cfg.DATETIME_COL].null_count()) #Cuántas fechas no se pudieron convertir
    df = df.drop_nulls(cfg.DATETIME_COL) #Una fila sin fecha válida no sirve en una serie de tiempo
    df, rep_faltantes = handle_missing(df) #Trata los faltantes
    reporte |= rep_dup | rep_faltantes #Junta los reportes en uno solo
    reporte["filas_finales"] = df.height #Filas que quedaron al final
    return df, reporte


def run():
    """
    Ejecuta los Pasos 1 y 2 y guarda el dataset limpio en data-processed
    """
    raw = load_raw() #Carga el csv crudo
    inspeccion = inspect_raw(raw) #Inspecciona el estado inicial
    df_limpio, reporte = clean(raw) #Limpia el dataset y arma el reporte

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True) #Crea la carpeta data-processed si no existe
    df_limpio.write_parquet(CLEAN_FILE, compression="zstd") #Guarda el dataset limpio en parquet
    CLEANING_REPORT.write_text( #Guarda el reporte en un archivo json
        json.dumps({"inspeccion_inicial": inspeccion, "limpieza": reporte}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[Paso 1] {inspeccion['filas']:,} filas | {inspeccion['fecha_inicio']} -> {inspeccion['fecha_fin']}") #Muestra filas y rango de fechas
    print(f"[Paso 1] faltantes (%): {inspeccion['faltantes_pct']}") #Muestra el porcentaje de faltantes
    print(f"[Paso 2] {json.dumps(reporte, indent=2)}") #Muestra el reporte de limpieza
    return df_limpio


if __name__ == "__main__":
    run() #Solo corre si se ejecuta este archivo directamente