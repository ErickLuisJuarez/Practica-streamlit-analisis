"""
Reglas de Asociación sobre consumo eléctrico
"""

import polars as pl

from config import rutas as cfg


# Días de la semana en español
DIAS_SEMANA = {
    1: "Lunes",
    2: "Martes",
    3: "Miércoles",
    4: "Jueves",
    5: "Viernes",
    6: "Sábado",
    7: "Domingo",
}


def cargar_datos():
    """
    Carga el CSV original y limpia los valores de texto
    """

    if not cfg.RAW_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo original: {cfg.RAW_FILE}"
        )

    df = pl.read_csv(
        cfg.RAW_FILE,
        separator=";",
        quote_char=None,
        infer_schema=False,
    )

    # Limpia espacios, comillas y caracteres sobrantes
    caracteres = '"\r\n \ufeff'

    df = df.rename({
        columna: columna.strip(caracteres)
        for columna in df.columns
    })

    # Convierte las variables numéricas y los valores faltantes
    df = df.with_columns([
        pl.col(columna)
        .str.strip_chars(caracteres)
        .replace("?", None)
        .cast(pl.Float64, strict=False)
        .alias(columna)
        for columna in cfg.NUMERIC_COLS
    ])

    # Construye la fecha y hora completas
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col(cfg.DATE_COL).str.strip_chars(caracteres),
                pl.col(cfg.TIME_COL).str.strip_chars(caracteres),
            ],
            separator=" ",
        )
        .str.to_datetime(
            format="%d/%m/%Y %H:%M:%S",
            strict=False,
        )
        .alias(cfg.DATETIME_COL)
    )

    # Descarta registros sin fecha válida
    df = df.filter(
        pl.col(cfg.DATETIME_COL).is_not_null()
    )

    return df


def construir_transacciones(df):
    """
    Agrupa los registros minuto a minuto en transacciones de una hora
    """

    # Ordena los registros cronológicamente
    df = df.sort(cfg.DATETIME_COL)

    # Métricas globales: promedio por hora
    promedios = [
        pl.col(columna).mean().alias(columna)
        for columna in cfg.NUMERIC_COLS[:4]
    ]

    # Submediciones: consumo acumulado por hora
    sumas = [
        pl.col(columna).sum().alias(columna)
        for columna in cfg.NUMERIC_COLS[4:]
    ]

    # Agrupa los registros de cada hora
    df = df.group_by_dynamic(
        cfg.DATETIME_COL,
        every="1h",
        period="1h",
    ).agg(promedios + sumas)

    # Descarta las horas sin mediciones válidas
    df = df.filter(
        pl.col("Global_active_power").is_not_null()
    )

    # Determina si cada habitación estuvo activa
    df = df.with_columns([
        (pl.col("Sub_metering_1") > 0).alias("Cocina"),

        (pl.col("Sub_metering_2") > 0).alias("Lavanderia"),

        (pl.col("Sub_metering_3") > 0).alias("Climatizacion"),
    ])

    return df


def agregar_variables_temporales(df):
    """
    Agrega el día de la semana y el tiempo del día
    """

    df = df.with_columns([
        pl.col(cfg.DATETIME_COL)
        .dt.weekday()
        .replace_strict(DIAS_SEMANA)
        .alias(cfg.DAY_COL),

        pl.col(cfg.DATETIME_COL)
        .dt.hour()
        .alias("hora"),
    ])

    # Construye la clasificación a partir de los rangos de configuración
    expresion_tiempo = None

    for nombre, (hora_inicio, hora_fin) in cfg.TIME_RANGES.items():
        condicion = pl.col("hora").is_between(
            hora_inicio,
            hora_fin,
        )

        if expresion_tiempo is None:
            expresion_tiempo = pl.when(condicion).then(
                pl.lit(nombre)
            )
        else:
            expresion_tiempo = expresion_tiempo.when(
                condicion
            ).then(
                pl.lit(nombre)
            )

    # Asigna la franja horaria correspondiente
    df = df.with_columns(
        expresion_tiempo
        .otherwise(None)
        .alias(cfg.TIME_OF_DAY_COL)
    )

    # Construye el conjunto de habitaciones activas
    df = df.with_columns(
        pl.concat_list([
            pl.when(pl.col("Cocina"))
            .then(pl.lit("Cocina"))
            .otherwise(None),

            pl.when(pl.col("Lavanderia"))
            .then(pl.lit("Lavandería"))
            .otherwise(None),

            pl.when(pl.col("Climatizacion"))
            .then(pl.lit("Climatización"))
            .otherwise(None),
        ])
        .list.drop_nulls()
        .list.join("; ")
        .alias(cfg.ACTIVE_ROOMS_COL)
    )

    # Elimina la columna auxiliar
    df = df.drop("hora")

    return df


def guardar_dataset(df):
    """
    Guarda el dataset preparado en data-processed
    """

    cfg.DATA_PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.write_csv(cfg.ASSOCIATION_FILE)

    print("\nDataset preparado correctamente")
    print(f"Archivo: {cfg.ASSOCIATION_FILE}")
    print(f"Filas: {df.height:,}")
    print(f"Columnas: {df.width}")
    print(f"Variables: {df.columns}")


def main():
    """
    Ejecuta la preparación completa
    """

    print("Cargando dataset original...")
    df = cargar_datos()

    print("Construyendo transacciones horarias...")
    df = construir_transacciones(df)

    print("Agregando variables temporales...")
    df = agregar_variables_temporales(df)

    guardar_dataset(df)


if __name__ == "__main__":
    main()