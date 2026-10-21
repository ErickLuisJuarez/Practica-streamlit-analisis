"""
KPIs, agregaciones y tablas para el dashboard s
"""
import numpy as np
import pandas as pd
import polars as pl

from config import rutas as cfg

DT = cfg.DATETIME_COL  #Nombre corto de la columna de fecha
GAP = "Global_active_power"  #Potencia activa global en kW
SUBS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]  #Los 3 submedidores en Wh

#Nivel de agregación: lo que ve el usuario y el intervalo que usa Polars
AGGREGATIONS = {"Hora": "1h", "Día": "1d", "Semana": "1w", "Mes": "1mo"}
WEEKDAYS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]  #Nombres de los días de la semana


def filter_range(df, start, end):
    """
    Filtra el dataset por rango de fechas, incluye ambos extremos
    """
    return df.filter(pl.col(DT).dt.date().is_between(start, end))  #Se queda con las filas cuya fecha está entre start y end


def aggregate(df, every, cols, how="mean"):
    """
    Agrupa la serie por hora, día, semana o mes aplicando media, suma o máximo
    """
    exprs = [getattr(pl.col(c), how)() for c in cols]  #getattr(pl.col(c), "mean")() equivale a pl.col(c).mean()
    return df.group_by_dynamic(DT, every=every).agg(exprs)  #Una fila por cada ventana de tiempo


def kpis(df):
    """
    Calcula los indicadores clave del rango seleccionado
    """
    energia_kwh = df[GAP].sum() / 60  #Cada fila es 1 minuto, entonces kW por 1/60 de hora da kWh
    sub_kwh = {c: df[c].sum() / 1000 for c in SUBS}  #Los submedidores vienen en Wh, se pasan a kWh
    no_submedida = energia_kwh - sum(sub_kwh.values())  #Lo que consume la casa fuera de los 3 submedidores
    pico = df.sort(GAP, descending=True).head(1)  #La fila con la mayor potencia

    return {
        "registros": df.height, #Número de minutos
        "energia_total_kwh": energia_kwh,  #Energía total
        "potencia_media_kw": df[GAP].mean(),  #Potencia promedio
        "potencia_max_kw": df[GAP].max(),  #Potencia máxima
        "momento_pico": pico[DT][0],  #Fecha y hora de la potencia máxima
        "voltaje_medio_v": df["Voltage"].mean(),  #Voltaje promedio
        "energia_submedida_kwh": sub_kwh,  #Energía de cada submedidor
        "energia_no_submedida_kwh": no_submedida,  #Energía fuera de los submedidores
        "dias": df[DT].dt.date().n_unique(),  #Días distintos con datos
    }


def monthly_energy(df):
    """
    Energía en kWh por mes, con los días observados para detectar meses incompletos
    """
    return (
        df.group_by_dynamic(DT, every="1mo")  #Agrupamos por mes
        .agg((pl.col(GAP).sum() / 60).alias("kwh"), pl.len().alias("minutos"))  #Calcula los kWh y cuenta los minutos con dato
        .with_columns((pl.col("minutos") / 1440).alias("dias_observados"))  #1440 minutos son 1 día
    )


def weekday_hour_matrix(df, col=GAP):
    """
    Promedio de una variable por día de la semana (filas) y hora (columnas)
    """
    tabla = (
        df.with_columns(pl.col(DT).dt.weekday().alias("dow"), pl.col(DT).dt.hour().alias("hora"))  #Agregamos día de la semana y hora
        .group_by("dow", "hora")  #Un grupo por cada combinación de día y hora
        .agg(pl.col(col).mean().alias("v"))  #Promedio de la variable en cada grupo
        .to_pandas()  #Pasamos a pandas para poder hacer el pivot
        .pivot(index="dow", columns="hora", values="v")  #Filas son los días y columnas son las horas
        .sort_index()  #Ordenamos de lunes a domingo
    )
    tabla.index = [WEEKDAYS[i - 1] for i in tabla.index]  #Cambiamos los números 1 a 7 por Lun a Dom
    return tabla


def histogram(serie, bins=60):
    """
    Calcula el histograma de una serie y devuelve el centro de cada barra y su conteo
    """
    conteos, bordes = np.histogram(serie.to_numpy(), bins=bins)  #Contamos cuántos valores caen en cada intervalo
    return (bordes[:-1] + bordes[1:]) / 2, conteos  #El centro de cada barra es el promedio de sus dos bordes


def hourly_distribution(df, col=GAP, max_rows=150_000):
    """
    Devuelve una muestra con la hora del día para hacer boxplots por hora
    """
    muestra = df if df.height <= max_rows else df.sample(max_rows, seed=cfg.RANDOM_STATE)  #Si hay demasiadas filas toma una muestra para que sea rápido
    return muestra.select(pl.col(DT).dt.hour().alias("hora"), pl.col(col))  #Se queda con la hora y la variable


def sample(df, n=8_000):
    """
    Muestra aleatoria solo para graficar dispersión donde los KPIs usan todos los datos
    """
    return df if df.height <= n else df.sample(n, seed=cfg.RANDOM_STATE)  #Si hay más de n filas toma n al azar con semilla fija


def correlation(df, cols):
    """
    Matriz de correlación de Pearson de los datos filtrados
    """
    return pd.DataFrame(np.corrcoef(df.select(cols).to_numpy(), rowvar=False), index=cols, columns=cols)  #Calculamos la matriz y le pone los nombres de las variables


def detail_table(df, every, cols):
    """
    Tabla agregada con media, mínimo y máximo de cada variable
    """
    exprs = []  #Aquí se guardan las columnas que se van a calcular
    for c in cols:  #Para cada variable elegida
        exprs += [
            pl.col(c).mean().alias(f"{c}_media"),  #Media de la ventana
            pl.col(c).min().alias(f"{c}_min"),  #Mínimo de la ventana
            pl.col(c).max().alias(f"{c}_max"),  #Máximo de la ventana
        ]
    return df.group_by_dynamic(DT, every=every).agg(exprs).with_columns(pl.selectors.float().round(3))  #Agrupamos por ventana de tiempo y redondea a 3 decimales