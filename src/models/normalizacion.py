"""
Normalización de las variables numéricas (z-score o min-max)
"""
import json

import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import rutas as cfg

#Archivo que genera este módulo en data-input-model
SCALER_FILE = cfg.DATA_INPUT_MODEL / "scaler_params.json"


def split_cutoff(df, train_fraction=cfg.TRAIN_FRACTION):
    """

    Devuelve la fecha de corte entre train y test y el número de filas de train"""
    df = df.sort(cfg.DATETIME_COL)  #El corte se hace por orden cronologico
    n_train = int(df.height * train_fraction) #Número de filas de train
    return df[cfg.DATETIME_COL][n_train - 1], n_train  #Última fecha que entra a train y el número de filas


def fit_scaler(train, cols, method=cfg.NORMALIZATION_METHOD):
    """
    Ajusta el escalador (z-score o min-max) usando únicamente los datos de train
    """
    escalador = StandardScaler() if method == "zscore" else MinMaxScaler()  #Se elige la tecnica
    escalador.fit(train.select(cols).to_numpy())  #Se calculan media y desviación 
    return escalador


def apply_scaler(df, escalador, cols):
    """
    Aplica un escalador ya ajustado a las columnas indicadas
    """
    escalado = escalador.transform(df.select(cols).to_numpy())  #Usamos los parametros de train
    return df.with_columns([pl.Series(c, escalado[:, i]) for i, c in enumerate(cols)]) #Reemplazamos cada columna original por su versión escalada


def scaler_params(escalador, cols, method, cutoff):
    """
    Junta los parámetros del escalador en un diccionario para guardarlos en JSON
    """
    if method == "zscore":
        #z-score: (x - media) / desviación
        params = {"mean": dict(zip(cols, escalador.mean_.tolist())), "std": dict(zip(cols, escalador.scale_.tolist()))}
    else:
        #min-max: (x - mín) / (máx - mín)
        params = {"min": dict(zip(cols, escalador.data_min_.tolist())), "max": dict(zip(cols, escalador.data_max_.tolist()))}
    return {"method": method, "ajustado_con_datos_hasta": str(cutoff), **params}


def normalize(df, cols=cfg.NUMERIC_COLS):
    """
    Normaliza las columnas y agrega la columna split (train/test) y G¿guarda los parámetros
    """
    df = df.sort(cfg.DATETIME_COL)  #Ordenamos por fecha
    cutoff, n_train = split_cutoff(df)  #Fecha de corte y filas de train

    escalador = fit_scaler(df.head(n_train), cols) #El escalador se ajusta solo con train sin mirar el futuro
    resultado = apply_scaler(df, escalador, cols) #Se aplica a todo el dataset

    #Marca cada fila como train  o test
    resultado = resultado.with_columns(
        pl.when(pl.int_range(pl.len()) < n_train).then(pl.lit("train")).otherwise(pl.lit("test")).alias("split")
    )
    params = scaler_params(escalador, cols, cfg.NORMALIZATION_METHOD, cutoff)  #Parámetros que se van a guardar
    cfg.DATA_INPUT_MODEL.mkdir(parents=True, exist_ok=True)  #Creamos la carpeta si no existe
    SCALER_FILE.write_text(json.dumps(params, indent=2), encoding="utf-8")  #Guardams los parámetros que sirven para escalar datos nuevos
    return resultado, params


def describe_after(df, cols):
    """
    Comprobación media y desviación de train después de normalizar (z-score: cerca de 0 y 1)
    """
    train = df.filter(pl.col("split") == "train").select(cols).to_numpy()  #Solo las filas de train
    return {"media_train": np.round(train.mean(0), 6).tolist(), "std_train": np.round(train.std(0), 6).tolist()}