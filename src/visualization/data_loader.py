"""
Carga de datos para el dashboard 
"""
import json

import pandas as pd
import polars as pl
import streamlit as st

from config import rutas as cfg

#Archivos que lee el dashboard, los generó el pipeline
CLEAN_FILE = cfg.DATA_PROCESSED / "household_clean.parquet"
PREPARED_FILE = cfg.DATA_INPUT_MODEL / "household_prepared.parquet"
CORR_REPORT = cfg.DATA_INPUT_MODEL / "correlation_report.json"
PCA_REPORT = cfg.DATA_INPUT_MODEL / "pca_report.json"
LOADINGS_FILE = cfg.DATA_INPUT_MODEL / "pca_loadings.csv"


def processed_files_exist():
    """
    Revisa que ya se haya corrido el pipeline, es decir, que existan los dos parquet
    """
    return CLEAN_FILE.exists() and PREPARED_FILE.exists()  #Rregresa True solo si existen los dos archivos


@st.cache_data(show_spinner="Cargando dataset limpio…")  #Guardamos el resultado en memoria para no leer el archivo en cada clic
def load_clean():
    """
    Lee el dataset limpio en unidades originales (kW, V, A, Wh) que se usa para KPIs y gráficas
    """
    return pl.read_parquet(CLEAN_FILE).sort(cfg.DATETIME_COL)  #Leemos el parquet y lo ordena por fecha


@st.cache_data(show_spinner="Cargando dataset preparado…")  #Guardamos el resultado en memoria
def load_prepared():
    """
    Lee el dataset final con las variables normalizadas y los componentes principales
    """
    return pl.read_parquet(PREPARED_FILE).sort(cfg.DATETIME_COL)  #Leemos el parquet y lo ordena por fecha


@st.cache_data  #Guardamos el resultado en memoria
def load_json(path):
    """
    Lee un reporte json generado por el pipeline
    """
    with open(path, encoding="utf-8") as f:  #Abrimos el archivo
        return json.load(f)  #Lo convierte en diccionario


@st.cache_data  #Guardamos el resultado en memoria
def load_loadings():
    """
    Lee la tabla de loadings del PCA
    """
    return pd.read_csv(LOADINGS_FILE, index_col=0)  #La primera columna del csv son los nombres de las variables