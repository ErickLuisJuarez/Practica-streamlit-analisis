"""
Selección de variables con el coeficiente de correlación de Pearson

"""
import json

import matplotlib

matplotlib.use("Agg")  #Hace que matplotlib solo guarde imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from config import rutas as cfg

#Archivo que genera este módulo en data-input-model
CORR_REPORT = cfg.DATA_INPUT_MODEL / "correlation_report.json"


def pearson_matrix(df, cols):
    """
    Calcula la matriz de correlación de Pearson entre las columnas dadas
    """
    corr = np.corrcoef(df.select(cols).to_numpy(), rowvar=False)  #rowvar=False porque cada columna es una variable
    return pd.DataFrame(corr, index=cols, columns=cols)  #La convierte en tabla con los nombres de las variables


def plot_heatmap(corr, threshold, path=cfg.FIGURES / "correlacion_pearson.png"):
    """
    Guarda el mapa de calor de la matriz de correlación
    """
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)  #Creamos la carpeta de figuras si no existe
    fig, ax = plt.subplots(figsize=(8, 6.5))  #Creamos la figura con su tamaño
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)  #Dibujamos el mapa con el valor de r en cada celda y escala de -1 a 1
    ax.set_title(f"Correlación de Pearson (umbral de redundancia |r| ≥ {threshold})")  #Pone el título
    fig.tight_layout()  #Acomodamos los márgenes
    fig.savefig(path, dpi=150)  #Guardamos la imagen
    plt.close(fig)  #Cerramos la figura para liberar memoria


def high_correlation_pairs(corr, threshold):
    """
    Lista los pares de variables con |r| mayor o igual al umbral de mayor a menor correlación
    """
    cols = list(corr.columns)  #Nombres de las variables
    pares = []  #Guardamos los pares encontrados
    for i, a in enumerate(cols):  #Recorremos cada variable
        for b in cols[i + 1:]:  #Comparamos solo con las que van después, para no repetir pares
            if abs(corr.loc[a, b]) >= threshold:  #Valor absoluto porque una correlación de -0.95 también es redundante
                pares.append({"var_a": a, "var_b": b, "r": float(corr.loc[a, b])})  #Guardamos el par y su correlación
    return sorted(pares, key=lambda p: -abs(p["r"]))  #Ordenamos de mayor a menor correlación


def select_variables(corr, threshold, prefer_keep):
    """
    Decide qué variables conservar y devuelve las conservadas y las decisiones con su razón
    """
    conservar = list(corr.columns)  #Al inicio se conservan todas
    media_abs = (corr.abs().sum() - 1) / (len(corr) - 1)  #Correlación media absoluta de cada variable con las demás 
    decisiones = []  #Se guarda que se descartó y por qué

    for par in high_correlation_pairs(corr, threshold):  #Recorremos los pares redundantes
        a, b = par["var_a"], par["var_b"]  #Las dos variables del par
        if a not in conservar or b not in conservar:  #Si alguna ya se descartó
            continue  #Ese par ya quedo resuelto y se salta

        if a in prefer_keep and b not in prefer_keep:  #Si a es la preferida
            descartar, razon = b, f"se conserva {a} (variable preferida)"  #Se descarta b
        elif b in prefer_keep and a not in prefer_keep:  #Si b es la preferida
            descartar, razon = a, f"se conserva {b} (variable preferida)"  #Se descarta a
        else:  #Si ninguna es preferida
            descartar = a if media_abs[a] >= media_abs[b] else b  #Se descarta la que más se parece al resto de variables
            razon = "se descarta la de mayor correlación media con el resto"

        conservar.remove(descartar)  #Quitamos la variable descartada de la lista
        decisiones.append({**par, "descartada": descartar, "razon": razon})  #Guardamos la decisión con su razon
    return conservar, decisiones


def run(df, cols=cfg.NUMERIC_COLS):
    """
    Ejecuta el Paso 4: matriz, mapa de calor, selección y reporte
    """
    train = df.filter(pl.col("split") == "train")  #Usamos solo train para no mirar el futuro
    corr = pearson_matrix(train, cols)  #Calculamos la matriz de correlación
    plot_heatmap(corr, cfg.CORR_THRESHOLD)  #Guardamos el mapa de calor

    conservar, decisiones = select_variables(corr, cfg.CORR_THRESHOLD, cfg.CORR_PREFER_KEEP)  #Decide quw variables se quedan
    reporte = {  #Armamos el reporte del paso
        "umbral": cfg.CORR_THRESHOLD,
        "pares_altamente_correlacionados": high_correlation_pairs(corr, cfg.CORR_THRESHOLD),
        "decisiones": decisiones,
        "conservadas": conservar,
        "descartadas": [c for c in cols if c not in conservar],
    }
    CORR_REPORT.write_text(json.dumps(reporte, indent=2), encoding="utf-8")  #Guardamos el reporte, lo lee el dashboard
    print(f"[Paso 4] pares con |r| >= {cfg.CORR_THRESHOLD}: {len(reporte['pares_altamente_correlacionados'])}")  #Muestra cuántos pares son redundantes
    print(f"[Paso 4] conservadas: {conservar} | descartadas: {reporte['descartadas']}")  #Muestra qué se conservó y qué se descartó
    return conservar, corr