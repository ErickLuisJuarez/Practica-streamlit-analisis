"""
Reducción de dimensionalidad con PCA
"""
import json

import matplotlib

matplotlib.use("Agg")  #Hace que matplotlib solo guarde imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import rutas as cfg

#Archivos que genera este módulo en data-input-model
PCA_REPORT = cfg.DATA_INPUT_MODEL / "pca_report.json"
LOADINGS_FILE = cfg.DATA_INPUT_MODEL / "pca_loadings.csv"


def ensure_standardized(df, cols, tol=0.05):
    """
    Revisa que las variables estén estandarizadas y si no, las estandariza
    """
    X_todo = df.select(cols).to_numpy()  #Todas las filas como matriz
    X_train = df.filter(pl.col("split") == "train").select(cols).to_numpy()  #Solo las filas de train

    #Estandarizado significa media cerca de 0 y desviación cerca de 1 en cada columna
    ya_estaba = bool(np.all(np.abs(X_train.mean(0)) < tol) and np.all(np.abs(X_train.std(0) - 1) < tol))
    if ya_estaba:  #Si ya lo estaba
        return X_train, X_todo, True  #Se devuelven tal cual

    escalador = StandardScaler().fit(X_train)  #Si no lo estaba  se ajusta con train
    return escalador.transform(X_train), escalador.transform(X_todo), False  #Y se estandarizan


def choose_n_components(pca, target=cfg.PCA_VARIANCE_TARGET):
    """
    Elige cuántos componentes conservar según la varianza acumulada y lo compara con Kaiser
    """
    acumulada = np.cumsum(pca.explained_variance_ratio_)  #Varianza acumulada componente por componente
    k_varianza = int(np.searchsorted(acumulada, target) + 1)  #Primer k que llega al objetivo de varianza
    k_kaiser = int((pca.explained_variance_ > 1).sum())  #Criterio de Kaiser: componentes con valor propio mayor a 1
    return {
        "k_por_varianza_acumulada": k_varianza,
        "k_kaiser": k_kaiser,
        "objetivo_varianza": target,
        "k_elegido": k_varianza,  #Aquí se decide con que criterio nos quedamos
    }


def plot_scree(pca, k, path=cfg.FIGURES / "pca_scree.png"):
    """
    Guarda el scree plot con la varianza explicada y la acumulada
    """
    ratio = pca.explained_variance_ratio_  #Varianza explicada por cada componente
    acumulada = np.cumsum(ratio)  #Varianza acumulada
    x = np.arange(1, len(ratio) + 1)  #Números de componente: 1, 2, 3...

    fig, ax = plt.subplots(figsize=(8, 5))  #Creamos la figura
    ax.bar(x, ratio, color="#B8C4CC", label="Varianza explicada")  #Barras: varianza de cada componente
    ax.plot(x, acumulada, "o-", color="#E4572E", label="Acumulada")  #Línea: varianza acumulada
    ax.axhline(cfg.PCA_VARIANCE_TARGET, ls="--", color="gray", lw=1)  #Línea horizontal en el objetivo de varianza
    ax.axvline(k, ls=":", color="#0F2A3D", lw=1.5, label=f"k = {k}")  #Línea vertical en los componentes elegidos
    ax.set(xlabel="Componente principal", ylabel="Proporción de varianza", xticks=x, ylim=(0, 1.02))  #Etiquetas de los ejes y límites
    ax.set_title("Scree plot")  #Título
    ax.legend()  #Mostrmaos la leyenda
    fig.tight_layout()  #Acomodamos los márgenes
    fig.savefig(path, dpi=150)  #Guardamos la imagen
    plt.close(fig)  #Cerramos la figura para liberar memoria


def plot_projection(scores, horas, path=cfg.FIGURES / "pca_2d.png"):
    """
    Guarda la proyección PC1 contra PC2 de una muestra de puntos, coloreada por hora del día
    """
    rng = np.random.default_rng(cfg.RANDOM_STATE)  #Generador aleatorio con semilla fija para que siempre salga la misma muestra
    idx = rng.choice(len(scores), size=min(cfg.PCA_PLOT_SAMPLE, len(scores)), replace=False)  #Elegimos los puntos de la muestra 

    fig, ax = plt.subplots(figsize=(8, 6))  #Creamos la figura
    puntos = ax.scatter(scores[idx, 0], scores[idx, 1], c=horas[idx], cmap="viridis", s=4, alpha=0.5)  #Dibujamos los puntos coloreados por hora
    fig.colorbar(puntos, label="Hora del día")  #Barra de colores
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Proyección PCA en 2D (muestra de {len(idx):,} puntos)")  #Ejes y título
    fig.tight_layout()  #Acomodamos los márgenes
    fig.savefig(path, dpi=150)  #Guardamos la imagen
    plt.close(fig)  #Cerramos la figura para liberar memoria


def run(df, keep_cols):
    """Ejecuta el Paso 5 y devuelve el dataset con las columnas PC1 a PCk agregadas"""
    X_train, X_todo, estaba_estandarizado = ensure_standardized(df, keep_cols)  #Revisamos o aplicamos la estandarización

    pca = PCA(random_state=cfg.RANDOM_STATE).fit(X_train)  #El PCA se ajusta solo con train
    eleccion = choose_n_components(pca)  #Decidimos cuántos componentes conservar
    k = eleccion["k_elegido"]  #Número de componentes elegidos

    cfg.FIGURES.mkdir(parents=True, exist_ok=True)  #Creamos la carpeta de figuras si no existe
    plot_scree(pca, k)  #Guardamos el scree plot

    nombres = [f"PC{i + 1}" for i in range(len(keep_cols))]  #Nombres de los componentes: PC1, PC2,etc
    loadings = pd.DataFrame(pca.components_.T, index=keep_cols, columns=nombres)  #Loadings es cuánto pesa cada variable original en cada componente
    loadings.to_csv(LOADINGS_FILE)  #Guardamos los loadings en csv

    scores = pca.transform(X_todo)  #Proyectamos todos los datos sobre los componentes
    plot_projection(scores, df[cfg.DATETIME_COL].dt.hour().to_numpy())  #Guardamos la proyección 2D coloreada por hora

    reporte = {  #Armamos el reporte del paso
        "variables_usadas": keep_cols,
        "ya_estaba_estandarizado": estaba_estandarizado,
        "valores_propios": pca.explained_variance_.tolist(),
        "varianza_explicada": pca.explained_variance_ratio_.tolist(),
        "varianza_acumulada": np.cumsum(pca.explained_variance_ratio_).tolist(),
        **eleccion,
    }
    PCA_REPORT.write_text(json.dumps(reporte, indent=2), encoding="utf-8")  #Guardamos el reporte, lo lee el dashboard
    print(f"[Paso 5] estandarizado previamente: {estaba_estandarizado} | k elegido: {k} "
          f"(varianza acumulada {reporte['varianza_acumulada'][k - 1]:.3f}; Kaiser sugiere {eleccion['k_kaiser']})")  #Muestra el resumen del PCA
    print(loadings.round(3).iloc[:, :k].to_string())  #Muestra los loadings de los componentes retenidos

    return df.with_columns([pl.Series(nombres[i], scores[:, i]) for i in range(k)])  #Agregamos solo los k componentes retenidos al dataset