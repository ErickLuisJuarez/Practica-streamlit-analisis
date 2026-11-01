"""
Visualización de los resultados de las reglas de asociación
"""

import pandas as pd
import matplotlib.pyplot as plt

from config import rutas as cfg


# Directorios donde se encuentran los resultados y se guardarán las gráficas
RESULTS_DIR = cfg.DATA_MODEL / "reglas_asociacion"

FIGURES_DIR = cfg.FIGURES / "reglas_asociacion"

# Archivo CSV de comparación
COMPARACION_FILE = (
    RESULTS_DIR / "comparacion_algoritmos.csv"
)

# Archivos CSV de reglas almacenados en data-processed
RULES_APRIORI_FILE = (
    cfg.DATA_PROCESSED / "reglas_apriori.csv"
)

RULES_FPGROWTH_FILE = (
    cfg.DATA_PROCESSED / "reglas_fpgrowth.csv"
)


def cargar_comparacion():
    """
    Carga los resultados de la comparación
    """

    # Comprueba que exista el archivo de comparación
    if not COMPARACION_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {COMPARACION_FILE}"
        )

    # Lee el CSV y lo convierte en un DataFrame
    return pd.read_csv(COMPARACION_FILE)


def cargar_reglas(ruta):
    """
    Carga las reglas generadas por un algoritmo
    """

    # Comprueba que exista el archivo de reglas solicitado
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}"
        )

    # Lee las reglas almacenadas en el archivo CSV
    return pd.read_csv(ruta)


def graficar_itemsets(df):
    """
    Compara la cantidad de itemsets frecuentes según el soporte mínimo
    """

    # Crea una figura donde se dibujarán los resultados
    plt.figure(figsize=(9, 5))

    # Dibuja la cantidad de itemsets encontrados por Apriori
    plt.plot(
        df["Soporte_Minimo"],
        df["Itemsets_Apriori"],
        marker="o",
        label="Apriori",
    )

    # Dibuja la cantidad de itemsets encontrados por FP-Growth
    plt.plot(
        df["Soporte_Minimo"],
        df["Itemsets_FPGrowth"],
        marker="s",
        label="FP-Growth",
    )

    # Define los nombres de los ejes y el título de la gráfica
    plt.xlabel("Soporte mínimo")
    plt.ylabel("Cantidad de itemsets frecuentes")
    plt.title("Itemsets frecuentes según el soporte mínimo")

    # Muestra las etiquetas que identifican cada algoritmo
    plt.legend()

    # Agrega una cuadrícula para facilitar la lectura de los valores
    plt.grid(True)

    # Ajusta los espacios para evitar que se encimen los elementos
    plt.tight_layout()

    # Guarda la gráfica como una imagen PNG
    plt.savefig(
        FIGURES_DIR / "itemsets_vs_soporte.png",
        dpi=300,
    )

    # Cierra la figura para liberar memoria
    plt.close()


def graficar_tiempos(df):
    """
    Compara los tiempos de ejecución de ambos algoritmos
    """

    # Crea una figura para representar los tiempos de ejecución
    plt.figure(figsize=(9, 5))

    # Dibuja los tiempos registrados para Apriori
    plt.plot(
        df["Soporte_Minimo"],
        df["Tiempo_Apriori"],
        marker="o",
        label="Apriori",
    )

    # Dibuja los tiempos registrados para FP-Growth
    plt.plot(
        df["Soporte_Minimo"],
        df["Tiempo_FPGrowth"],
        marker="s",
        label="FP-Growth",
    )

    # Define los nombres de los ejes y el título
    plt.xlabel("Soporte mínimo")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.title("Comparación de tiempos de ejecución")

    # Identifica cada línea mediante una leyenda
    plt.legend()

    # Agrega una cuadrícula para facilitar la comparación
    plt.grid(True)

    # Ajusta los espacios de la gráfica
    plt.tight_layout()

    # Guarda la gráfica generada
    plt.savefig(
        FIGURES_DIR / "tiempos_ejecucion.png",
        dpi=300,
    )

    # Cierra la figura después de guardarla
    plt.close()


def graficar_metricas(reglas_apriori, reglas_fpgrowth):
    """
    Genera histogramas comparativos de soporte, confianza y lift
    """

    # Define las métricas que se analizarán
    metricas = [
        "support",
        "confidence",
        "lift",
    ]

    # Recorre cada métrica para generar una gráfica independiente
    for metrica in metricas:

        # Crea una figura para representar la distribución de la métrica
        plt.figure(figsize=(9, 5))

        # Comprueba si Apriori generó reglas antes de dibujar su histograma
        if not reglas_apriori.empty:
            plt.hist(
                reglas_apriori[metrica],
                bins=20,
                alpha=0.6,
                label="Apriori",
            )

        # Comprueba si FP-Growth generó reglas
        if not reglas_fpgrowth.empty:
            plt.hist(
                reglas_fpgrowth[metrica],
                bins=20,
                alpha=0.6,
                label="FP-Growth",
            )

        # Define los nombres de los ejes y el título del histograma
        plt.xlabel(metrica.capitalize())
        plt.ylabel("Cantidad de reglas")
        plt.title(f"Distribución de {metrica}")

        # Muestra los algoritmos representados en los histogramas
        plt.legend()

        # Agrega una cuadrícula para facilitar la lectura
        plt.grid(True)

        # Ajusta los espacios de la figura
        plt.tight_layout()

        # Guarda el histograma utilizando el nombre de la métrica
        plt.savefig(
            FIGURES_DIR / f"distribucion_{metrica}.png",
            dpi=300,
        )

        # Cierra la figura antes de continuar con la siguiente métrica
        plt.close()


def main():
    """
    Ejecuta la generación de las gráficas
    """

    # Crea el directorio de las gráficas si todavía no existe
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Carga los resultados de la comparación de algoritmos
    comparacion = cargar_comparacion()

    # Carga las reglas obtenidas mediante Apriori
    reglas_apriori = cargar_reglas(
        RULES_APRIORI_FILE
    )

    # Carga las reglas obtenidas mediante FP-Growth
    reglas_fpgrowth = cargar_reglas(
        RULES_FPGROWTH_FILE
    )

    # Genera la gráfica de itemsets frecuentes según el soporte
    graficar_itemsets(comparacion)

    # Genera la gráfica que compara los tiempos de ejecución
    graficar_tiempos(comparacion)

    # Genera los histogramas de soporte, confianza y lift
    graficar_metricas(
        reglas_apriori,
        reglas_fpgrowth,
    )

    # Informa que las gráficas se generaron correctamente
    print("\nGráficas generadas correctamente")
    print(f"Directorio: {FIGURES_DIR}")


if __name__ == "__main__":
    # Ejecuta el programa cuando este archivo se inicia directamente
    main()