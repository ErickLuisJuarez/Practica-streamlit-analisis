"""
Comparación de los algoritmos Apriori y FP-Growth
"""

import time
import pandas as pd

from config import rutas as cfg
from src.models import reglas_asociacion as ra


# Valores de soporte mínimo que se evaluarán en los algoritmos
SOPORTES_MINIMOS = [
    0.01, 0.02, 0.03, 0.05,
    0.10, 0.30, 0.50, 0.70,
]

# Directorio donde se almacenarán los resultados de la comparación
RESULTS_DIR = cfg.DATA_MODEL / "reglas_asociacion"

# Archivo que almacenará la comparación general de los algoritmos
COMPARACION_FILE = (
    RESULTS_DIR / "comparacion_algoritmos.csv"
)

# Archivo que almacenará los resultados individuales de cada algoritmo
DETALLE_FILE = (
    RESULTS_DIR / "detalle_comparacion.csv"
)


def medir_apriori(matriz, soporte):
    """
    Ejecuta Apriori y mide su tiempo de ejecución
    """

    # Registra el instante en que comienza la ejecución
    inicio = time.perf_counter()

    # Ejecuta Apriori utilizando el soporte mínimo indicado
    itemsets = ra.ejecutar_apriori(
        matriz,
        min_support=soporte,
    )

    # Calcula el tiempo transcurrido desde el inicio
    tiempo = time.perf_counter() - inicio

    # Devuelve los itemsets encontrados y el tiempo empleado
    return itemsets, tiempo


def medir_fpgrowth(matriz, soporte):
    """
    Ejecuta FP-Growth y mide su tiempo de ejecución
    """

    # Registra el instante en que comienza la ejecución
    inicio = time.perf_counter()

    # Ejecuta FP-Growth utilizando el soporte mínimo indicado
    itemsets = ra.ejecutar_fpgrowth(
        matriz,
        min_support=soporte,
    )

    # Calcula el tiempo transcurrido desde el inicio
    tiempo = time.perf_counter() - inicio

    # Devuelve los itemsets encontrados y el tiempo empleado
    return itemsets, tiempo


def comparar_itemsets(itemsets_apriori, itemsets_fpgrowth):
    """
    Compara los conjuntos frecuentes de ambos algoritmos
    """

    # Convierte los itemsets de Apriori en un conjunto de Python
    conjuntos_apriori = set(
        itemsets_apriori["itemsets"]
    )

    # Convierte los itemsets de FP-Growth en otro conjunto
    conjuntos_fpgrowth = set(
        itemsets_fpgrowth["itemsets"]
    )

    # Obtiene los itemsets que aparecen en ambos algoritmos
    comunes = conjuntos_apriori & conjuntos_fpgrowth

    # Obtiene los itemsets exclusivos de Apriori
    exclusivos_apriori = (
        conjuntos_apriori - conjuntos_fpgrowth
    )

    # Obtiene los itemsets exclusivos de FP-Growth
    exclusivos_fpgrowth = (
        conjuntos_fpgrowth - conjuntos_apriori
    )

    # Devuelve las cantidades y verifica si los resultados coinciden
    return {
        "Itemsets_Apriori": len(conjuntos_apriori),
        "Itemsets_FPGrowth": len(conjuntos_fpgrowth),
        "Itemsets_Comunes": len(comunes),
        "Solo_Apriori": len(exclusivos_apriori),
        "Solo_FPGrowth": len(exclusivos_fpgrowth),
        "Coinciden": (
            conjuntos_apriori == conjuntos_fpgrowth
        ),
    }


def ejecutar_comparacion(matriz):
    """
    Ejecuta la comparación para todos los soportes mínimos
    """

    # Listas que almacenarán los resultados generales y detallados
    resultados = []
    detalles = []

    # Recorre los valores de soporte mínimo definidos
    for soporte in SOPORTES_MINIMOS:
        print(f"\nEvaluando soporte mínimo: {soporte}")

        # Ejecuta Apriori y obtiene sus itemsets y tiempo de ejecución
        itemsets_apriori, tiempo_apriori = medir_apriori(
            matriz,
            soporte,
        )

        # Genera las reglas con la confianza mínima predeterminada
        reglas_apriori = ra.generar_reglas(
            itemsets_apriori,
        )

        # Ejecuta FP-Growth y obtiene sus itemsets y tiempo de ejecución
        itemsets_fpgrowth, tiempo_fpgrowth = medir_fpgrowth(
            matriz,
            soporte,
        )

        # Genera las reglas con la misma confianza mínima
        reglas_fpgrowth = ra.generar_reglas(
            itemsets_fpgrowth,
        )

        # Compara los conjuntos frecuentes obtenidos por ambos algoritmos
        comparacion = comparar_itemsets(
            itemsets_apriori,
            itemsets_fpgrowth,
        )

        # Agrega los tiempos, las reglas y los itemsets encontrados
        resultados.append({
            "Soporte_Minimo": soporte,
            "Tiempo_Apriori": tiempo_apriori,
            "Tiempo_FPGrowth": tiempo_fpgrowth,
            "Reglas_Apriori": len(reglas_apriori),
            "Reglas_FPGrowth": len(reglas_fpgrowth),
            **comparacion,
        })

        # Registra los resultados individuales de Apriori
        detalles.append({
            "Soporte_Minimo": soporte,
            "Algoritmo": "Apriori",
            "Tiempo_Ejecucion": tiempo_apriori,
            "Cantidad_Itemsets": len(itemsets_apriori),
            "Cantidad_Reglas": len(reglas_apriori),
        })

        # Registra los resultados individuales de FP-Growth
        detalles.append({
            "Soporte_Minimo": soporte,
            "Algoritmo": "FP-Growth",
            "Tiempo_Ejecucion": tiempo_fpgrowth,
            "Cantidad_Itemsets": len(itemsets_fpgrowth),
            "Cantidad_Reglas": len(reglas_fpgrowth),
        })

        # Muestra el tiempo empleado por cada algoritmo
        print(f"Apriori: {tiempo_apriori:.4f} segundos")
        print(f"FP-Growth: {tiempo_fpgrowth:.4f} segundos")

        # Indica si ambos algoritmos encontraron los mismos itemsets
        print(
            "Coinciden los itemsets:",
            comparacion["Coinciden"],
        )

    # Convierte las listas de resultados en DataFrames
    return (
        pd.DataFrame(resultados),
        pd.DataFrame(detalles),
    )


def guardar_comparacion(df_resultados, df_detalles):
    """
    Guarda los resultados de la comparación en archivos CSV
    """

    # Crea el directorio de resultados si todavía no existe
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exporta la comparación general sin agregar índices
    df_resultados.to_csv(
        COMPARACION_FILE,
        index=False,
    )

    # Exporta los resultados individuales de cada algoritmo
    df_detalles.to_csv(
        DETALLE_FILE,
        index=False,
    )

    # Muestra las ubicaciones de los archivos generados
    print("\nArchivos de comparación guardados:")
    print(COMPARACION_FILE)
    print(DETALLE_FILE)


def main():
    """
    Ejecuta la comparación completa
    """

    # Carga el dataset preparado para las reglas de asociación
    df = ra.cargar_datos()

    # Construye las transacciones con habitaciones y datos temporales
    transacciones = ra.construir_transacciones(df)

    # Convierte las transacciones en una matriz binaria de ítems
    matriz = ra.construir_matriz(transacciones)

    # Muestra la cantidad de transacciones y de ítems disponibles
    print(f"Transacciones: {len(transacciones):,}")
    print(f"Ítems: {matriz.shape[1]}")

    # Ejecuta la comparación con todos los soportes mínimos
    resultados, detalles = ejecutar_comparacion(matriz)

    # Guarda los resultados obtenidos en archivos CSV
    guardar_comparacion(resultados, detalles)

    # Informa que el proceso terminó
    print("\nComparación finalizada.")


if __name__ == "__main__":
    # Ejecuta el programa cuando este archivo se inicia directamente
    main()