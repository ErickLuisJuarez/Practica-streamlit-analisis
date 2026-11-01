"""
Minería de reglas de asociación mediante Apriori y FP-Growth
"""

import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

from config import rutas as cfg


# Parámetros iniciales de los algoritmos
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.50
MIN_LIFT = 1.10

# Directorios donde se guardarán los resultados
RESULTS_DIR = cfg.DATA_MODEL / "reglas_asociacion"

ITEMSETS_APRIORI_FILE = (
    RESULTS_DIR / "itemsets_apriori.csv"
)

ITEMSETS_FPGROWTH_FILE = (
    RESULTS_DIR / "itemsets_fpgrowth.csv"
)

# Las reglas de asociación se guardan en data-processed
RULES_APRIORI_FILE = (
    cfg.DATA_PROCESSED / "reglas_apriori.csv"
)

RULES_FPGROWTH_FILE = (
    cfg.DATA_PROCESSED / "reglas_fpgrowth.csv"
)


def cargar_datos():
    """
    Carga el dataset preparado para reglas de asociación
    """

    if not cfg.ASSOCIATION_FILE.exists():
        raise FileNotFoundError(
            "No se encontró el dataset preparado: "
            f"{cfg.ASSOCIATION_FILE}"
        )

    df = pd.read_csv(cfg.ASSOCIATION_FILE)

    columnas_necesarias = [
        cfg.DAY_COL,
        cfg.TIME_OF_DAY_COL,
        *cfg.ASSOCIATION_ITEM_COLS,
        cfg.ACTIVE_ROOMS_COL,
    ]

    columnas_faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in df.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas en el dataset: "
            f"{columnas_faltantes}"
        )

    if df.empty:
        raise ValueError("El dataset no contiene registros")

    print(f"Registros cargados: {len(df):,}")

    return df


def esta_activo(valor):
    """
    Comprueba si una habitación estuvo activa
    """

    if pd.isna(valor):
        return False

    if isinstance(valor, str):
        return valor.strip().lower() in (
            "true", "1", "si", "sí"
        )

    return bool(valor)


def construir_transacciones(df):
    """
    Construye una transacción por cada hora

    Cada transacción contiene el día de la semana,
    el tiempo del día y las habitaciones activas
    """

    transacciones = []

    for _, fila in df.iterrows():
        transaccion = []

        # Agrega el día de la semana
        dia = fila[cfg.DAY_COL]

        if pd.notna(dia):
            transaccion.append(f"Dia={dia}")

        # Agrega el tiempo del día
        tiempo = fila[cfg.TIME_OF_DAY_COL]

        if pd.notna(tiempo):
            transaccion.append(f"Tiempo={tiempo}")

        # Agrega las habitaciones activas
        habitaciones = fila[cfg.ACTIVE_ROOMS_COL]

        if pd.notna(habitaciones):
            for habitacion in str(habitaciones).split(";"):
                habitacion = habitacion.strip()

                if habitacion:
                    transaccion.append(habitacion)

        transacciones.append(transaccion)

    return transacciones


def construir_matriz(transacciones):
    """
    Convierte las transacciones en una matriz booleana
    """

    codificador = TransactionEncoder()

    matriz_codificada = codificador.fit(
        transacciones
    ).transform(transacciones)

    df_matriz = pd.DataFrame(
        matriz_codificada,
        columns=codificador.columns_,
    )

    print(f"Cantidad de ítems: {df_matriz.shape[1]}")

    return df_matriz


def ejecutar_apriori(df_matriz, min_support=MIN_SUPPORT):
    """
    Ejecuta Apriori y obtiene los itemsets frecuentes
    """

    itemsets = apriori(
        df_matriz,
        min_support=min_support,
        use_colnames=True,
    )

    itemsets = itemsets.sort_values(
        by="support",
        ascending=False,
    ).reset_index(drop=True)

    print(f"Apriori encontró {len(itemsets)} itemsets")

    return itemsets


def ejecutar_fpgrowth(df_matriz, min_support=MIN_SUPPORT):
    """
    Ejecuta FP-Growth y obtiene los itemsets frecuentes
    """

    itemsets = fpgrowth(
        df_matriz,
        min_support=min_support,
        use_colnames=True,
    )

    itemsets = itemsets.sort_values(
        by="support",
        ascending=False,
    ).reset_index(drop=True)

    print(f"FP-Growth encontró {len(itemsets)} itemsets")

    return itemsets


def generar_reglas(itemsets, min_confidence=MIN_CONFIDENCE):
    """
    Genera reglas de asociación y calcula sus métricas
    """

    # Comprueba que existan itemsets con al menos dos elementos
    if (
        itemsets.empty
        or not itemsets["itemsets"].map(len).ge(2).any()
    ):
        return pd.DataFrame()

    reglas = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    if reglas.empty:
        return reglas

    # Conserva las asociaciones con lift igual o superior al mínimo
    reglas = reglas[
        reglas["lift"] >= MIN_LIFT
    ].copy()

    # Ordena las reglas por lift, confianza y soporte descendente
    reglas = reglas.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False,
    ).reset_index(drop=True)

    print(f"Reglas relevantes encontradas: {len(reglas)}")

    return reglas


def preparar_exportacion_itemsets(itemsets):
    """
    Convierte los conjuntos de ítems a texto para el CSV
    """

    resultado = itemsets.copy()

    if not resultado.empty:
        resultado["itemsets"] = resultado["itemsets"].apply(
            lambda conjunto: ", ".join(sorted(conjunto))
        )

    return resultado


def preparar_exportacion_reglas(reglas):
    """
    Prepara los antecedentes y consecuentes para el CSV
    """

    resultado = reglas.copy()

    if not resultado.empty:
        resultado["antecedents"] = resultado[
            "antecedents"
        ].apply(
            lambda conjunto: ", ".join(sorted(conjunto))
        )

        resultado["consequents"] = resultado[
            "consequents"
        ].apply(
            lambda conjunto: ", ".join(sorted(conjunto))
        )

    return resultado


def guardar_resultados(
    itemsets_apriori,
    itemsets_fpgrowth,
    reglas_apriori,
    reglas_fpgrowth,
):
    """
    Guarda los itemsets y las reglas en archivos CSV
    """

    # Crea los directorios de resultados si todavía no existen
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.DATA_PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Guarda los itemsets de Apriori
    preparar_exportacion_itemsets(
        itemsets_apriori
    ).to_csv(
        ITEMSETS_APRIORI_FILE,
        index=False,
    )

    # Guarda los itemsets de FP-Growth
    preparar_exportacion_itemsets(
        itemsets_fpgrowth
    ).to_csv(
        ITEMSETS_FPGROWTH_FILE,
        index=False,
    )

    # Guarda las reglas de Apriori
    preparar_exportacion_reglas(
        reglas_apriori
    ).to_csv(
        RULES_APRIORI_FILE,
        index=False,
    )

    # Guarda las reglas de FP-Growth
    preparar_exportacion_reglas(
        reglas_fpgrowth
    ).to_csv(
        RULES_FPGROWTH_FILE,
        index=False,
    )

    print("\nResultados guardados correctamente")
    print(f"Itemsets: {RESULTS_DIR}")
    print(f"Reglas: {cfg.DATA_PROCESSED}")


def ejecutar_modelos(
    df_matriz,
    min_support=MIN_SUPPORT,
    min_confidence=MIN_CONFIDENCE,
):
    """
    Ejecuta ambos algoritmos y genera sus reglas
    """

    print("\nEjecutando Apriori...")

    itemsets_apriori = ejecutar_apriori(
        df_matriz,
        min_support,
    )

    reglas_apriori = generar_reglas(
        itemsets_apriori,
        min_confidence,
    )

    print("\nEjecutando FP-Growth...")

    itemsets_fpgrowth = ejecutar_fpgrowth(
        df_matriz,
        min_support,
    )

    reglas_fpgrowth = generar_reglas(
        itemsets_fpgrowth,
        min_confidence,
    )

    return (
        itemsets_apriori,
        itemsets_fpgrowth,
        reglas_apriori,
        reglas_fpgrowth,
    )


def main():
    """
    Ejecuta el proceso completo de reglas de asociación
    """

    print("Cargando dataset...")
    df = cargar_datos()

    print("\nConstruyendo transacciones...")
    transacciones = construir_transacciones(df)

    print("\nConstruyendo matriz de ítems...")
    df_matriz = construir_matriz(transacciones)

    print("\nEjecutando algoritmos...")

    resultados = ejecutar_modelos(
        df_matriz,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
    )

    guardar_resultados(*resultados)

    print("\nProceso de reglas de asociación finalizado")


if __name__ == "__main__":
    main()