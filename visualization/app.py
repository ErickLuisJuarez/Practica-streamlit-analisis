"""
Dashboard de consumo eléctrico doméstico en Streamlit
"""

import sys
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[1]
)  # La raíz del proyecto está un nivel arriba de visualization
if str(ROOT) not in sys.path:  # Si la raíz todavía no está en las rutas de Python
    sys.path.insert(
        0, str(ROOT)
    )  # La agrega así Python encuentra las carpetas config y src

import polars as pl
import streamlit as st

from config import rutas as cfg
from src.visualization import charts, data_loader, metrics

st.set_page_config(
    page_title="Consumo eléctrico doméstico", page_icon="⚡", layout="wide"
)  # Título de la pestaña del navegador y ancho completo

# Hallazgos donde van las conclusiones de cada gráfica, si una cadena está vacía no se muestra nada
HALLAZGOS = {
    "tendencia": "Se observa un comportamiento cíclico anual. Las variaciones extremas coinciden con los descensos de temperatura a final de año.",
    "mensual": "Durante el primer trimestre, la necesidad de calefacción eleva el consumo sustancialmente frente a los meses de verano, donde la demanda cae a su punto mínimo.",
    "patron_semanal": "Existe una similitud notable entre los días de la semana, aunque los fines de semana muestran un consumo matutino ligeramente más prolongado.",
    "distribucion": "La distribución está fuertemente sesgada a la derecha. El hogar pasa la mayor parte del tiempo con un consumo base muy bajo, interrumpido por ráfagas cortas de alta demanda.",
    "horaria": "El valle de consumo ocurre entre la 1:00 AM y las 6:00 AM. A partir de las 18:00 hrs inicia un ascenso pronunciado que culmina alrededor de las 21:00 hrs.",
    "relaciones": "La nube de puntos muestra una dependencia perfecta entre variables como Intensidad y Potencia Activa, sugiriendo que ambas miden el mismo fenómeno físico.",
    "correlacion": "Como se observa en el coeficiente cercano a 1.0, la variable 'Global_intensity' es redundante y puede ser descartada para evitar multicolinealidad en futuros modelos.",
    "pca": "El PCA confirma que la dimensionalidad del problema es mucho menor que las variables originales. Los primeros componentes logran resumir el estado operativo del hogar con alta fidelidad.",
}

# Títulos de cada gráfica escritos como conclusión, si una cadena está vacía se usa el título descriptivo
TITULOS = {
    "tendencia": "El consumo base se ha mantenido estable, con picos estacionales previsibles",
    "mensual": "Diciembre y Enero concentran la mayor demanda energética del año",
    "patron_semanal": "El consumo nocturno domina sin importar el día de la semana",
    "distribucion": "La mayor parte del tiempo el hogar opera en un estado de reposo (< 1 kW)",
    "horaria": "Pico agudo de actividad entre las 19:00 y las 22:00 horas",
    "relaciones": "Relación lineal estricta entre la potencia activa y la intensidad",
    "correlacion": "Global_intensity y Global_active_power son variables redundantes",
    "pca_varianza": "Un número reducido de componentes captura casi toda la variabilidad",
    "pca_loadings": "El primer componente refleja el consumo general; el segundo, perfiles específicos de uso",
    "pca_proyeccion": "Clara separación geométrica entre las horas valle y las horas pico",
}

# Las 4 frases con las que abre el dashboard, si las 4 están vacías no se muestra nada
SCQA = {
    "situacion": "Contamos con un registro detallado minuto a minuto del consumo eléctrico de un hogar durante casi cuatro años.",
    "complicacion": "El volumen de datos (>2 millones de registros) y la alta dimensionalidad dificultan identificar patrones de ahorro y variables redundantes.",
    "pregunta": "¿Cuáles son los verdaderos impulsores del consumo y cómo podemos reducir la dimensionalidad sin perder información clave?",
    "respuesta": "El análisis revela una estacionalidad marcada en invierno y horarios nocturnos, permitiendo optimizar el dataset descartando variables perfectamente correlacionadas mediante Pearson y PCA.",
}

# Periodo a destacar en las gráficas de tendencia, con fechas en formato "AAAA-MM-DD"
RESALTAR = {
    "inicio": "2008-11-01",
    "fin": "2009-02-28",
    "nota": "Temporada invernal de alta demanda",
}


def nota(key):
    """
    Muestra el hallazgo bajo la gráfica
    """
    if HALLAZGOS.get(key):  # Si la cadena no está vacía
        st.info(HALLAZGOS[key])  # La muestra en un recuadro azul


def titulo(key, por_defecto):
    """
    Devuelve el título-conclusión si se escribió en TITULOS, si no devuelve el título descriptivo
    """
    return (
        TITULOS.get(key) or por_defecto
    )  # Si la cadena está vacía usa el título por defecto


def periodo_resaltado():
    """
    Devuelve (inicio, fin, nota) si se escribieron las fechas en RESALTAR, si no devuelve None
    """
    if RESALTAR["inicio"] and RESALTAR["fin"]:  # Solo si están las dos fechas
        return RESALTAR["inicio"], RESALTAR["fin"], RESALTAR["nota"]
    return None


def mostrar_scqa():
    """
    Muestra las 4 frases SCQA en cuatro columnas, solo si se escribió al menos una
    """
    if not any(SCQA.values()):  # Si las 4 están vacías no se muestra nada
        return
    nombres = [
        ("situacion", "S · Situación"),
        ("complicacion", "C · Complicación"),
        ("pregunta", "Q · Pregunta"),
        ("respuesta", "A · Respuesta"),
    ]  # Clave y nombre de cada frase
    for columna, (clave, nombre) in zip(
        st.columns(4), nombres
    ):  # Una columna por frase
        columna.markdown(f"**{nombre}**")  # Nombre de la frase en negritas
        columna.write(SCQA[clave])  # El texto de la frase


# Unidad de cada variable
UNITS = {
    "Global_active_power": "kW",
    "Global_reactive_power": "kW",
    "Voltage": "V",
    "Global_intensity": "A",
    "Sub_metering_1": "Wh",
    "Sub_metering_2": "Wh",
    "Sub_metering_3": "Wh",
}
# Nombre legible de cada variable
LABELS = {
    "Global_active_power": "Potencia activa global",
    "Global_reactive_power": "Potencia reactiva global",
    "Voltage": "Voltaje",
    "Global_intensity": "Intensidad global",
    "Sub_metering_1": "Submedidor 1 (cocina)",
    "Sub_metering_2": "Submedidor 2 (lavandería)",
    "Sub_metering_3": "Submedidor 3 (calentador/A.C.)",
}


def label(col):
    """
    Devuelve el nombre legible de una variable con su unidad, por ejemplo Voltaje (V)
    """
    return (
        f"{LABELS.get(col, col)} ({UNITS[col]})" if col in UNITS else col
    )  # Si la variable no está en UNITS devuelve su nombre tal cual


if not data_loader.processed_files_exist():  # Si todavía no se corrió el pipeline
    st.error(
        "No encuentro los datos procesados. Ejecuta primero `python -m src.cleaning.run_pipeline` (ver README)."
    )  # Muestra un aviso de error
    st.stop()  # Detenemos la página porque no hay nada que mostrar

clean = (
    data_loader.load_clean()
)  # Dataset limpio en unidades originales, para KPIs y gráficas
prepared = (
    data_loader.load_prepared()
)  # Dataset normalizado con componentes, para la pestaña PCA

d_min, d_max = (
    clean[cfg.DATETIME_COL].min().date(),
    clean[cfg.DATETIME_COL].max().date(),
)  # Primera y última fecha, son los límites del selector de fechas

st.sidebar.header("Filtros")  # Título de la barra lateral
rango = st.sidebar.date_input(
    "Rango de fechas", value=(d_min, d_max), min_value=d_min, max_value=d_max
)  # Selector de rango de fechas
if not (
    isinstance(rango, tuple) and len(rango) == 2
):  # Mientras el usuario elige solo la fecha de inicio devuelve un valor suelto
    st.sidebar.info(
        "Selecciona fecha de inicio y de fin."
    )  # Le pide la fecha que falta
    st.stop()  # Esperamos a que elija las dos fechas
start, end = rango  # Fecha de inicio y fecha de fin
nivel = st.sidebar.selectbox(
    "Nivel de agregación temporal", list(metrics.AGGREGATIONS), index=1
)  # Elige Hora, Día, Semana o Mes (Día por defecto)
every = metrics.AGGREGATIONS[nivel]  # Convertimos la etiqueta al intervalo
variable = st.sidebar.selectbox(
    "Variable de análisis", cfg.NUMERIC_COLS, format_func=label
)  # Elegimos la variable a analizar

df = metrics.filter_range(
    clean, start, end
)  # Dataset limpio filtrado por las fechas elegidas
dfp = metrics.filter_range(
    prepared, start, end
)  # Dataset preparado filtrado por las fechas elegidas
if df.is_empty():  # Si el rango no tiene datos
    st.warning("No hay datos en el rango seleccionado.")  # Muestra una advertencia
    st.stop()  # Detenemos la página

st.title("⚡ Consumo eléctrico de un hogar")  # Título principal
st.caption(
    f"Individual Household Electric Power Consumption (UCI) · {start:%d/%m/%Y} – {end:%d/%m/%Y} · {len(df):,} minutos observados"
)  # Subtítulo con el rango y los minutos

mostrar_scqa()  # Las 4 frases de apertura, solo si se escribieron
resaltar = (
    periodo_resaltado()
)  # Periodo a resaltar en las gráficas de tendencia, o None si no hay

k = metrics.kpis(
    df
)  # Calcula los KPIs con TODOS los minutos del rango (no con muestra)
c1, c2, c3, c4, c5 = st.columns(5)  # Fila 1 de KPIs: 5 columnas
c1.metric(
    "Energía total", f"{k['energia_total_kwh']:,.0f} kWh"
)  # Energía total consumida
c2.metric("Potencia media", f"{k['potencia_media_kw']:.2f} kW")  # Potencia promedio
c3.metric(
    "Potencia máxima",
    f"{k['potencia_max_kw']:.2f} kW",
    help=f"Registrada el {k['momento_pico']:%d/%m/%Y %H:%M}",
)  # Potencia máxima y cuándo ocurrió
c4.metric("Voltaje medio", f"{k['voltaje_medio_v']:.1f} V")  # Voltaje promedio
c5.metric("Días con datos", f"{k['dias']:,}")  # Días distintos con datos

s1, s2, s3, s4 = st.columns(4)  # Fila 2 de KPIs
s1.metric(
    "Cocina (sub 1)", f"{k['energia_submedida_kwh']['Sub_metering_1']:,.0f} kWh"
)  # Energía del submedidor 1
s2.metric(
    "Lavandería (sub 2)", f"{k['energia_submedida_kwh']['Sub_metering_2']:,.0f} kWh"
)  # Energía del submedidor 2
s3.metric(
    "Calentador/A.C. (sub 3)",
    f"{k['energia_submedida_kwh']['Sub_metering_3']:,.0f} kWh",
)  # Energía del submedidor 3
s4.metric(
    "Energía no submedida",
    f"{k['energia_no_submedida_kwh']:,.0f} kWh",
    help="Energía total menos la suma de los tres submedidores.",
)  # Energía que no pasa por ningún submedidor

st.divider()  # Línea separadora

tab_t, tab_p, tab_r, tab_pca, tab_d = st.tabs(
    ["Tendencia", "Patrones", "Relaciones", "PCA", "Tabla detallada"]
)  # Creamos las 5 pestañas

with tab_t:  # Pestaña 1: cómo cambia el consumo en el tiempo
    agg = metrics.aggregate(
        df, every, [variable]
    ).to_pandas()  # Promediamos por el nivel elegido para que la gráfica no tenga 2 millones de puntos
    st.plotly_chart(
        charts.line_chart(
            agg,
            cfg.DATETIME_COL,
            [variable],
            titulo("tendencia", f"{LABELS[variable]} · promedio por {nivel.lower()}"),
            UNITS[variable],
            resaltar,
        ),
        width="stretch",
    )  # Gráfica de líneas de la variable elegida
    nota("tendencia")
    st.plotly_chart(
        charts.monthly_bars(
            metrics.monthly_energy(df).to_pandas(),
            titulo("mensual", "Energía consumida por mes (kWh)"),
            resaltar,
        ),
        width="stretch",
    )  # Barras de energía por mes
    st.caption(
        "Los meses con pocos días observados pueden verse artificialmente bajos; el detalle aparece al pasar el cursor."
    )  # Aviso sobre meses incompletos
    nota("mensual")

with tab_p:  # Pestaña 2: patrones por día de la semana y hora, y distribuciones
    st.plotly_chart(
        charts.heatmap_matrix(
            metrics.weekday_hour_matrix(df, variable),
            titulo(
                "patron_semanal",
                f"{LABELS[variable]} · promedio por día de la semana y hora",
            ),
            UNITS[variable],
        ),
        width="stretch",
    )  # Mapa de calor de día de la semana por hora
    nota("patron_semanal")
    left, right = st.columns(
        2
    )  # Dos columnas: histograma a la izquierda y boxplot a la derecha
    with left:  # Columna izquierda
        centers, counts = metrics.histogram(
            df[variable]
        )  # Calcula el histograma de la variable
        st.plotly_chart(
            charts.histogram_chart(
                centers,
                counts,
                titulo("distribucion", f"Distribución de {LABELS[variable].lower()}"),
                label(variable),
            ),
            width="stretch",
        )  # Dibujamos el histograma
        nota("distribucion")
    with right:  # Columna derecha
        st.plotly_chart(
            charts.box_by_hour(
                metrics.hourly_distribution(df, variable).to_pandas(),
                variable,
                titulo("horaria", f"{LABELS[variable]} por hora del día (muestra)"),
                label(variable),
            ),
            width="stretch",
        )  # Dibujamos el boxplot por hora
        nota("horaria")

with tab_r:  # Pestaña 3: relaciones entre variables
    a, b = st.columns(2)  # Dos columnas para los selectores
    x = a.selectbox(
        "Eje X", cfg.NUMERIC_COLS, index=0, format_func=label
    )  # Variable del eje x
    y = b.selectbox(
        "Eje Y", cfg.NUMERIC_COLS, index=3, format_func=label
    )  # Variable del eje y
    st.plotly_chart(
        charts.scatter_chart(
            metrics.sample(df).to_pandas(),
            x,
            y,
            titulo("relaciones", f"{x} vs {y} (muestra aleatoria de 8,000 minutos)"),
        ),
        width="stretch",
    )  # Dispersión con una muestra porque con 2 millones de puntos sería ilegible
    nota("relaciones")
    st.plotly_chart(
        charts.correlation_heatmap(
            metrics.correlation(df, cfg.NUMERIC_COLS),
            titulo("correlacion", "Correlación de Pearson en el rango seleccionado"),
        ),
        width="stretch",
    )  # Mapa de correlación con todos los datos del rango
    pear = data_loader.load_json(
        str(data_loader.CORR_REPORT)
    )  # Leemos las decisiones del Paso 4
    st.caption(
        f"Variables conservadas tras el Paso 4 (umbral |r| ≥ {pear['umbral']}): {', '.join(pear['conservadas'])}. Descartadas: {', '.join(pear['descartadas']) or 'ninguna'}."
    )  # Mostramos qué variables se conservaron y cuáles no
    nota("correlacion")

with tab_pca:  # Pestaña 4: resultados del PCA
    rep = data_loader.load_json(
        str(data_loader.PCA_REPORT)
    )  # Lee el reporte que guardó pca_model.py
    k_pc = rep["k_elegido"]  # Número de componentes retenidos
    m1, m2, m3 = st.columns(3)  # Tres columnas para los indicadores
    m1.metric("Componentes retenidos", k_pc)  # Cuántos componentes se conservaron
    m2.metric(
        "Varianza acumulada retenida", f"{rep['varianza_acumulada'][k_pc - 1]:.1%}"
    )  # Varianza que explican esos componentes
    m3.metric(
        "Sugerencia criterio de Kaiser", rep["k_kaiser"]
    )  # Cuántos sugiere el criterio de Kaiser
    st.plotly_chart(
        charts.scree_chart(
            rep["varianza_explicada"],
            k_pc,
            rep["objetivo_varianza"],
            titulo("pca_varianza", "Varianza explicada por componente (scree plot)"),
        ),
        width="stretch",
    )  # Scree plot
    st.plotly_chart(
        charts.loadings_heatmap(
            data_loader.load_loadings().iloc[:, :k_pc],
            titulo("pca_loadings", "Loadings de los componentes retenidos"),
        ),
        width="stretch",
    )  # Loadings solo de los componentes retenidos
    proj = (
        metrics.sample(dfp, 8_000)
        .with_columns(pl.col(cfg.DATETIME_COL).dt.hour().alias("hora"))
        .to_pandas()
    )  # Tomamos una muestra de puntos con la hora de cada uno
    st.plotly_chart(
        charts.pca_scatter(
            proj,
            "hora",
            titulo("pca_proyeccion", "Proyección en PC1 vs PC2 (color = hora del día)"),
        ),
        width="stretch",
    )  # Proyección en 2D coloreada por hora
    nota("pca")

with tab_d:  # Pestaña 5: tabla detallada
    cols = st.multiselect(
        "Variables a mostrar",
        cfg.NUMERIC_COLS,
        default=["Global_active_power", "Voltage"],
        format_func=label,
    )  # Elegimos las variables de la tabla
    if cols:  # Si eligió al menos una
        tabla = metrics.detail_table(
            df, every, cols
        )  # Media, mínimo y máximo por ventana de tiempo
        st.caption(
            f"{tabla.height:,} filas · agregación por {nivel.lower()}"
        )  # Cuántas filas tiene la tabla
        st.dataframe(
            tabla.to_pandas(), width="stretch", height=420
        )  # Mostramos la tabla
        st.download_button(
            "Descargar CSV",
            tabla.write_csv().encode("utf-8"),
            file_name=f"consumo_{nivel.lower()}.csv",
            mime="text/csv",
        )  # Botón para descargar la tabla en csv
    else:  # Si no eligió ninguna
        st.info("Elige al menos una variable.")  # Le pide que elija una
    with st.expander(
        "Dataset preparado (Paso 6): normalizado + componentes principales"
    ):  # Sección que se puede abrir y cerrar
        st.dataframe(
            dfp.head(500).to_pandas(), width="stretch"
        )  # Primeras 500 filas del dataset final
