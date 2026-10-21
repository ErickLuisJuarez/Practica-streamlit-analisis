"""
Gráficas del dashboard con Plotly
"""
import numpy as np
import plotly.graph_objects as go

#Paleta de colores: gris para el contexto y colores de acento para lo importante
NAVY = "#0F2A3D"
TEAL = "#1B7490"
ORANGE = "#E4572E"
GREY = "#B8C4CC"
SERIES_COLORS = [TEAL, ORANGE, NAVY, "#F2B134", "#7B5195", "#2BA58C", GREY]  #Colores para cuando hay varias líneas


def _style(fig, title, height=380, ytitle=None, xtitle=None):
    """
    Aplica el estilo común a todas las gráficas
    """
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color=NAVY)),  #Título alineado a la izquierda
        height=height,  #Altura de la gráfica
        margin=dict(l=10, r=10, t=50, b=10),  #Márgenes chicos
        plot_bgcolor="white",  #Fondo del área de la gráfica
        paper_bgcolor="white",  #Fondo de toda la figura
        font=dict(color=NAVY),  #Color del texto
        legend=dict(orientation="h", y=-0.2),  #Leyenda horizontal debajo de la gráfica
        hovermode="x unified",  #Al pasar el cursor muestra todos los valores de ese x
    )
    fig.update_xaxes(showgrid=False, title=xtitle, linecolor=GREY)  #Eje x sin cuadrícula
    fig.update_yaxes(gridcolor="#E9EEF1", title=ytitle, zeroline=False)  #Eje y con cuadrícula muy tenue
    return fig


def line_chart(df, x, ys, title, ytitle=None):
    """
    Gráfica de líneas de una o más variables a lo largo del tiempo
    """
    fig = go.Figure()  #Figura vacía
    for i, y in enumerate(ys):  #Una línea por variable
        fig.add_scatter(x=df[x], y=df[y], name=y, mode="lines",
                        line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=1.6))  #Dibujamos la línea con un color de la paleta
    return _style(fig, title, ytitle=ytitle)  #Aplicamos el estilo común


def monthly_bars(df, title):
    """
    Barras de energía por mes, al pasar el cursor muestra los días observados
    """
    fig = go.Figure(go.Bar(
        x=df["datetime"], y=df["kwh"], marker_color=TEAL, customdata=df["dias_observados"].round(1),  #Barras de kWh por mes y los días observados para el tooltip
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} kWh<br>%{customdata} días observados<extra></extra>"))  #Texto que aparece al pasar el cursor
    fig = _style(fig, title, ytitle="kWh")  #Aplicamos el estilo común
    fig.update_yaxes(rangemode="tozero")  #Las barras siempre parten de cero
    return fig


def heatmap_matrix(m, title, colorbar):
    """
    Mapa de calor de día de la semana por hora
    """
    fig = go.Figure(go.Heatmap(
        z=m.values, x=m.columns, y=m.index, colorscale=[[0, "#F4F7F9"], [1, NAVY]],  #Valores, ejes y escala de casi blanco a azul marino
        colorbar=dict(title=colorbar), hovertemplate="%{y} · %{x} h<br>%{z:.2f}<extra></extra>"))  #Barra de colores y texto del cursor
    fig = _style(fig, title)  #Aplicamos el estilo común
    fig.update_yaxes(autorange="reversed", gridcolor="white")  #Pone el lunes arriba
    fig.update_xaxes(dtick=2, title="Hora del día")  #Marca una etiqueta cada 2 horas
    return fig


def histogram_chart(centers, counts, title, xtitle):
    """
    Histograma dibujado como barras pegadas
    """
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=TEAL, marker_line_width=0))  #Una barra por intervalo
    fig = _style(fig, title, xtitle=xtitle, ytitle="Registros")  #Aplicamos el estilo común
    fig.update_layout(bargap=0.02, hovermode="closest")  #Barras casi pegadas porque la variable es continua
    return fig


def box_by_hour(df, col, title, ytitle):
    """
    Boxplot de una variable para cada hora del día
    """
    fig = go.Figure()  #Figura vacía
    for hora, grupo in df.groupby("hora"):  #Una caja por cada hora
        fig.add_box(y=grupo[col], name=str(hora), marker_color=TEAL, line=dict(color=TEAL, width=1),
                    marker_size=2, showlegend=False)  #Caja con mediana, cuartiles y valores atípicos
    fig = _style(fig, title, ytitle=ytitle, xtitle="Hora del día")  #Aplicamos el estilo común
    fig.update_layout(hovermode="closest")  #El cursor muestra solo el punto más cercano
    return fig


def scatter_chart(df, x, y, title):
    """
    Dispersión entre dos variables
    """
    fig = go.Figure(go.Scattergl(x=df[x], y=df[y], mode="markers", marker=dict(color=TEAL, size=4, opacity=0.35)))  #Puntos semitransparentes, donde se encimen se ve más oscuro
    fig = _style(fig, title, xtitle=x, ytitle=y)  #Aplica el estilo común
    fig.update_layout(hovermode="closest")  #El cursor muestra solo el punto más cercano
    return fig


def correlation_heatmap(corr, title):
    """
    Mapa de calor de la matriz de correlación con el valor de r en cada celda
    """
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu_r",  #Escala fija de -1 a 1
        text=np.round(corr.values, 2), texttemplate="%{text}", colorbar=dict(title="r")))  #Escribe r redondeado en cada celda
    fig = _style(fig, title, height=460)  #Aplica el estilo común
    fig.update_yaxes(autorange="reversed", gridcolor="white")  #La primera variable queda arriba
    fig.update_layout(hovermode="closest")  #El cursor muestra solo la celda más cercana
    return fig


def scree_chart(ratio, k, target, title):
    """
    Scree plot con varianza por componente, acumulada, objetivo y k elegido
    """
    x = [f"PC{i + 1}" for i in range(len(ratio))]  #Nombres de los componentes
    fig = go.Figure()  #Figura vacía
    fig.add_bar(x=x, y=ratio, name="Varianza explicada", marker_color=GREY)  #Barras: varianza de cada componente
    fig.add_scatter(x=x, y=np.cumsum(ratio), name="Acumulada", mode="lines+markers", line=dict(color=ORANGE))  #Línea: varianza acumulada
    fig.add_hline(y=target, line_dash="dash", line_color="gray")  #Línea horizontal en el objetivo de varianza
    fig.add_vline(x=k - 0.5, line_dash="dot", line_color=NAVY)  #Línea vertical donde se corta en k componentes
    fig = _style(fig, title, ytitle="Proporción de varianza")  #Aplicamos el estilo común
    fig.update_yaxes(range=[0, 1.02])  #El eje y va de 0 a 1
    return fig


def loadings_heatmap(loadings, title):
    """
    Mapa de calor de los loadings, qué tanto pesa cada variable en cada componente
    """
    fig = go.Figure(go.Heatmap(
        z=loadings.values, x=loadings.columns, y=loadings.index, zmin=-1, zmax=1, colorscale="RdBu_r",  #Escala fija de -1 a 1
        text=np.round(loadings.values, 2), texttemplate="%{text}", colorbar=dict(title="loading")))  #Escribe el loading en cada celda
    fig = _style(fig, title, height=400)  #Aplicamos el estilo común
    fig.update_yaxes(autorange="reversed", gridcolor="white")  #La primera variable queda arriba
    fig.update_layout(hovermode="closest")  #El cursor muestra solo la celda más cercana
    return fig


def pca_scatter(df, color, title):
    """
    Proyección de los datos en PC1 contra PC2, coloreada por la variable de hora
    """
    fig = go.Figure(go.Scattergl(
        x=df["PC1"], y=df["PC2"], mode="markers",  #PC1 en el eje x y PC2 en el eje y
        marker=dict(color=df[color], colorscale="Viridis", size=4, opacity=0.5, colorbar=dict(title="Hora"))))  #El color de cada punto depende de la hora
    fig = _style(fig, title, xtitle="PC1", ytitle="PC2", height=460)  #Aplicamos el estilo común
    fig.update_layout(hovermode="closest")  #El cursor muestra solo el punto más cercano
    return fig