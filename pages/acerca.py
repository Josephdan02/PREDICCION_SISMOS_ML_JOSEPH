"""Página: Acerca del proyecto."""

from __future__ import annotations

import streamlit as st

from pages._shared import load_everything
from src import config, ui

TECH = [
    ("fa-brands fa-python", "Python 3.11", "Lenguaje base del sistema."),
    ("fa-gauge-high", "Streamlit", "Interfaz web interactiva."),
    ("fa-table", "Pandas / NumPy", "Procesamiento y cálculo numérico."),
    ("fa-brain", "Scikit-learn", "Pipeline KNN, validación y métricas."),
    ("fa-chart-line", "Plotly", "Visualizaciones interactivas."),
    ("fa-map", "Folium", "Mapa geográfico de eventos."),
    ("fa-file-pdf", "ReportLab", "Generación del informe PDF."),
    ("fa-floppy-disk", "Joblib", "Persistencia del modelo."),
]


def render() -> None:
    ui.institutional_header()
    ui.page_title(
        "Acerca de SISMO AI PERÚ",
        "Sistema inteligente de análisis y estimación sísmica con K-Nearest Neighbors, "
        "desarrollado como reto académico para SENATI.",
        eyebrow="SOBRE EL PROYECTO",
    )

    df, summary, artifact = load_everything()

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        ui.section_title("Descripción y objetivo académico", None, "fa-graduation-cap")
        st.markdown(
            """
<div class="glass">
<p style="font-size:12.9px;color:#c7d2ef;line-height:1.75;margin:0 0 10px">
<b>SISMO AI PERÚ</b> es una aplicación web educativa que estima la <b>categoría de intensidad
sísmica</b> esperada en una localidad para un evento hipotético, a partir de tres variables:
magnitud, profundidad hipocentral y distancia epicentral. Utiliza un clasificador
<b>K-Nearest Neighbors</b> entrenado, validado con división agrupada por evento y evaluado
sobre un conjunto de prueba independiente.
</p>
<p style="font-size:12.9px;color:#c7d2ef;line-height:1.75;margin:0">
El <b>objetivo académico</b> es demostrar el ciclo completo de un proyecto de Machine Learning:
obtención y documentación de datos, limpieza, ingeniería de variables, entrenamiento,
validación cruzada, evaluación estadística, persistencia del modelo e integración en una
interfaz de usuario funcional.
</p>
</div>
            """,
            unsafe_allow_html=True,
        )

        ui.section_title("Magnitud vs. intensidad", None, "fa-scale-balanced")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
<div class="glass">
<h4 style="color:#22d3ee!important;font-size:13px;margin:0 0 6px">MAGNITUD (M)</h4>
<p style="font-size:11.9px;color:#97a3c2;line-height:1.65;margin:0">
Propiedad <b>del evento</b>. Mide la energía liberada en el foco sísmico. Es única por
sismo y no depende de dónde se observe. Escalas habituales: Mw, mb, ML.
</p>
</div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
<div class="glass">
<h4 style="color:#a78bfa!important;font-size:13px;margin:0 0 6px">INTENSIDAD (MMI)</h4>
<p style="font-size:11.9px;color:#97a3c2;line-height:1.65;margin:0">
Propiedad <b>del lugar</b>. Describe los efectos observados y sentidos en un sitio.
Un mismo sismo puede sentirse con intensidad baja en un lugar y fuerte en otro.
</p>
</div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="formula" style="white-space:normal;font-family:inherit">'
            'Baja: I–IV · Moderada: V–VI · Fuerte: VII–XII</div>',
            unsafe_allow_html=True,
        )

        ui.section_title("Funcionamiento general de KNN", None, "fa-diagram-project")
        st.markdown(
            """
1. El escenario se normaliza con el `StandardScaler` ajustado en entrenamiento.
2. Se calcula la distancia euclidiana a todos los registros del conjunto de entrenamiento.
3. Se seleccionan los **K** vecinos más próximos.
4. Cada vecino vota por su categoría de intensidad; gana la mayoría.
5. La proporción de votos se informa como distribución de apoyo entre vecinos.
            """
        )

    with right:
        ui.section_title("Ficha del sistema", None, "fa-circle-info")
        if artifact:
            meta = artifact["metadata"]
            rows = [
                ("Institución", config.INSTITUTION),
                ("Proyecto", config.PROJECT_NAME),
                ("Modelo", f"KNN · K = {meta.get('k_seleccionado')}"),
                ("Registros", f"{summary.get('total_registros', 0)}"),
                ("Eventos", f"{summary.get('eventos', 0)}"),
                ("Accuracy (test)", f"{meta.get('metricas_test', {}).get('accuracy', 0)*100:.1f}%"),
                ("Versión", "1.0.0"),
            ]
        else:
            rows = [("Institución", config.INSTITUTION), ("Estado", "Modelo no entrenado")]
        st.markdown(f'<div class="glass">{ui.info_rows(rows)}</div>', unsafe_allow_html=True)

        ui.section_title("Fuentes de referencia", None, "fa-book")
        st.markdown(
            """
- [IGP — Instituto Geofísico del Perú](https://www.gob.pe/igp)
- [CENSIS — Centro Sismológico Nacional](https://ultimosismo.igp.gob.pe/)
- [USGS — Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/)
- [USGS ShakeMap](https://earthquake.usgs.gov/data/shakemap/)
            """
        )
        ui.alert(
            "info",
            "Los catálogos oficiales documentan magnitud, profundidad y coordenadas; la intensidad "
            "Mercalli observada por localidad no está disponible de forma completa. Por ello, el "
            "dataset del sistema es simulado con fines académicos.",
        )

    ui.section_title("Tecnologías utilizadas", None, "fa-microchip")
    tech_html = "".join(
        f'<div class="stat-card" style="--accent:#a78bfa">'
        f'<div class="stat-top"><span class="stat-label">{ui.esc(name)}</span>'
        f'<span class="stat-icon"><i class="{icon}"></i></span></div>'
        f'<div class="stat-hint" style="margin-top:10px">{ui.esc(desc)}</div></div>'
        for icon, name, desc in TECH
    )
    ui.stat_grid([tech_html])

    ui.section_title("Alcance y advertencia", None, "fa-triangle-exclamation")
    st.markdown(
        """
<div class="glass">
<p style="font-size:12.4px;color:#fde68a;line-height:1.8;margin:0">
<b>Este sistema NO predice terremotos.</b> No estima la fecha de un sismo, su ubicación futura,
su ocurrencia ni su magnitud. Estima, para un escenario de entrada definido por el usuario, la
categoría de intensidad que el modelo considera más probable según los datos con los que fue
entrenado. No es un sistema oficial de alerta temprana ni reemplaza las evaluaciones del IGP,
INDECI u otras entidades competentes.
</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    ui.disclaimer_footer()
