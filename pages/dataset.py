"""Página: Exploración y administración del dataset."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pages._shared import load_everything, to_csv_bytes
from src import charts, config, data_loader, preprocessing as pp, ui


def render() -> None:
    ui.institutional_header()
    ui.page_title(
        "Exploración del conjunto de datos",
        "Calidad de los datos, tipos, valores faltantes, distribución de clases y descarga del CSV.",
        eyebrow="DATASET",
    )

    df, summary, _ = load_everything()
    if df is None:
        return

    # ---------------- Procedencia ----------------
    tipos = df["tipo_registro"].value_counts()
    simulados = int(tipos.get("SIMULADO", 0))
    otros = int(len(df) - simulados)
    if otros == 0:
        ui.alert(
            "warn",
            "<b>Dataset 100 % simulado.</b> Los registros fueron generados con un modelo "
            "probabilístico académico; no son observaciones reales del IGP ni del USGS.",
        )
    else:
        ui.alert(
            "info",
            f"Dataset mixto: <b>{simulados}</b> registros SIMULADOS y <b>{otros}</b> de otras "
            "fuentes. Cada fila declara su origen en <b>tipo_registro</b> y <b>fuente</b>.",
        )

    # ---------------- Métricas ----------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", f"{df.shape[1]}")
    c3.metric("Valores faltantes", f"{int(df.isna().sum().sum())}")
    c4.metric("Duplicados exactos", f"{int(df.duplicated().sum())}")
    c5.metric("Eventos únicos", f"{summary['eventos']}")

    # ---------------- Distribución de clases / descriptivas ----------------
    tab_overview, tab_types, tab_stats, tab_source, tab_explore = st.tabs(
        ["Distribución y clases", "Tipos y faltantes", "Estadísticas descriptivas",
         "Procedencia", "Explorar y descargar"]
    )

    with tab_overview:
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.plotly_chart(charts.class_bar(df, title="Registros por categoría"),
                            width='stretch', config={"displayModeBar": False})
        with right:
            dist = pp.class_distribution(df)
            dist["rango MMI"] = dist["categoria_intensidad"].map(config.INTENSITY_RANGE)
            st.dataframe(
                dist.rename(columns={"categoria_intensidad": "Categoría",
                                     "registros": "Registros", "porcentaje": "%"}),
                width='stretch', hide_index=True,
            )
            st.caption(
                "Agrupación: I–IV = Baja · V–VI = Moderada · VII–XII = Fuerte. "
                "Un desbalance moderado es habitual: las intensidades bajas son más frecuentes."
            )
        st.plotly_chart(
            charts.histogram(df, "intensidad_observada",
                             "Distribución de la intensidad observada (escala MMI)",
                             nbins=12, color="#22d3ee"),
            width='stretch', config={"displayModeBar": False},
        )

    with tab_types:
        st.dataframe(data_loader.column_report(df), width='stretch',
                     hide_index=True)
        st.caption(
            "El dataset limpio no contiene valores faltantes en columnas críticas. "
            "Las columnas opcionales (por ejemplo escala_magnitud) pueden presentarlos."
        )

    with tab_stats:
        numeric = df.select_dtypes(include="number")
        st.dataframe(
            numeric.describe().T.round(2).rename(columns={
                "count": "n", "mean": "media", "std": "desv.est", "min": "mín.",
                "25%": "Q1", "50%": "mediana", "75%": "Q3", "max": "máx.",
            }),
            width='stretch',
        )
        st.plotly_chart(charts.box_by_class(df, "magnitud", "Magnitud por categoría"),
                        width='stretch', config={"displayModeBar": False})
        st.plotly_chart(charts.box_by_class(df, "distancia_km", "Distancia por categoría"),
                        width='stretch', config={"displayModeBar": False})

    with tab_source:
        st.markdown(
            f'<div class="glass">{ui.info_rows([("Fuentes", " · ".join(map(str, df["fuente"].dropna().unique()[:4]))), ("Tipos de registro", " · ".join(f"{k}: {v}" for k, v in tipos.items())), ("Escalas de magnitud", ", ".join(map(str, df["escala_magnitud"].dropna().unique()))), ("Última fecha", str(summary.get("ultima_actualizacion", "N/D")))])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
**Procedencia y metodología**

- **Catálogos reales (USGS, IGP):** documentan magnitud, profundidad y coordenadas con alta
  cobertura. Sin embargo, un catálogo de magnitudes **no** contiene intensidad Mercalli
  observada por localidad. El atributo `mmi` del catálogo USGS es una estimación
  (ShakeMap/estaciones) y solo existe para parte de los eventos modernos.
- **Dataset académico simulado:** al no ser posible obtener ≥ 500 registros con intensidad
  observada verificable, se generó un dataset evento-localidad con un modelo probabilístico
  de atenuación: magnitudes Gutenberg–Richter truncadas, profundidades en bandas
  (superficial/intermedia/profunda), distancias con cola larga y ruido gaussiano sobre la
  intensidad. Esto evita etiquetas producidas por una regla determinista.
- **Columnas:** `evento_id`, `fecha`, `latitud`, `longitud`, `localidad`, `departamento`,
  `magnitud`, `escala_magnitud`, `profundidad_km`, `distancia_km`, `intensidad_observada`,
  `categoria_intensidad`, `tipo_registro`, `fuente`.
            """
        )

    with tab_explore:
        f1, f2, f3 = st.columns(3)
        with f1:
            dept_opts = ["Todos"] + sorted(df["departamento"].dropna().unique().tolist())
            dept = st.selectbox("Departamento", dept_opts, key="ds_dept")
        with f2:
            mag_r = st.slider(
                "Magnitud mínima / máxima",
                float(round(df["magnitud"].min(), 1)), float(round(df["magnitud"].max(), 1)),
                (float(round(df["magnitud"].min(), 1)), float(round(df["magnitud"].max(), 1))),
                step=0.1, key="ds_mag",
            )
        with f3:
            cat = st.selectbox("Intensidad", ["Todas"] + config.CLASS_ORDER, key="ds_cat")

        view = df.copy()
        if dept != "Todos":
            view = view[view["departamento"] == dept]
        view = view[view["magnitud"].between(*mag_r)]
        if cat != "Todas":
            view = view[view["categoria_intensidad"] == cat]

        st.caption(f"{len(view)} registro(s) coinciden con el filtro.")
        display = view.copy()
        display["fecha"] = pd.to_datetime(display["fecha"]).dt.strftime("%Y-%m-%d")
        st.dataframe(display, width='stretch', hide_index=True, height=380)

        d1, d2 = st.columns([1, 3])
        with d1:
            st.download_button(
                "⬇️ Descargar CSV filtrado",
                data=to_csv_bytes(view),
                file_name="sismos_peru_filtrado.csv",
                mime="text/csv",
                width='stretch',
            )
        with d2:
            st.download_button(
                "⬇️ Descargar dataset completo",
                data=to_csv_bytes(df),
                file_name="sismos_peru.csv",
                mime="text/csv",
                width='stretch',
            )

    ui.disclaimer_footer()
