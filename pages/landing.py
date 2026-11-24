"""Página: Dashboard principal con indicadores reales del sistema."""

from __future__ import annotations

import streamlit as st

from pages._shared import load_everything, require_model
from src import charts, config, map_service, ui
from streamlit_folium import st_folium


def render() -> None:
    ui.institutional_header()
    ui.page_title(
        "Panel general del sistema",
        "Estado del catálogo sísmico, distribución de los datos y desempeño del modelo KNN "
        "cargado actualmente.",
        eyebrow="DASHBOARD",
    )

    df, summary, artifact = load_everything()
    if df is None:
        return

    meta = artifact["metadata"] if artifact else {}
    metrics = meta.get("metricas_test", {})

    # ---------------- Tarjetas estadísticas (datos reales) ----------------
    acc_txt = f"{metrics.get('accuracy', 0) * 100:.1f}" if metrics else "—"
    cards = [
        ui.stat_card("Total registros", f"{summary['total_registros']:,}", "",
                     "fa-database", "#4f8cff",
                     hint=f"{summary['eventos']} eventos únicos"),
        ui.stat_card("Magnitud promedio", f"{summary['magnitud_promedio']}", "Mw",
                     "fa-wave-square", "#22d3ee",
                     hint="Escala de momento"),
        ui.stat_card("Profundidad prom.", f"{summary['profundidad_promedio']}", "km",
                     "fa-arrow-down-long", "#a78bfa",
                     hint="Hipocentro"),
        ui.stat_card("Dptos. cubiertos", f"{summary['departamentos']}", "",
                     "fa-map-location-dot", "#f472b6",
                     hint=f"{summary['localidades']} localidades"),
        ui.stat_card("Exactitud modelo", acc_txt, "%",
                     "fa-bullseye", "#34d399",
                     hint="Conjunto de prueba reservado" if metrics else "Modelo sin entrenar"),
        ui.stat_card("Eventos del modelo",
                     f"{meta.get('dataset', {}).get('eventos', 0) or summary['eventos']}", "",
                     "fa-layer-group", "#fbbf24",
                     hint=f"K = {meta.get('k_seleccionado', '—')}"),
    ]
    ui.stat_grid(cards)

    # ---------------- Gráficos principales ----------------
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.plotly_chart(
            charts.histogram(df, "magnitud", "Distribución de magnitudes", nbins=26,
                             color="#4f8cff"),
            width='stretch', config={"displayModeBar": False},
        )
    with col2:
        st.plotly_chart(charts.class_bar(df), width='stretch',
                        config={"displayModeBar": False})

    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.plotly_chart(charts.department_bar(df, top_n=12), width='stretch',
                        config={"displayModeBar": False})
    with col4:
        st.plotly_chart(
            charts.histogram(df, "profundidad_km", "Distribución de profundidades (km)",
                             nbins=28, color="#a78bfa"),
            width='stretch', config={"displayModeBar": False},
        )

    # ---------------- Mapa resumen + modelo ----------------
    left, right = st.columns([2.1, 1], gap="medium")
    with left:
        ui.section_title("Mapa resumen de eventos", "Epicentros con coordenadas válidas.",
                         "fa-earth-americas")
        try:
            sample = df.sample(min(len(df), 400), random_state=config.RANDOM_STATE) \
                if len(df) > 400 else df
            fmap, shown, excluded = map_service.build_events_map(sample, use_cluster=True)
            st_folium(fmap, height=380, width=None, returned_objects=[])
            st.caption(
                f"Mostrando {shown} de {len(df)} registros · "
                f"{excluded} sin coordenadas válidas excluidos del mapa."
            )
        except Exception as exc:  # noqa: BLE001
            ui.alert("error", f"No se pudo generar el mapa: {exc}")

    with right:
        ui.section_title("Modelo cargado", None, "fa-microchip")
        if artifact:
            ui.model_card(meta)
            ui.alert("ok", "El simulador usa este pipeline entrenado.")
        else:
            require_model(artifact)

        ui.section_title("Actualización del dataset", None, "fa-calendar-check")
        last = summary.get("ultima_actualizacion") or "N/D"
        tipos = summary.get("tipos_registro", {})
        tipo_txt = " · ".join(f"{k}: {v}" for k, v in tipos.items()) or "N/D"
        st.markdown(
            f'<div class="glass">{ui.info_rows([("Última fecha registrada", str(last)), ("Origen de los registros", tipo_txt), ("Con coordenadas", str(summary["con_coordenadas"])), ("Sin coordenadas", str(summary["sin_coordenadas"]))])}</div>',
            unsafe_allow_html=True,
        )
        if any(t != "SIMULADO" for t in tipos):
            ui.alert("info", "El dataset combina registros simulados y de otras fuentes; "
                             "revisa la columna <b>tipo_registro</b>.")
        else:
            ui.alert("warn", "Este dataset es <b>100% simulado</b> con fines académicos.")

    ui.section_title("Relación entre variables predictoras", None, "fa-diagram-project")
    st.plotly_chart(charts.scatter_features(df), width='stretch',
                    config={"displayModeBar": False})

    ui.disclaimer_footer()
