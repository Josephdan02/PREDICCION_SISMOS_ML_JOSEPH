"""Página: Mapa interactivo de eventos sísmicos."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from pages._shared import load_everything
from src import config, map_service, ui


def render() -> None:
    ui.institutional_header()
    ui.page_title(
        "Mapa de eventos sísmicos",
        "Epicentros del catálogo con filtros por departamento, magnitud, profundidad e intensidad. "
        "Los registros sin coordenadas válidas se excluyen y se contabilizan.",
        eyebrow="GEOLOCALIZACIÓN",
    )

    df, summary, _ = load_everything()
    if df is None:
        return

    ui.section_title("Filtros del mapa", None, "fa-filter")
    with st.container():
        f1, f2, f3, f4 = st.columns([1.2, 1, 1, 1])
        with f1:
            dept = st.selectbox(
                "Departamento",
                ["Todos"] + sorted(df["departamento"].dropna().unique().tolist()),
                key="map_dept",
            )
        with f2:
            mag_min, mag_max = float(df["magnitud"].min()), float(df["magnitud"].max())
            mag_range = st.slider(
                "Rango de magnitud", round(mag_min, 1), round(mag_max, 1),
                (round(mag_min, 1), round(mag_max, 1)), step=0.1, key="map_mag",
            )
        with f3:
            p_min, p_max = float(df["profundidad_km"].min()), float(df["profundidad_km"].max())
            prof_range = st.slider(
                "Rango de profundidad (km)", int(p_min), int(round(p_max)),
                (int(p_min), int(round(p_max))), key="map_prof",
            )
        with f4:
            categoria = st.selectbox("Categoría de intensidad",
                                     ["Todas"] + config.CLASS_ORDER, key="map_cat")

        o1, o2, o3 = st.columns([1, 1, 1.2])
        with o1:
            heatmap = st.toggle("Mostrar mapa de calor", value=False)
        with o2:
            cluster = st.toggle("Agrupar marcadores", value=True)
        with o3:
            show_huanuco = st.toggle("Marcar Huánuco (referencia)", value=True)

    filtered = map_service.filter_events(df, dept, mag_range, prof_range, categoria)
    geo, excluded = map_service.valid_coords(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eventos filtrados", f"{len(filtered)}")
    c2.metric("Con coordenadas", f"{len(geo)}")
    c3.metric("Sin coordenadas", f"{excluded}")
    c4.metric("Departamentos", f"{filtered['departamento'].nunique()}")

    if filtered.empty:
        ui.alert("warn", "No hay eventos que coincidan con los filtros seleccionados.")
        ui.disclaimer_footer()
        return
    if excluded:
        ui.alert(
            "info",
            f"{excluded} registro(s) no tienen coordenadas válidas y se excluyen del mapa "
            "(se mantienen en el dataset y en el entrenamiento).",
        )

    try:
        fmap, shown, _ = map_service.build_events_map(
            geo, use_cluster=cluster, heatmap=heatmap, show_huanuco=show_huanuco
        )
        st_folium(fmap, height=540, width=None, returned_objects=[])
        st.caption(f"Mostrando {shown} epicentros. Haz clic en un marcador para ver su ficha.")
    except Exception as exc:  # noqa: BLE001
        ui.alert("error", f"No se pudo generar el mapa: {exc}")
        return

    # ---------------- Consulta de evento ----------------
    ui.section_title("Consultar un evento", "Selecciona un registro del filtro actual.",
                     "fa-magnifying-glass-location")
    options = (
        filtered["evento_id"].astype(str) + " · " + filtered["localidad"].astype(str)
        + " (" + filtered["departamento"].astype(str) + ")"
    ).tolist()
    if not options:
        return
    picked = st.selectbox("Evento", options, key="map_event_pick")
    evento_id = picked.split(" · ")[0]
    row = filtered[filtered["evento_id"].astype(str) == evento_id].iloc[0]

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        fecha = pd.to_datetime(row["fecha"])
        st.markdown(
            f'<div class="glass">'
            f'<div class="section"><h3>{ui.esc(row["localidad"])} — {ui.esc(row["departamento"])}</h3>'
            f'<p class="section-sub">Evento {ui.esc(row["evento_id"])}</p></div>'
            + ui.info_rows([
                ("Fecha", fecha.strftime("%Y-%m-%d") if pd.notna(fecha) else "N/D"),
                ("Magnitud", f"{row['magnitud']:.1f} {ui.esc(row['escala_magnitud'] or '')}"),
                ("Profundidad", f"{row['profundidad_km']:.1f} km"),
                ("Distancia epicentral", f"{row['distancia_km']:.1f} km"),
                ("Intensidad observada", f"MMI {int(row['intensidad_observada'])}"),
                ("Categoría", ui.category_badge(row["categoria_intensidad"])),
                ("Tipo de registro", ui.badge(row["tipo_registro"],
                                              "warn" if row["tipo_registro"] == "SIMULADO" else "ok")),
                ("Fuente", ui.esc(row["fuente"])),
                ("Coordenadas", f"{row['latitud']:.4f}, {row['longitud']:.4f}"),
            ]) + "</div>",
            unsafe_allow_html=True,
        )
    with right:
        same_event = df[df["evento_id"] == row["evento_id"]]
        st.markdown(
            f'<div class="glass"><div class="section"><h3>Localidades del mismo evento</h3>'
            f'<p class="section-sub">Agrupadas por evento_id: nunca se separan entre '
            f'entrenamiento y prueba.</p></div>'
            f'<span class="badge badge-info">{len(same_event)} registro(s)</span></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            same_event[["localidad", "departamento", "magnitud", "distancia_km",
                        "intensidad_observada", "categoria_intensidad"]],
            width='stretch', hide_index=True, height=240,
        )

    ui.alert(
        "info",
        "Los marcadores son <b>epicentros</b>: un mismo evento puede afectar varias localidades "
        "con distinta intensidad. El color y tamaño representan la <b>magnitud</b>, no la intensidad.",
    )
    ui.disclaimer_footer()
