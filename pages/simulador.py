"""Página: Simulador sísmico conectado al Pipeline KNN real."""

from __future__ import annotations

import streamlit as st

from pages._shared import add_simulation, load_everything, require_model, run_prediction
from src import charts, config, map_service, peru_reference as ref, ui
from streamlit_folium import st_folium

DEFAULT = {
    "departamento": "Áncash",
    "magnitud": 6.4,
    "profundidad_km": 35.0,
    "distancia_km": 60.0,
    "k": 5,
}


def _apply_scenario(scenario: dict) -> None:
    for key in ("departamento", "magnitud", "profundidad_km", "distancia_km", "k"):
        st.session_state[f"sim_{key}"] = scenario[key]


def _init_state() -> None:
    for key, value in DEFAULT.items():
        if f"sim_{key}" not in st.session_state or st.session_state[f"sim_{key}"] is None:
            st.session_state[f"sim_{key}"] = value
    st.session_state.setdefault("sim_run", False)
    st.session_state.setdefault("sim_err_ack", [])
    st.session_state.setdefault("sim_err_values", None)
    st.session_state.setdefault("sim_errors", [])


def render() -> None:
    ui.institutional_header()
    ui.page_title(
        "Simulador de escenario sísmico",
        "Introduce las características de un evento hipotético. La categoría, las probabilidades "
        "y los vecinos provienen del modelo KNN entrenado — no de reglas manuales.",
        eyebrow="ESTIMACIÓN EN TIEMPO REAL",
    )

    df, summary, artifact = load_everything()
    if not require_model(artifact):
        return
    meta = artifact["metadata"]

    _init_state()

    # ---------------- Escenarios de demostración ----------------
    ui.section_title("Escenarios de demostración", "Cargan parámetros de entrada; el modelo se ejecuta igual.", "fa-flask")
    cols = st.columns(len(ref.DEMO_SCENARIOS))
    for i, (col, scenario) in enumerate(zip(cols, ref.DEMO_SCENARIOS)):
        with col:
            st.markdown(
                f'<div class="scenario-card"><h4>{ui.esc(scenario["nombre"])}</h4>'
                f'<p>{ui.esc(scenario["descripcion"])}</p>'
                f'<div class="tags"><span class="tag">M {scenario["magnitud"]}</span>'
                f'<span class="tag">{int(scenario["profundidad_km"])} km</span>'
                f'<span class="tag">{int(scenario["distancia_km"])} km</span>'
                f'<span class="tag">{ui.esc(scenario["departamento"])}</span></div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Cargar escenario", key=f"demo_btn_{i}",
                         width='stretch'):
                _apply_scenario(scenario)
                st.session_state["sim_run"] = True
                st.rerun()
    st.caption(
        "Al cargar un escenario se completan los campos y se ejecuta el modelo."
    )

    st.divider()

    # ---------------- Formulario ----------------
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        ui.section_title("Parámetros del evento", None, "fa-sliders")

        departamento = st.selectbox(
            "Departamento o localidad",
            ref.DEPARTAMENTOS_LIST,
            key="sim_departamento",
            help="Se usa como contexto geográfico; no entra al modelo como variable categórica.",
        )
        localidades = ref.LOCALIDADES_EXTRA.get(
            departamento, [ref.DEPARTAMENTOS[departamento][2]]
        )
        localidad = st.selectbox("Localidad de referencia", localidades, key="sim_loc")

        magnitud = st.slider(
            "Magnitud del evento (Mw)", 3.0, 9.5,
            step=0.1, key="sim_magnitud",
        )
        profundidad = st.number_input(
            "Profundidad hipocentral (km)", min_value=0.0, max_value=700.0,
            step=1.0, key="sim_profundidad_km",
        )
        distancia = st.slider(
            "Distancia epicentral a la localidad (km)", 0.0, 600.0,
            step=1.0, key="sim_distancia_km",
        )
        k = st.slider(
            "Número de vecinos (K)", 1, 25, key="sim_k",
            help=f"El modelo persistido usa K = {meta.get('k_seleccionado')}. Puedes probar otros K: "
                 "se reutiliza el mismo StandardScaler ya ajustado.",
        )

        ejecutar = st.button("⚡ ESTIMAR INTENSIDAD", type="primary", width='stretch')
        if ejecutar:
            st.session_state["sim_run"] = True
            st.session_state["sim_err_ack"] = []
            st.session_state["sim_err_values"] = None
            # Ejecutar una nueva estimación aunque los parámetros no hayan cambiado.
            st.session_state["last_sim"] = None

    with right:
        ui.section_title("Contexto geográfico", None, "fa-location-dot")
        lat, lon, _ = ref.DEPARTAMENTOS[departamento]
        try:
            st_folium(map_service.location_map(lat, lon, localidad), height=300,
                      returned_objects=[], key="sim_map")
        except Exception as exc:  # noqa: BLE001
            ui.alert("warn", f"No se pudo mostrar el mapa de contexto: {exc}")
        st.markdown(
            f'<div class="glass">{ui.info_rows([("Departamento", ui.esc(departamento)), ("Localidad", ui.esc(localidad)), ("Rango de entrenamiento (M)", " {} – {}".format(*meta.get("dataset", {}).get("rango_magnitud", config.MAGNITUDE_RANGE))), ("Rango de entrenamiento (prof.)", " {} – {} km".format(*[round(v, 1) for v in meta.get("dataset", {}).get("rango_profundidad", config.DEPTH_RANGE)])), ("Rango de entrenamiento (dist.)", " {} – {} km".format(*[round(v, 1) for v in meta.get("dataset", {}).get("rango_distancia", config.DISTANCE_RANGE)]))])}</div>',
            unsafe_allow_html=True,
        )

    # ---------------- Resultado ----------------
    if not st.session_state.get("sim_run"):
        ui.alert("info", "Configura los parámetros y pulsa <b>ESTIMAR INTENSIDAD</b> para ejecutar el modelo KNN.")
        ui.disclaimer_footer()
        return

    # Validación previa mostrada en la interfaz (no se usa para clasificar).
    current_vals = (magnitud, profundidad, distancia, int(k))
    if st.session_state.get("sim_err_values") != current_vals:
        from src import prediction as _prediction
        st.session_state["sim_errors"] = _prediction.validate_inputs(
            magnitud, profundidad, distancia, int(k), meta
        )
        st.session_state["sim_err_values"] = current_vals
    validation_errors = st.session_state.get("sim_errors", [])
    if validation_errors:
        idx = st.session_state.get("sim_err_ack", [])
        remaining = [e for i, e in enumerate(validation_errors) if i not in idx]
        if remaining:
            ui.alert("error", "Valores fuera de rango (corrígelos para estimar):")
            for i, msg in enumerate(validation_errors):
                if i in idx:
                    continue
                c_msg, c_btn = st.columns([5, 1])
                c_msg.markdown(f"<span style='color:#fecaca;font-size:12.6px'>• {ui.esc(msg)}</span>",
                               unsafe_allow_html=True)
                if c_btn.button("Entendido", key=f"ack_{i}", width='stretch'):
                    st.session_state["sim_err_ack"] = idx + [i]
                    st.rerun()

    st.divider()
    result = run_prediction(
        artifact,
        magnitud=magnitud, profundidad_km=profundidad, distancia_km=distancia, k=int(k),
    )
    if result is None:
        ui.alert("warn", "Corrige los valores señalados para obtener una estimación.")
        ui.disclaimer_footer()
        return

    if "last_sim" not in st.session_state or st.session_state["last_sim"] != (
        departamento, magnitud, profundidad, distancia, k
    ):
        add_simulation(result, departamento)
        st.session_state["last_sim"] = (departamento, magnitud, profundidad, distancia, k)

    ui.result_hero(result["categoria"], result["prob_mayor"], result["k"],
                   f"{localidad}, {departamento}")
    ui.section_title("Resultado de la predicción", None, "fa-bullseye")

    c1, c2, c3 = st.columns([1.15, 1, 1], gap="medium")
    with c1:
        st.plotly_chart(charts.probability_chart(result["probabilidades"],
                                                 predicted=result["categoria"]),
                        width='stretch', config={"displayModeBar": False})
    with c2:
        st.plotly_chart(charts.vote_donut(result["votos"]), width='stretch',
                        config={"displayModeBar": False})
    with c3:
        votos_rows = [(f"{c}", f"{result['votos'].get(c, 0)} votos") for c in config.CLASS_ORDER]
        prob_rows = [(f"P({c})", f"{result['probabilidades'].get(c, 0) * 100:.1f}%")
                     for c in config.CLASS_ORDER]
        st.markdown(
            f'<div class="glass"><div class="section"><h3>Detalle</h3></div>'
            f'{ui.info_rows(votos_rows)}<div style="height:8px"></div>'
            f'{ui.info_rows(prob_rows)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="glass">{ui.info_rows([("K usado", str(result["k"])), ("K del modelo", str(result["k_modelo"])), ("Clase estimada", ui.category_badge(result["categoria"]))])}</div>',
            unsafe_allow_html=True,
        )

    # ---------------- Escenario evaluado ----------------
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        ui.section_title("Características del escenario", None, "fa-list-check")
        e = result["entradas"]
        rows = [
            ("Magnitud", f"{e['magnitud']:.1f} Mw"),
            ("Profundidad", f"{e['profundidad_km']:.1f} km"),
            ("Distancia epicentral", f"{e['distancia_km']:.1f} km"),
            ("Vecinos K", str(result["k"])),
            ("Categoría estimada", ui.category_badge(result["categoria"])),
            ("Confianza de votos", f"{result['prob_mayor'] * 100:.1f}%"),
        ]
        st.markdown(
            f'<div class="glass">{ui.info_rows(rows)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="formula" style="font-family:inherit;white-space:normal;font-size:11.6px">'
            f'La categoría <b>{ui.esc(result["categoria"])}</b> es el resultado de '
            f'<code>predict()</code>; los porcentajes son la proporción de votos de los '
            f'{result["k"]} vecinos más próximos (<code>predict_proba()</code>), '
            f'no probabilidades calibradas de ocurrencia.</div>',
            unsafe_allow_html=True,
        )
    with right:
        ui.section_title("Vecinos más cercanos",
                         "Distancias medidas en el espacio normalizado (media 0, σ 1).",
                         "fa-users")
        neighbors = result["vecinos"].copy()
        neighbors["fecha"] = neighbors["fecha"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            neighbors.rename(columns={
                "vecino": "#", "evento_id": "Evento", "fecha": "Fecha",
                "localidad": "Localidad", "departamento": "Departamento",
                "magnitud": "Magnitud", "profundidad_km": "Prof. (km)",
                "distancia_km": "Dist. (km)", "intensidad_observada": "MMI",
                "categoria_intensidad": "Categoría",
                "distancia_escalada": "Distancia normalizada",
            }),
            width='stretch', hide_index=True, height=min(60 + 35 * len(neighbors), 420),
        )

    # ---------------- Visualización de vecinos ----------------
    ui.section_title("Ubicación de la consulta y sus vecinos", None, "fa-braille")
    proj = artifact.get("projection", {})
    if proj:
        st.plotly_chart(
            charts.neighbor_projection(
                artifact["train_data"], proj["coords"],
                result["query_2d"], result["neighbor_2d"], result["categoria"],
            ),
            width='stretch', config={"displayModeBar": False},
        )
        st.caption(
            "La estrella blanca es el escenario evaluado; los rombos ámbar son los K vecinos "
            "que votaron. Proyección PCA de las 3 variables normalizadas."
        )

    ui.alert(
        "info",
        "Las estimaciones dependen de la calidad y representatividad del dataset. "
        "Este simulador <b>no predice terremotos</b>: estima efectos esperados para un escenario "
        "dado, de forma educativa.",
    )
    ui.disclaimer_footer()
