"""Página: Análisis del algoritmo KNN (teoría + entrenamiento real)."""

from __future__ import annotations

import streamlit as st

from pages._shared import clear_caches, load_everything, require_model
from src import charts, config, evaluation as ev, training_service, ui


def render() -> None:
    ui.institutional_header()
    ui.page_title(
        "Cómo decide el modelo",
        "El algoritmo K-Nearest Neighbors clasifica un escenario comparándolo con los "
        "registros históricos más parecidos del conjunto de entrenamiento.",
        eyebrow="ALGORITMO KNN",
    )

    df, summary, artifact = load_everything()

    # ---------------- Explicación ----------------
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        ui.section_title("¿Qué hace K-Nearest Neighbors?", None, "fa-brain")
        st.markdown(
            """
<div class="step-list">
  <div class="step"><div class="num">1</div><div>
    <h5>Normalización</h5>
    <p>Las tres variables tienen escalas distintas (magnitud ≈ 5, profundidad ≈ 50 km,
    distancia ≈ 150 km). <b>StandardScaler</b> las lleva a media 0 y desviación 1 dentro del
    Pipeline, de modo que ninguna domina el cálculo de distancias y no hay fuga de información.</p>
  </div></div>
  <div class="step"><div class="num">2</div><div>
    <h5>Cálculo de distancias</h5>
    <p>Para un escenario nuevo se mide la distancia euclidiana en el espacio normalizado
    contra todos los registros de entrenamiento.</p>
  </div></div>
  <div class="step"><div class="num">3</div><div>
    <h5>Selección de K vecinos</h5>
    <p>Se toman los K registros más cercanos. Cada vecino aporta su categoría de intensidad
    observada (Baja, Moderada o Fuerte).</p>
  </div></div>
  <div class="step"><div class="num">4</div><div>
    <h5>Votación por mayoría</h5>
    <p>La categoría con más votos es la estimación. La proporción de votos se expone como
    <code>predict_proba()</code>: <b>no</b> es una probabilidad científica calibrada de un terremoto.</p>
  </div></div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        ui.section_title("Espacio de características", None, "fa-cube")
        st.markdown(
            f'<div class="glass">{ui.info_rows([("Variables predictoras", "magnitud · profundidad_km · distancia_km"), ("Variable objetivo", "categoria_intensidad"), ("Clases", "Baja (I–IV) · Moderada (V–VI) · Fuerte (VII–XII)"), ("Escalado", "StandardScaler (dentro del Pipeline)"), ("Métrica", "Euclidiana")])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="formula">d(x, q) = √ Σᵢ ( zᵢ(x) − zᵢ(q) )²</div>',
            unsafe_allow_html=True,
        )
        st.caption("Donde zᵢ son las variables estandarizadas (μ = 0, σ = 1).")

    st.divider()

    # ---------------- División y entrenamiento ----------------
    if not require_model(artifact):
        return

    meta = artifact["metadata"]
    ds = meta.get("dataset", {})

    ui.section_title("División de datos y entrenamiento", None, "fa-split")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros de entrenamiento", f"{ds.get('train_registros', 0)}",
              f"{ds.get('train_eventos', 0)} eventos")
    c2.metric("Registros de prueba", f"{ds.get('test_registros', 0)}",
              f"{ds.get('test_eventos', 0)} eventos")
    c3.metric("Estrategia de split", "Agrupada 80/20",
              f"random_state = {meta.get('random_state')}")
    c4.metric("Validación cruzada", f"{meta.get('cv_folds', '—')} pliegues",
              meta.get("cv_estrategia", "—"))

    ui.alert(
        "info",
        "La división usa <b>GroupShuffleSplit por evento_id</b>: todas las localidades de un mismo "
        "evento permanecen juntas en entrenamiento o en prueba, evitando que el modelo memorice "
        "eventos vistos.",
    )

    # ---------------- Comparación de K ----------------
    ui.section_title(
        "Comparación de valores de K",
        "Accuracy, precisión, recall y F1-score macro con validación cruzada agrupada "
        "exclusivamente dentro del entrenamiento.",
        "fa-chart-line",
    )

    cv_res = meta.get("cv_resultados", {})
    best_k = meta.get("k_seleccionado")

    tab_table, tab_chart, tab_folds = st.tabs(
        ["Tabla comparativa", "Gráfico comparativo", "Detalle por pliegue"]
    )
    with tab_table:
        comp = ev.k_comparison_df({int(k): v for k, v in cv_res.items()})
        display = comp.copy()
        for col in ("Accuracy", "Precision macro", "Recall macro", "F1-score macro"):
            display[col] = (display[col] * 100).round(1).astype(str) + " %"
        display["Selección"] = [
            "★ Mejor F1" if int(k) == best_k else "" for k in comp["K"]
        ]
        st.dataframe(display, width='stretch', hide_index=True)
        st.info(f"**Criterio de selección:** {meta.get('criterio_seleccion', '')}")

    with tab_chart:
        st.plotly_chart(charts.k_comparison_chart(cv_res), width='stretch',
                        config={"displayModeBar": False})

    with tab_folds:
        folds = meta.get("cv_folds_detalle", [])
        if folds:
            import pandas as pd
            fdf = pd.DataFrame(folds)
            st.dataframe(
                fdf.round(4), width='stretch', hide_index=True, height=340
            )
            st.caption(
                "Cada fila corresponde a un pliegue de la validación cruzada agrupada. "
                "Las métricas se promedian para comparar los valores de K."
            )
        else:
            ui.alert("warn", "No hay detalle de pliegues disponible.")

    # ---------------- Configuración final ----------------
    ui.section_title("Configuración del modelo final", None, "fa-sliders")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        f1_test = f"{meta.get('metricas_test', {}).get('f1_macro', 0) * 100:.1f}%"
        acc_test = f"{meta.get('metricas_test', {}).get('accuracy', 0) * 100:.1f}%"
        rows = [
            ("Pipeline", "StandardScaler → KNeighborsClassifier"),
            ("K seleccionado", f"<b>{best_k}</b>"),
            ("Pesos", "uniform"),
            ("Métrica", "euclidiana"),
            ("F1 macro (test)", f1_test),
            ("Accuracy (test)", acc_test),
        ]
        st.markdown(
            f'<div class="glass">{ui.info_rows(rows)}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="glass"><div class="section"><h3>¿Por qué no reentrenar en cada página?</h3>'
            '<p class="section-sub">El entrenamiento completo (comparación de K + validación cruzada) '
            'se ejecuta una sola vez y sus artefactos se guardan con Joblib. La aplicación carga el '
            'pipeline persistido, garantizando resultados idénticos y arranques rápidos.</p></div></div>',
            unsafe_allow_html=True,
        )
        conf = st.checkbox(
            "Confirmo que deseo reentrenar el modelo (puede tomar unos segundos)"
        )
        if st.button("🔄 Reentrenar modelo", disabled=not conf, type="secondary"):
            with st.spinner("Entrenando pipeline y comparando K = 3, 5, 7, 9…"):
                try:
                    training_service.retrain()
                    clear_caches()
                    st.success("Modelo reentrenado y guardado correctamente.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    ui.alert("error", f"No se pudo reentrenar: {exc}")

    ui.section_title("Proyección 2D del espacio de entrenamiento", None, "fa-braille")
    proj = artifact.get("projection", {})
    if proj:
        train_df = artifact["train_data"]
        st.plotly_chart(
            charts.neighbor_projection(train_df, proj["coords"], None, None,
                                       predicted="Baja"),
            width='stretch', config={"displayModeBar": False},
        )
        var = proj.get("explained_variance", [0, 0])
        st.caption(
            f"Proyección PCA de las 3 variables normalizadas (varianza explicada: "
            f"{var[0]*100:.1f}% + {var[1]*100:.1f}%). Es solo una visualización: la clasificación "
            "usa las 3 dimensiones completas."
        )

    ui.disclaimer_footer()
