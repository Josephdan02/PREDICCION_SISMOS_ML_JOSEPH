"""Página: Generación de informe PDF y exportación de resultados."""

from __future__ import annotations

import streamlit as st

from pages._shared import (
    build_pdf,
    clear_history,
    get_history,
    load_everything,
    require_model,
    save_history_csv,
    to_csv_bytes,
)
from src import config, ui


def render() -> None:
    ui.institutional_header()
    ui.page_title(
        "Reportes y exportación",
        "Genera un informe PDF con datos y métricas reales del sistema, y exporta los "
        "resultados de las simulaciones realizadas.",
        eyebrow="DOCUMENTACIÓN",
    )

    df, summary, artifact = load_everything()
    if not require_model(artifact):
        return
    meta = artifact["metadata"]
    metrics = meta.get("metricas_test", {})
    history = get_history()

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        ui.section_title("Informe técnico PDF", None, "fa-file-pdf")
        k_sel = meta.get("k_seleccionado")
        acc_pct = f"{metrics.get('accuracy', 0) * 100:.1f}%"
        f1_pct = f"{metrics.get('f1_macro', 0) * 100:.1f}%"
        rows = [
            ("Proyecto", config.PROJECT_NAME),
            ("Modelo", f"KNN · K = {k_sel}"),
            ("Accuracy (test)", acc_pct),
            ("F1 macro (test)", f1_pct),
            ("Registros del dataset", f"{summary.get('total_registros', 0)}"),
            ("Simulaciones registradas", f"{len(history)}"),
        ]
        st.markdown(
            f'<div class="glass">{ui.info_rows(rows)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "El informe incluye objetivo, descripción y procedencia del dataset, preparación, "
            "metodología KNN, comparación de K, configuración, evaluación, matriz de confusión, "
            "simulaciones recientes, conclusiones y limitaciones."
        )
        if st.button("📄 Generar informe PDF", type="primary", width='stretch'):
            with st.spinner("Construyendo el informe con datos reales…"):
                try:
                    pdf_bytes = build_pdf(artifact, summary)
                    st.session_state["pdf_bytes"] = pdf_bytes
                    config.ensure_directories()
                    out = config.REPORTS_DIR / "informe_sismo_ai_peru.pdf"
                    out.write_bytes(pdf_bytes)
                    st.success(f"Informe generado ({len(pdf_bytes)/1024:.0f} KB).")
                except Exception as exc:  # noqa: BLE001
                    ui.alert("error", f"No se pudo generar el PDF: {exc}")

        if "pdf_bytes" in st.session_state:
            st.download_button(
                "⬇️ Descargar informe PDF",
                data=st.session_state["pdf_bytes"],
                file_name="SISMO_AI_PERU_informe.pdf",
                mime="application/pdf",
                width='stretch',
            )

    with right:
        ui.section_title("Exportar resultados", None, "fa-file-csv")
        st.markdown(
            f'<div class="glass"><div class="section"><h3>Simulaciones de la sesión</h3>'
            f'<p class="section-sub">Cada estimación ejecutada con el modelo KNN se registra '
            f'automáticamente.</p></div>'
            f'<span class="badge badge-info">{len(history)} registro(s)</span></div>',
            unsafe_allow_html=True,
        )
        if history.empty:
            ui.alert("warn", "Aún no has ejecutado simulaciones. Ve al <b>Simulador sísmico</b> "
                             "y ejecuta el modelo; aquí aparecerán tus resultados.")
        else:
            st.dataframe(history, width='stretch', hide_index=True, height=220)
            e1, e2 = st.columns(2)
            with e1:
                st.download_button(
                    "⬇️ CSV de simulaciones",
                    data=to_csv_bytes(history),
                    file_name="simulaciones_sismo_ai.csv",
                    mime="text/csv",
                    width='stretch',
                )
            with e2:
                if st.button("💾 Guardar en reports/", width='stretch'):
                    path = save_history_csv(history)
                    if path:
                        st.success(f"Guardado en {path}")
            if st.button("🗑️ Limpiar historial de la sesión", width='stretch'):
                clear_history()
                st.rerun()

        st.divider()
        st.download_button(
            "⬇️ Dataset completo (CSV)",
            data=to_csv_bytes(df),
            file_name="sismos_peru.csv",
            mime="text/csv",
            width='stretch',
        )

        # Exportación de las métricas del modelo.
        import json
        meta_json = json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "⬇️ Metadatos del modelo (JSON)",
            data=meta_json,
            file_name="modelo_metadata.json",
            mime="application/json",
            width='stretch',
        )

    st.divider()
    ui.section_title("Contenido del informe", None, "fa-list-ol")
    st.markdown(
        """
1. **Objetivo** — alcance académico del sistema y definición de la tarea de clasificación.
2. **Descripción del dataset** — registros, eventos, departamentos y tipos de registro.
3. **Procedencia y naturaleza de los datos** — diferencia entre catálogos de magnitud e
   intensidad observada, y declaración explícita del dataset simulado.
4. **Preparación y limpieza** — nulos, duplicados, rangos físicos y agrupación de intensidades.
5. **Metodología KNN** — Pipeline, normalización, división agrupada y validación cruzada.
6. **Comparación de K** — tabla de accuracy, precision, recall y F1 macro, con K seleccionado.
7. **Resultados de evaluación** — métricas del conjunto de prueba independiente.
8. **Matriz de confusión** e informe por categoría.
9. **Registros de simulaciones** — últimas estimaciones realizadas en la sesión.
10. **Conclusiones y limitaciones** — alcance real de los resultados.
        """
    )
    ui.disclaimer_footer()
