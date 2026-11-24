"""SISMO AI PERÚ — Sistema Inteligente de Análisis y Estimación Sísmica con KNN.

Aplicación principal (Streamlit). Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

st.set_page_config(
    page_title="SISMO AI PERÚ · KNN",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilos e iconografía
# ---------------------------------------------------------------------------
css = config.STYLES_PATH.read_text(encoding="utf-8") if config.STYLES_PATH.exists() else ""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navegación lateral
# ---------------------------------------------------------------------------
PAGES = {
    "Dashboard": "dashboard",
    "Análisis KNN": "analisis_knn",
    "Simulador sísmico": "simulador",
    "Mapa de eventos": "mapa_eventos",
    "Dataset": "dataset",
    "Evaluación del modelo": "evaluacion",
    "Reportes": "reportes",
    "Acerca del proyecto": "acerca",
}

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
          <div class="logo"><i class="fa-solid fa-wave-square"></i></div>
          <div>
            <div class="tag">SENATI · IA</div>
            <div class="name">SISMO AI PERÚ</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:10px;letter-spacing:.22em;color:#6b7897;'
        'font-weight:700;margin:6px 0 10px 4px;">NAVEGACIÓN</div>',
        unsafe_allow_html=True,
    )
    selection = st.radio("Navegación", list(PAGES.keys()), label_visibility="collapsed")

    # Estado del sistema calculado con datos reales (si es posible).
    try:
        from pages._shared import cached_artifact, cached_summary
        summary = cached_summary()
        artifact = cached_artifact()
        k_val = artifact["metadata"].get("k_seleccionado")
        acc = artifact["metadata"].get("metricas_test", {}).get("accuracy", 0) * 100
        model_ok = True
    except Exception:  # noqa: BLE001 - la sidebar nunca debe romper la app
        summary, k_val, acc, model_ok = {}, "—", 0, False

    st.markdown(
        f"""
        <div class="side-status">
          <b style="color:{'#7ef0c0' if model_ok else '#fde68a'}">
            {'● MODELO LISTO' if model_ok else '● SIN MODELO'}</b>
          <div class="row"><span>Registros</span><b>{summary.get('total_registros', 0)}</b></div>
          <div class="row"><span>Eventos</span><b>{summary.get('eventos', 0)}</b></div>
          <div class="row"><span>K del modelo</span><b>{k_val}</b></div>
          <div class="row"><span>Accuracy test</span><b>{acc:.1f}%</b></div>
        </div>
        <div style="font-size:10px;color:#4b5674;padding:10px 6px;line-height:1.5">
          Uso académico · SENATI<br>No es un sistema de alerta temprana.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Enrutado
# ---------------------------------------------------------------------------
module = PAGES[selection]
PAGE_MODULES = {
    "dashboard": ("pages.landing", "render"),
    "analisis_knn": ("pages.analisis_knn", "render"),
    "simulador": ("pages.simulador", "render"),
    "mapa_eventos": ("pages.mapa_eventos", "render"),
    "dataset": ("pages.dataset", "render"),
    "evaluacion": ("pages.evaluacion", "render"),
    "reportes": ("pages.reportes", "render"),
    "acerca": ("pages.acerca", "render"),
}

try:
    import importlib
    mod_name, func_name = PAGE_MODULES[module]
    page_module = importlib.import_module(mod_name)
    getattr(page_module, func_name)()
except Exception as exc:  # noqa: BLE001 - manejo de errores visible
    st.error(f"No se pudo cargar la sección '{selection}': {exc}")
    with st.expander("Detalle del error"):
        st.exception(exc)
