"""Componentes visuales reutilizables (HTML/CSS) para la interfaz Streamlit."""

from __future__ import annotations

import html

import streamlit as st

from src import config

C = config.CLASS_COLORS


def esc(text) -> str:
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# Encabezados
# ---------------------------------------------------------------------------
def institutional_header(subtitle: str | None = None) -> None:
    subtitle = subtitle or config.PROJECT_TAGLINE
    st.markdown(
        f"""
        <div class="institutional">
          <div class="inst-left">
            <div class="inst-brand">
              <span class="inst-logo"><i class="fa-solid fa-hill-rockslide"></i></span>
              <div>
                <div class="inst-eyebrow">{esc(config.INSTITUTION)}</div>
                <div class="inst-title">{esc(config.PROJECT_NAME)}</div>
              </div>
            </div>
            <div class="inst-sub">{esc(subtitle)}</div>
          </div>
          <div class="inst-right">
            <span class="pill pill-live"><span class="dot"></span> SISTEMA ACTIVO</span>
            <span class="pill">KNN</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str, subtitle: str, eyebrow: str = "MÓDULO") -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <div class="eyebrow">{esc(eyebrow)}</div>
          <h1>{esc(title)}</h1>
          <p>{esc(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str | None = None, icon: str = "") -> None:
    """Cabecera de seccion. ``icon`` recibe el sufijo, p. ej. "fa-flask".

    Se usa un <div> en lugar de <h3>: Streamlit reescribe el contenido de los
    encabezados y elimina las clases del <i>, lo que impediria cargar el icono.
    """
    if icon:
        icon_html = f'<i class="fa-solid {icon}"></i>'
    else:
        icon_html = ""
    sub = f'<p class="section-sub">{esc(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section"><div class="section-title">{icon_html}'
        f'<span>{esc(title)}</span></div>{sub}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tarjetas
# ---------------------------------------------------------------------------
def stat_card(label: str, value, unit: str = "", icon: str = "fa-chart-simple",
              accent: str = "#a78bfa", hint: str | None = None) -> str:
    hint_html = f'<div class="stat-hint">{esc(hint)}</div>' if hint else ""
    return (
        f'<div class="stat-card" style="--accent:{accent}">'
        f'<div class="stat-top">'
        f'<span class="stat-label">{esc(label)}</span>'
        f'<span class="stat-icon"><i class="fa-solid {icon}"></i></span>'
        f'</div>'
        f'<div class="stat-value">{esc(value)}<span class="stat-unit">{esc(unit)}</span></div>'
        f'{hint_html}'
        f'</div>'
    )


def stat_grid(cards: list[str]) -> None:
    """Dibuja las tarjetas usando columnas de Streamlit (layout nativo y robusto)."""
    if not cards:
        return
    cols = st.columns(min(len(cards), 6), gap="small")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)


def glass_panel(content_html: str, extra_class: str = "") -> None:
    st.markdown(
        f'<div class="glass {extra_class}">{content_html}</div>', unsafe_allow_html=True
    )


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{esc(kind)}">{esc(text)}</span>'


def category_badge(category: str) -> str:
    color = C.get(category, "#94a3b8")
    return (
        f'<span class="cat-badge" style="--cat:{color}">'
        f'<span class="cat-dot"></span>{esc(category)}</span>'
    )


# ---------------------------------------------------------------------------
# Resultado del simulador
# ---------------------------------------------------------------------------
def result_hero(category: str, confidence: float, k: int, locality: str) -> None:
    color = C.get(category, "#a78bfa")
    st.markdown(
        f"""
        <div class="result-hero" style="--cat:{color}">
          <div class="rh-left">
            <div class="rh-label">INTENSIDAD ESTIMADA</div>
            <div class="rh-category">{esc(category.upper())}</div>
            <div class="rh-meta">Escenario: <b>{esc(locality)}</b> · K = {k}</div>
          </div>
          <div class="rh-right">
            <div class="rh-conf">{confidence*100:.0f}<span>%</span></div>
            <div class="rh-conf-label">confianza de los vecinos</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_rows(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="kv"><span>{esc(k)}</span><b>{v}</b></div>' for k, v in rows
    )
    return f'<div class="kv-list">{body}</div>'


def model_card(meta: dict) -> None:
    k = meta.get("k_seleccionado")
    metrics = meta.get("metricas_test", {})
    rows = [
        ("Algoritmo", "K-Nearest Neighbors"),
        ("Pipeline", "StandardScaler → KNN"),
        ("K seleccionado", f"<b>{k}</b>"),
        ("Accuracy (test)", f"{metrics.get('accuracy', 0)*100:.1f}%"),
        ("F1-score macro", f"{metrics.get('f1_macro', 0)*100:.1f}%"),
        ("Entrenado", esc(meta.get("fecha_entrenamiento", "—"))[:16]),
    ]
    body = "".join(f'<div class="kv"><span>{a}</span><b>{b}</b></div>' for a, b in rows)
    st.markdown(f'<div class="glass model-card">{body}</div>', unsafe_allow_html=True)


def alert(kind: str, message: str) -> None:
    icons = {"info": "fa-circle-info", "warn": "fa-triangle-exclamation",
             "error": "fa-circle-xmark", "ok": "fa-circle-check"}
    st.markdown(
        f'<div class="alert alert-{kind}"><i class="fa-solid {icons.get(kind, "fa-circle-info")}"></i>'
        f"<span>{message}</span></div>",
        unsafe_allow_html=True,
    )


def disclaimer_footer() -> None:
    st.markdown(
        f'<div class="disclaimer"><i class="fa-solid fa-shield-halved"></i> '
        f"<span>{esc(config.DISCLAIMER)}</span></div>",
        unsafe_allow_html=True,
    )


def kpi_inline(items: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="kpi-inline"><span>{esc(a)}</span><b>{b}</b></div>' for a, b in items
    )
    return f'<div class="kpi-inline-wrap">{body}</div>'
