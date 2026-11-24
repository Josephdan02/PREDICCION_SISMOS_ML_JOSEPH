"""Graficos Plotly reutilizables con la identidad visual de SISMO AI PERU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config

C = config.CLASS_COLORS
P = config.PALETTE


def apply_theme(fig: go.Figure, height: int = 300, show_legend: bool = True) -> go.Figure:
    """Aplica el tema oscuro glass al grafico."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=P["text"], size=12),
        margin=dict(l=10, r=10, t=36, b=10),
        height=height,
        showlegend=show_legend,
        colorway=config.PLOTLY_SEQUENCE,
        hoverlabel=dict(bgcolor="#101830", bordercolor=P["border"],
                        font=dict(color=P["text"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    return fig


def histogram(df: pd.DataFrame, column: str, title: str, nbins: int = 24,
              color: str | None = None, height: int = 280) -> go.Figure:
    color = color or P["blue"]
    fig = px.histogram(df, x=column, nbins=nbins, title=title)
    fig.update_traces(marker_color=color, marker_line_width=0,
                      opacity=0.85, hovertemplate="%{x}<br>Registros: %{y}<extra></extra>")
    apply_theme(fig, height=height, show_legend=False)
    fig.update_layout(title_font=dict(size=13), bargap=0.06)
    return fig


def class_bar(df: pd.DataFrame, height: int = 280, title: str = "Distribución de intensidades") -> go.Figure:
    dist = df["categoria_intensidad"].value_counts().reindex(config.CLASS_ORDER).fillna(0)
    values = [int(v) for v in dist.values]
    fig = go.Figure(go.Bar(
        x=config.CLASS_ORDER, y=values,
        marker_color=[C[k] for k in config.CLASS_ORDER],
        text=values, textposition="outside",
        hovertemplate="%{x}: %{y} registros<extra></extra>",
    ))
    apply_theme(fig, height=height, show_legend=False)
    fig.update_layout(title=title, title_font=dict(size=13), yaxis_title=None,
                      bargap=0.45)
    fig.update_yaxes(range=[0, max(max(values, default=1) * 1.2, 1)])
    return fig


def department_bar(df: pd.DataFrame, top_n: int = 12, height: int = 320) -> go.Figure:
    counts = df["departamento"].value_counts().head(top_n).sort_values()
    values = [int(v) for v in counts.values]
    fig = go.Figure(go.Bar(
        x=values, y=list(counts.index), orientation="h",
        marker=dict(color=values,
                    colorscale=[[0, "#1e3a8a"], [0.55, "#6d5bd0"], [1, "#22d3ee"]]),
        hovertemplate="%{y}: %{x} registros<extra></extra>",
    ))
    apply_theme(fig, height=height, show_legend=False)
    fig.update_layout(title=f"Eventos por departamento (top {top_n})",
                      title_font=dict(size=13))
    return fig


def box_by_class(df: pd.DataFrame, column: str, title: str, height: int = 290) -> go.Figure:
    fig = px.box(df, x="categoria_intensidad", y=column, color="categoria_intensidad",
                 category_orders={"categoria_intensidad": config.CLASS_ORDER},
                 color_discrete_map=C, points=False)
    apply_theme(fig, height=height, show_legend=False)
    fig.update_layout(title=title, title_font=dict(size=13), xaxis_title=None)
    return fig


def scatter_features(df: pd.DataFrame, height: int = 360) -> go.Figure:
    sample = df.sample(min(len(df), 900), random_state=config.RANDOM_STATE)
    fig = px.scatter(
        sample, x="distancia_km", y="magnitud", color="categoria_intensidad",
        size="intensidad_observada",
        color_discrete_map=C, category_orders={"categoria_intensidad": config.CLASS_ORDER},
        hover_data=["localidad", "profundidad_km", "intensidad_observada"],
        render_mode="webgl" if len(sample) > 400 else "svg",
    )
    apply_theme(fig, height=height)
    fig.update_layout(title="Magnitud vs. distancia epicentral por categoría",
                      title_font=dict(size=13))
    fig.update_traces(marker=dict(sizemode="diameter", sizeref=0.6))
    return fig


def k_comparison_chart(cv_results: dict, height: int = 340) -> go.Figure:
    ks = sorted(int(k) for k in cv_results)
    metrics = [("accuracy", "Accuracy", P["blue"]),
               ("precision_macro", "Precision macro", P["cyan"]),
               ("recall_macro", "Recall macro", P["amber"]),
               ("f1_macro", "F1-score macro", P["violet"])]
    fig = go.Figure()
    for key, label, color in metrics:
        fig.add_trace(go.Scatter(
            x=ks, y=[cv_results[str(k)][key] if str(k) in cv_results else cv_results[k][key] for k in ks],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.4),
            marker=dict(size=8, line=dict(width=1.5, color="#0b1120")),
        ))
    apply_theme(fig, height=height)
    fig.update_layout(title="Validación cruzada agrupada por K",
                      title_font=dict(size=13),
                      xaxis=dict(title="K", tickmode="array", tickvals=ks))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return fig


def probability_chart(probabilities: dict, height: int = 300,
                      predicted: str | None = None) -> go.Figure:
    cats = config.CLASS_ORDER
    values = [probabilities.get(c, 0) for c in cats]
    colors = [C[c] for c in cats]
    opacity = [1.0 if (predicted is None or c == predicted) else 0.38 for c in cats]
    fig = go.Figure(go.Bar(
        x=cats, y=values, marker=dict(color=colors, opacity=opacity),
        text=[f"{v*100:.0f}%" for v in values], textposition="outside",
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    apply_theme(fig, height=height, show_legend=False)
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
    fig.update_layout(title="Distribución de votos de los vecinos",
                      title_font=dict(size=13), bargap=0.5)
    return fig


def vote_donut(votes: dict, height: int = 260) -> go.Figure:
    labels = [c for c in config.CLASS_ORDER if votes.get(c, 0) > 0] or list(votes.keys())
    values = [votes.get(c, 0) for c in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=[C.get(c, P["blue"]) for c in labels],
                    line=dict(color="#0b1120", width=2)),
        textinfo="label+value", textfont=dict(size=11),
        hovertemplate="%{label}: %{value} votos<extra></extra>",
    ))
    apply_theme(fig, height=height, show_legend=False)
    fig.update_layout(title="Votos por clase", title_font=dict(size=13),
                      margin=dict(l=10, r=10, t=40, b=6))
    return fig


def confusion_heatmap(cm_dict: dict, height: int = 360) -> go.Figure:
    """Matriz de confusion: filas = real, columnas = predicho."""
    matrix = [[int(cm_dict.get(f"Real {r}", {}).get(f"Pred {c}", 0))
               for c in config.CLASS_ORDER] for r in config.CLASS_ORDER]
    labels = config.CLASS_ORDER
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"Pred {l}" for l in labels],
        y=[f"Real {l}" for l in labels],
        colorscale=[[0, "#0d1428"], [0.4, "#4f8cff"], [1, "#a78bfa"]],
        text=matrix, texttemplate="%{text}", textfont=dict(size=15),
        hovertemplate="%{y} → %{x}: %{z} casos<extra></extra>",
        showscale=True,
    ))
    apply_theme(fig, height=height, show_legend=False)
    fig.update_layout(title="Matriz de confusión — conjunto de prueba",
                      title_font=dict(size=13))
    fig.update_yaxes(autorange="reversed")
    return fig


def neighbor_projection(df: pd.DataFrame, coords: np.ndarray, query_2d, neighbor_2d,
                        predicted: str, height: int = 420) -> go.Figure:
    """Proyeccion PCA 2D: nube de entrenamiento + consulta + vecinos."""
    fig = go.Figure()
    for cat in config.CLASS_ORDER:
        mask = (df["categoria_intensidad"] == cat).to_numpy()
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=cat,
            marker=dict(size=6, color=C[cat], opacity=0.35),
            hovertemplate=f"{cat}<extra></extra>",
        ))
    if neighbor_2d is not None and len(neighbor_2d):
        fig.add_trace(go.Scatter(
            x=neighbor_2d[:, 0], y=neighbor_2d[:, 1], mode="markers",
            name="Vecinos seleccionados",
            marker=dict(size=13, color="#fbbf24", symbol="diamond",
                        line=dict(color="#0b1120", width=1.5)),
            hovertemplate="Vecino<extra></extra>",
        ))
    if query_2d is not None:
        fig.add_trace(go.Scatter(
            x=[query_2d[0]], y=[query_2d[1]], mode="markers", name="Escenario evaluado",
            marker=dict(size=18, color="#ffffff", symbol="star",
                        line=dict(color=C.get(predicted, "#fff"), width=3)),
            hovertemplate="Escenario evaluado<extra></extra>",
        ))
    apply_theme(fig, height=height)
    fig.update_layout(title="Espacio normalizado (proyección PCA 2D de las 3 variables)",
                      title_font=dict(size=13),
                      xaxis_title="Componente 1", yaxis_title="Componente 2")
    return fig


def magnitude_vs_depth_3d(df: pd.DataFrame, height: int = 420) -> go.Figure:
    sample = df.sample(min(len(df), 700), random_state=config.RANDOM_STATE)
    fig = px.scatter_3d(
        sample, x="magnitud", y="profundidad_km", z="distancia_km",
        color="categoria_intensidad", color_discrete_map=C,
        category_orders={"categoria_intensidad": config.CLASS_ORDER},
        hover_data=["localidad"], opacity=0.75,
    )
    apply_theme(fig, height=height)
    fig.update_traces(marker=dict(size=3.5))
    fig.update_layout(title="Espacio de características 3D",
                      title_font=dict(size=13),
                      scene=dict(
                          xaxis=dict(backgroundcolor="rgba(0,0,0,0)",
                                     gridcolor="rgba(255,255,255,0.07)"),
                          yaxis=dict(backgroundcolor="rgba(0,0,0,0)",
                                     gridcolor="rgba(255,255,255,0.07)"),
                          zaxis=dict(backgroundcolor="rgba(0,0,0,0)",
                                     gridcolor="rgba(255,255,255,0.07)"),
                      ))
    return fig


def gauge(value: float, title: str, height: int = 200, color: str | None = None) -> go.Figure:
    color = color or P["violet"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(0.0, min(1.0, float(value))) * 100,
        number=dict(suffix="%", font=dict(size=26, color=P["text"])),
        title=dict(text=title, font=dict(size=12, color=P["muted"])),
        gauge=dict(
            axis=dict(range=[0, 100], showticklabels=False, ticks=""),
            bar=dict(color=color, thickness=0.30),
            bgcolor="rgba(255,255,255,0.05)",
            borderwidth=0,
            steps=[dict(range=[0, 100], color="rgba(255,255,255,0.02)")],
        ),
    ))
    apply_theme(fig, height=height, show_legend=False)
    fig.update_layout(margin=dict(l=24, r=24, t=36, b=4))
    return fig
