"""Servicio de mapas: construye el mapa Folium de eventos sismicos."""

from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

from . import config


def _marker_color(magnitud: float) -> str:
    """Color del marcador segun magnitud (no segun la categoria estimada)."""
    if magnitud >= 7.0:
        return "#b91c1c"
    if magnitud >= 6.0:
        return "#ef4444"
    if magnitud >= 5.0:
        return "#f97316"
    if magnitud >= 4.0:
        return "#22d3ee"
    return "#3b82f6"


# Teselas oscuras: ArcGIS World Dark Gray (Esri).
# Se eligio este proveedor porque CARTO comenzó a exigir API key (devuelve una
# imagen con marca de agua) y OSM estándar es de fondo claro poco legible.
DARK_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
DARK_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, OpenStreetMap contributors"
# Capa de referencias (fronteras y nombres) para el mapa principal.
LABELS_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
)


def filter_events(
    df: pd.DataFrame,
    departamento: str | None = None,
    mag_range: tuple[float, float] | None = None,
    prof_range: tuple[float, float] | None = None,
    categoria: str | None = None,
) -> pd.DataFrame:
    """Aplica filtros de la interfaz sobre el dataset georreferenciado."""
    out = df.copy()
    if departamento and departamento != "Todos":
        out = out[out["departamento"] == departamento]
    if mag_range:
        out = out[out["magnitud"].between(*mag_range)]
    if prof_range:
        out = out[out["profundidad_km"].between(*prof_range)]
    if categoria and categoria != "Todas":
        out = out[out["categoria_intensidad"] == categoria]
    return out


def valid_coords(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Filtra registros con coordenadas validas. Devuelve (df, excluidos)."""
    mask = df["latitud"].between(-90, 90) & df["longitud"].between(-180, 180)
    return df[mask.fillna(False)], int((~mask.fillna(False)).sum())


def build_events_map(
    df: pd.DataFrame,
    max_points: int = 800,
    show_huanuco: bool = True,
    use_cluster: bool = True,
    heatmap: bool = False,
) -> tuple[folium.Map, int, int]:
    """Construye el mapa de epicentros.

    Returns:
        (mapa, n_mostrados, n_excluidos_por_coordenadas)
    """
    geo, excluded = valid_coords(df)
    plotted = geo.head(max_points)

    fmap = folium.Map(
        location=config.PERU_CENTER,
        zoom_start=config.PERU_ZOOM,
        tiles=DARK_TILES,
        attr=DARK_ATTR,
        control_scale=True,
        prefer_canvas=True,
    )
    # Etiquetas (fronteras, ciudades) sobre el fondo oscuro.
    folium.TileLayer(
        tiles=LABELS_TILES, attr=DARK_ATTR, name="Etiquetas",
        overlay=True, control=False, show=True,
    ).add_to(fmap)

    if heatmap and not plotted.empty:
        HeatMap(
            plotted[["latitud", "longitud"]].values.tolist(),
            radius=12, blur=18, min_opacity=0.35,
        ).add_to(fmap)

    group = MarkerCluster(name="Eventos sísmicos") if use_cluster else folium.FeatureGroup(
        name="Eventos sísmicos")
    if not use_cluster:
        group.add_to(fmap)

    for _, row in plotted.iterrows():
        categoria = row["categoria_intensidad"]
        popup_html = f"""
        <div style="font-family: Inter, Segoe UI, sans-serif; font-size:12px; min-width:190px">
          <b style="color:#5b21b6">{row['localidad']} — {row['departamento']}</b><br>
          <span style="color:#6b7280">Evento</span> <b>{row['evento_id']}</b><br>
          <span style="color:#6b7280">Fecha</span> {pd.to_datetime(row['fecha']).date() if pd.notna(row['fecha']) else 'N/D'}<br>
          <span style="color:#6b7280">Magnitud</span> <b>{row['magnitud']}</b> {row['escala_magnitud'] or ''}<br>
          <span style="color:#6b7280">Profundidad</span> {row['profundidad_km']} km<br>
          <span style="color:#6b7280">Distancia epicentral</span> {row['distancia_km']} km<br>
          <span style="color:#6b7280">Intensidad observada</span> MMI {row['intensidad_observada']}<br>
          <span style="color:#6b7280">Categoría</span> <b>{categoria}</b><br>
          <span style="color:#6b7280">Registro</span> {row['tipo_registro']}
        </div>
        """
        folium.CircleMarker(
            location=[row["latitud"], row["longitud"]],
            radius=max(3, min(9, float(row["magnitud"]) * 1.1)),
            color=_marker_color(float(row["magnitud"])),
            weight=1,
            fill=True,
            fill_color=_marker_color(float(row["magnitud"])),
            fill_opacity=0.72,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['localidad']} · M{row['magnitud']} · {categoria}",
        ).add_to(group)

    if not use_cluster:
        pass
    else:
        group.add_to(fmap)

    if show_huanuco:
        folium.CircleMarker(
            location=config.HUANUCO_COORDS,
            radius=9,
            color="#a78bfa",
            weight=2,
            fill=True,
            fill_color="#a78bfa",
            fill_opacity=0.9,
            tooltip="Huánuco (referencia)",
            popup="<b>Huánuco</b><br>Localidad de referencia del proyecto.",
        ).add_to(fmap)
        folium.Circle(
            location=config.HUANUCO_COORDS, radius=28000, color="#a78bfa",
            fill=True, fill_opacity=0.10, weight=1,
        ).add_to(fmap)

    # Leyenda por magnitud.
    legend = f"""
    <div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999;
                background: rgba(10,16,34,0.88); color:#e9eefc; padding:12px 14px;
                border-radius:12px; border:1px solid rgba(255,255,255,.14);
                font-family: Inter, Segoe UI, sans-serif; font-size:12px; line-height:1.7;">
      <b style="letter-spacing:.08em; font-size:11px; color:#a78bfa">MAGNITUD</b><br>
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#3b82f6;margin-right:7px"></span>&lt; 4.0<br>
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#22d3ee;margin-right:7px"></span>4.0 – 4.9<br>
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#f97316;margin-right:7px"></span>5.0 – 5.9<br>
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#ef4444;margin-right:7px"></span>6.0 – 6.9<br>
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#b91c1c;margin-right:7px"></span>&ge; 7.0<br>
      <span style="color:#97a3c2; font-size:10.5px">El tamaño crece con la magnitud</span>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=True).add_to(fmap)

    return fmap, len(plotted), excluded


def location_map(lat: float, lon: float, label: str = "Escenario") -> folium.Map:
    """Mini mapa centrado en la localidad del escenario."""
    fmap = folium.Map(location=[lat, lon], zoom_start=7, tiles=DARK_TILES,
                      attr=DARK_ATTR, control_scale=False)
    folium.TileLayer(
        tiles=LABELS_TILES, attr=DARK_ATTR, name="Etiquetas",
        overlay=True, control=False, show=True,
    ).add_to(fmap)
    folium.CircleMarker(
        [lat, lon], radius=9, color="#a78bfa", weight=2, fill=True,
        fill_color="#a78bfa", fill_opacity=0.9, tooltip=label,
    ).add_to(fmap)
    folium.Circle([lat, lon], radius=30000, color="#a78bfa",
                  fill=True, fill_opacity=0.08, weight=1).add_to(fmap)
    return fmap
