"""Cliente opcional para el catalogo real de USGS (ComCat).

Solo se utiliza para intentar incorporar eventos REALES con intensidad MMI
estimada (ShakeMap) o CDI (intensidad reportada por la comunidad). Si la red
no esta disponible o no hay campos utiles, el sistema funciona igualmente con
el dataset academico simulado.

Fuente: https://earthquake.usgs.gov/earthquakes/search/
API  : https://earthquake.usgs.gov/fdsnws/event/1/query
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_TIMEOUT = 20

# Caja aproximada del Peru y aguas cercanas.
PERU_BBOX = {
    "minlatitude": -19.0,
    "maxlatitude": 0.5,
    "minlongitude": -82.0,
    "maxlongitude": -68.0,
}


def fetch_usgs_events(
    start: str = "2007-08-15",  # inicio de ShakeMap en USGS
    end: str | None = None,
    min_magnitude: float = 5.0,
    limit: int = 2000,
    timeout: int = USGS_TIMEOUT,
) -> pd.DataFrame:
    """Descarga eventos del Peru en GeoJSON y los devuelve como DataFrame.

    Devuelve DataFrame vacio si la consulta falla (sin lanzar excepcion),
    para no romper la aplicacion.
    """
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    params = {
        "format": "geojson",
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_magnitude,
        "limit": limit,
        "orderby": "time",
        **PERU_BBOX,
    }
    try:
        resp = requests.get(USGS_QUERY_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise ConnectionError(f"No se pudo consultar USGS ComCat: {exc}") from exc

    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {}) or {}
        coords = geom.get("coordinates", [None, None, None])
        if not coords or coords[0] is None:
            continue
        rows.append(
            {
                "usgs_id": feature.get("id"),
                "time_ms": props.get("time"),
                "latitude": coords[1],
                "longitude": coords[0],
                "depth_km": coords[2],
                "mag": props.get("mag"),
                "magType": props.get("magType"),
                "place": props.get("place"),
                "mmi": props.get("mmi"),
                "cdi": props.get("cdi"),
                "felt": props.get("felt"),
                "url": props.get("url"),
            }
        )
    return pd.DataFrame(rows)


def has_intensity_fields(df: pd.DataFrame) -> bool:
    """True si el catalogo trae informacion de intensidad utilizable."""
    if df.empty:
        return False
    for col in ("mmi", "cdi"):
        if col in df.columns and df[col].notna().sum() > 0:
            return True
    return False


def mmi_category(mmi: float) -> str | None:
    """Agrupa una intensidad MMI (I-XII) en Baja / Moderada / Fuerte."""
    if mmi is None or pd.isna(mmi):
        return None
    mmi = float(mmi)
    if mmi < 1 or mmi > 12:
        return None
    if mmi <= 4:
        return "Baja"
    if mmi <= 6:
        return "Moderada"
    return "Fuerte"
