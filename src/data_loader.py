"""Carga del dataset con validaciones y resumen estadistico.

Usa cache en disco + cache de Streamlit (opcional) para no reprocesar el CSV
en cada interaccion de la interfaz.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from . import preprocessing as pp


class DatasetError(Exception):
    """Error controlado al cargar o validar el dataset."""


def load_dataset(path: str | Path | None = None, clean: bool = True) -> pd.DataFrame:
    """Carga el CSV (por defecto data/sismos_peru.csv) y lo normaliza."""
    path = Path(path) if path else config.DATASET_PATH
    if not path.exists():
        raise DatasetError(
            f"No se encontró el dataset en '{path}'. "
            "Ejecuta `python -m src.train_model` para generarlo y entrenar el modelo."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - se informa al usuario
        raise DatasetError(f"No se pudo leer el dataset: {exc}") from exc

    df = pp.normalize_schema(df)
    if clean:
        df, _ = pp.clean_dataset(df)
    return df


def get_dataset() -> pd.DataFrame:
    """Carga con fallback al dataset por defecto si el principal no existe."""
    try:
        return load_dataset(config.DATASET_PATH)
    except DatasetError:
        if config.DEFAULT_DATASET_PATH.exists():
            return load_dataset(config.DEFAULT_DATASET_PATH)
        df = pp.build_default_dataset()
        return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Calcula indicadores REALES a partir del dataset cargado."""
    if df is None or df.empty:
        return {
            "total_registros": 0, "magnitud_promedio": 0.0,
            "profundidad_promedio": 0.0, "departamentos": 0,
            "eventos": 0, "localidades": 0, "ultima_actualizacion": None,
            "con_coordenadas": 0, "sin_coordenadas": 0,
        }
    last_date = df["fecha"].max() if "fecha" in df.columns else None
    coord_ok = df["latitud"].between(-90, 90) & df["longitud"].between(-180, 180)
    return {
        "total_registros": int(len(df)),
        "magnitud_promedio": round(float(df["magnitud"].mean()), 2),
        "profundidad_promedio": round(float(df["profundidad_km"].mean()), 1),
        "distancia_promedio": round(float(df["distancia_km"].mean()), 1),
        "departamentos": int(df["departamento"].nunique(dropna=True)),
        "eventos": int(df["evento_id"].nunique(dropna=True)),
        "localidades": int(df["localidad"].nunique(dropna=True)),
        "ultima_actualizacion": last_date,
        "con_coordenadas": int(coord_ok.sum()),
        "sin_coordenadas": int((~coord_ok).sum()),
        "tipos_registro": df["tipo_registro"].value_counts().to_dict(),
    }


def column_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de tipos, nulos y valores unicos por columna."""
    return pd.DataFrame(
        {
            "columna": df.columns,
            "tipo": [str(t) for t in df.dtypes],
            "no_nulos": [int(df[c].notna().sum()) for c in df.columns],
            "faltantes": [int(df[c].isna().sum()) for c in df.columns],
            "pct_faltantes": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
            "unicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
