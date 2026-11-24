"""Utilidades compartidas por las páginas: caché, estado y exportaciones."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src import data_loader
from src import prediction
from src import train_model
from src.ui import alert


# ---------------------------------------------------------------------------
# Caché
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    """Dataset limpio (cacheado por sesión de Streamlit)."""
    return data_loader.get_dataset()


@st.cache_data(show_spinner=False)
def cached_summary() -> dict:
    df = cached_dataset()
    summary = data_loader.dataset_summary(df)
    # Fecha serializable.
    last = summary.get("ultima_actualizacion")
    if last is not None and not pd.isna(last):
        summary["ultima_actualizacion"] = str(pd.to_datetime(last).date())
    return summary


@st.cache_resource(show_spinner=False)
def cached_artifact() -> dict:
    """Artefacto del modelo (Pipeline + metadatos), cargado una única vez."""
    return train_model.load_artifact()


def clear_caches() -> None:
    cached_dataset.clear()
    cached_summary.clear()
    cached_artifact.clear()


# ---------------------------------------------------------------------------
# Carga segura
# ---------------------------------------------------------------------------
def load_everything() -> tuple[pd.DataFrame | None, dict | None, dict | None]:
    """Devuelve (dataset, summary, artifact) o muestra error controlado."""
    try:
        df = cached_dataset()
    except data_loader.DatasetError as exc:
        alert("error", str(exc))
        return None, None, None
    try:
        artifact = cached_artifact()
    except train_model.TrainingError as exc:
        summary = cached_summary()
        alert("warn", str(exc))
        return df, summary, None
    summary = cached_summary()
    return df, summary, artifact


def require_model(artifact: dict | None) -> bool:
    if artifact is None:
        alert("warn",
              "El modelo KNN aún no está entrenado. Ejecuta "
              "<code>python -m src.train_model</code> y recarga la página.")
        return False
    return True


# ---------------------------------------------------------------------------
# Historial de simulaciones (sesión)
# ---------------------------------------------------------------------------
HISTORY_KEY = "sim_history"


def add_simulation(result: dict, departamento: str) -> None:
    """Registra una simulación ejecutada con el modelo en la sesión."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    st.session_state[HISTORY_KEY].append(
        {
            "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "departamento": departamento,
            "magnitud": result["entradas"]["magnitud"],
            "profundidad_km": result["entradas"]["profundidad_km"],
            "distancia_km": result["entradas"]["distancia_km"],
            "k": result["k"],
            "categoria_estimada": result["categoria"],
            "probabilidad": result["prob_mayor"],
            "prob_baja": result["probabilidades"].get("Baja", 0),
            "prob_moderada": result["probabilidades"].get("Moderada", 0),
            "prob_fuerte": result["probabilidades"].get("Fuerte", 0),
        }
    )


def get_history() -> pd.DataFrame:
    rows = st.session_state.get(HISTORY_KEY, [])
    if not rows:
        return pd.DataFrame(columns=[
            "fecha_hora", "departamento", "magnitud", "profundidad_km",
            "distancia_km", "k", "categoria_estimada", "probabilidad",
            "prob_baja", "prob_moderada", "prob_fuerte",
        ])
    return pd.DataFrame(rows)


def clear_history() -> None:
    st.session_state[HISTORY_KEY] = []


# ---------------------------------------------------------------------------
# Exportaciones
# ---------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def save_history_csv(df: pd.DataFrame) -> Path | None:
    """Guarda el historial en reports/ y devuelve la ruta."""
    if df.empty:
        return None
    config.ensure_directories()
    df.to_csv(config.CSV_EXPORT_PATH, index=False, encoding="utf-8-sig")
    return config.CSV_EXPORT_PATH


def build_pdf(artifact: dict, summary: dict) -> bytes:
    from src import report_generator
    history = get_history()
    return report_generator.build_report_pdf(
        artifact["metadata"], summary, history if not history.empty else None
    )


def run_prediction(artifact: dict, **kwargs) -> dict | None:
    """Ejecuta una predicción controlando errores de validación."""
    try:
        return prediction.predict(artifact, **kwargs)
    except prediction.PredictionError as exc:
        for msg in str(exc).split(". "):
            if msg.strip():
                alert("error", msg.strip().rstrip(".") + ".")
        return None
    except Exception as exc:  # noqa: BLE001
        alert("error", f"Error inesperado al ejecutar el modelo: {exc}")
        return None
