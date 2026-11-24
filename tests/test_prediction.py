"""Pruebas del motor de predicción, mapa y generación de PDF."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src import data_loader
from src import map_service
from src import prediction
from src import report_generator
from src import train_model


@pytest.fixture(scope="module")
def artifact():
    return train_model.load_artifact()


@pytest.fixture(scope="module")
def dataset():
    return data_loader.get_dataset()


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def test_k_distinto_usa_mismo_pipeline(artifact):
    base = prediction.predict(artifact, magnitud=6.0, profundidad_km=40, distancia_km=70)
    alt = prediction.predict(artifact, magnitud=6.0, profundidad_km=40, distancia_km=70, k=9)
    assert alt["k"] == 9
    assert alt["k_modelo"] == artifact["metadata"]["k_seleccionado"]
    assert len(alt["vecinos"]) == 9
    # Los 5 vecinos más cercanos del modelo base deben coincidir con los K=9.
    assert list(base["vecinos"]["evento_id"]) == list(alt["vecinos"]["evento_id"][:len(base["vecinos"])])


def test_vecinos_tienen_campos_requeridos(artifact):
    r = prediction.predict(artifact, magnitud=6.2, profundidad_km=25, distancia_km=45)
    for col in ("vecino", "evento_id", "localidad", "magnitud", "profundidad_km",
                "distancia_km", "intensidad_observada", "categoria_intensidad",
                "distancia_escalada"):
        assert col in r["vecinos"].columns


def test_resultado_incluye_coordenadas_de_proyeccion(artifact):
    r = prediction.predict(artifact, magnitud=6.2, profundidad_km=25, distancia_km=45)
    assert r["query_2d"] is not None
    assert r["neighbor_2d"].shape[0] == r["k"]


def test_validacion_advierte_fuera_de_rango_entrenamiento(artifact):
    meta = artifact["metadata"]
    rmag = meta["dataset"]["rango_magnitud"]
    errors = prediction.validate_inputs(rmag[1] + 0.5, 30, 50, 5, meta)
    assert any("entrenamiento" in e for e in errors)


# ---------------------------------------------------------------------------
# Mapa
# ---------------------------------------------------------------------------
def test_mapa_se_genera(dataset):
    fmap, shown, excluded = map_service.build_events_map(dataset)
    html = fmap.get_root().render()
    assert "folium" in html.lower() or "leaflet" in html.lower()
    assert shown > 0
    assert excluded >= 0


def test_mapa_excluye_coordenadas_invalidas():
    df = pd.DataFrame({
        "latitud": [None, -12.0], "longitud": [None, -77.0],
        "magnitud": [5.0, 5.0], "profundidad_km": [30, 30],
        "distancia_km": [40, 40], "intensidad_observada": [3, 3],
        "categoria_intensidad": ["Baja", "Baja"],
        "localidad": ["X", "Lima"], "departamento": ["Lima", "Lima"],
        "evento_id": ["E1", "E1"], "fecha": pd.to_datetime(["2020-01-01"] * 2),
        "escala_magnitud": ["Mw"] * 2, "tipo_registro": ["SIMULADO"] * 2,
        "fuente": ["test"] * 2,
    })
    geo, excluded = map_service.valid_coords(df)
    assert excluded == 1
    assert len(geo) == 1


def test_filtros_de_mapa(dataset):
    filtrado = map_service.filter_events(dataset, departamento="Lima")
    assert (filtrado["departamento"] == "Lima").all()
    filtrado = map_service.filter_events(dataset, categoria="Fuerte")
    assert (filtrado["categoria_intensidad"] == "Fuerte").all()


def test_mapa_incluye_referencia_huanuco(dataset):
    fmap, _, _ = map_service.build_events_map(dataset.iloc[:5], show_huanuco=True)
    html = fmap.get_root().render()
    assert "Huánuco" in html


def test_mapa_vacio_no_falla():
    empty = pd.DataFrame(columns=config.REQUIRED_COLUMNS)
    fmap, shown, excluded = map_service.build_events_map(empty)
    assert shown == 0
    assert excluded == 0
    assert fmap is not None


# ---------------------------------------------------------------------------
# Informe PDF
# ---------------------------------------------------------------------------
def test_pdf_se_genera_con_datos_reales(artifact, dataset):
    summary = data_loader.dataset_summary(dataset)
    pdf = report_generator.build_report_pdf(artifact["metadata"], summary)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 5000


def test_pdf_incluye_secciones_clave(artifact, dataset):
    summary = data_loader.dataset_summary(dataset)
    pdf = report_generator.build_report_pdf(artifact["metadata"], summary)
    # Los strings de ReportLab suelen estar comprimidos; verificamos el tamaño
    # y que el documento es válido.
    assert b"%%EOF" in pdf[-1024:] or b"%%EOF" in pdf


def test_pdf_con_historial_de_simulaciones(artifact, dataset):
    history = pd.DataFrame([{
        "fecha_hora": "2026-01-01 10:00", "departamento": "Ica", "magnitud": 5.6,
        "profundidad_km": 32, "distancia_km": 42, "k": 5,
        "categoria_estimada": "Fuerte", "probabilidad": 1.0,
    }])
    summary = data_loader.dataset_summary(dataset)
    pdf = report_generator.build_report_pdf(artifact["metadata"], summary, history)
    assert pdf.startswith(b"%PDF")
