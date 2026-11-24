"""Pruebas del dataset simulado y del preprocesamiento."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src import data_loader
from src import preprocessing as pp


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return data_loader.get_dataset()


# ---------------------------------------------------------------------------
# Generación del dataset
# ---------------------------------------------------------------------------
def test_dataset_tiene_al_menos_500_registros(dataset):
    assert len(dataset) >= 500, "El reto exige al menos 500 registros evento-localidad."


def test_columnas_obligatorias_presentes(dataset):
    for col in config.REQUIRED_COLUMNS:
        assert col in dataset.columns, f"Falta la columna {col}"


def test_tipos_de_datos_correctos(dataset):
    assert pd.api.types.is_numeric_dtype(dataset["magnitud"])
    assert pd.api.types.is_numeric_dtype(dataset["profundidad_km"])
    assert pd.api.types.is_numeric_dtype(dataset["distancia_km"])
    assert pd.api.types.is_numeric_dtype(dataset["intensidad_observada"])


def test_sin_nulos_en_columnas_criticas(dataset):
    criticas = ["evento_id", "magnitud", "profundidad_km", "distancia_km",
                "intensidad_observada", "categoria_intensidad"]
    assert dataset[criticas].isna().sum().sum() == 0


def test_sin_duplicados_exactos(dataset):
    assert dataset.duplicated().sum() == 0


def test_rangos_fisicos_validos(dataset):
    assert dataset["magnitud"].between(*config.MAGNITUDE_RANGE).all()
    assert dataset["profundidad_km"].between(*config.DEPTH_RANGE).all()
    assert dataset["distancia_km"].between(*config.DISTANCE_RANGE).all()
    assert dataset["intensidad_observada"].between(1, 12).all()


def test_categorias_son_las_tres_esperadas(dataset):
    assert set(dataset["categoria_intensidad"].unique()) == set(config.CLASS_ORDER)


def test_intensidad_consistente_con_categoria(dataset):
    for intensidad, categoria in zip(dataset["intensidad_observada"],
                                     dataset["categoria_intensidad"]):
        assert pp.mmi_to_category(intensidad) == categoria


def test_todas_las_clases_tienen_suficientes_registros(dataset):
    counts = dataset["categoria_intensidad"].value_counts()
    for clase in config.CLASS_ORDER:
        assert counts.get(clase, 0) >= 30, f"Clase {clase} con pocos registros"


def test_registros_simulados_identificados(dataset):
    tipos = dataset["tipo_registro"].unique()
    assert "SIMULADO" in tipos
    if set(tipos) == {"SIMULADO"}:
        assert dataset["fuente"].str.contains("sint|simulad", case=False).all()


def test_eventos_agrupan_multiples_localidades(dataset):
    por_evento = dataset.groupby("evento_id").size()
    assert por_evento.max() > 1, "La agrupación por evento requiere varios registros"
    assert dataset["evento_id"].nunique() >= 100


# ---------------------------------------------------------------------------
# Generador reproducible y con solapamiento de clases
# ---------------------------------------------------------------------------
def test_generador_es_reproducible():
    a = pp.generate_synthetic_dataset(120, random_state=7)
    b = pp.generate_synthetic_dataset(120, random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_generador_produce_variabilidad_de_etiquetas():
    df = pp.generate_synthetic_dataset(600, random_state=config.RANDOM_STATE)
    counts = df["categoria_intensidad"].value_counts()
    assert len(counts) == 3
    assert counts.min() >= 40


def test_etiqueta_no_es_determinista():
    """Con las mismas entradas debe haber categorías distintas (ruido)."""
    df = pp.generate_synthetic_dataset(800, random_state=config.RANDOM_STATE)
    df["mag_bin"] = df["magnitud"].round(0)
    grupos = df.groupby("mag_bin")["categoria_intensidad"].nunique()
    assert (grupos > 1).any(), "El etiquetado parece una regla determinista"


# ---------------------------------------------------------------------------
# Normalización y limpieza
# ---------------------------------------------------------------------------
def test_normalize_schema_rellena_columnas_faltantes():
    raw = pd.DataFrame({"magnitud": [5.0], "profundidad_km": [30], "distancia_km": [50]})
    out = pp.normalize_schema(raw)
    assert list(out.columns) == config.REQUIRED_COLUMNS


def test_clean_dataset_elimina_nulos_duplicados_y_rangos():
    raw = pd.DataFrame({
        "evento_id": ["E1", "E1", "E2", "E3"],
        "magnitud": [5.0, 5.0, -1.0, 6.0],
        "profundidad_km": [30, 30, 20, 25],
        "distancia_km": [40, 40, 50, 60],
        "intensidad_observada": [4, 4, 3, 7],
        "categoria_intensidad": ["Baja", "Baja", "Baja", "Fuerte"],
        "fecha": ["2020-01-01"] * 4,
        "latitud": [-12] * 4, "longitud": [-77] * 4,
        "localidad": ["Lima"] * 4, "departamento": ["Lima"] * 4,
        "escala_magnitud": ["Mw"] * 4, "tipo_registro": ["SIMULADO"] * 4,
        "fuente": ["test"] * 4,
    })
    clean, report = pp.clean_dataset(raw)
    assert report["eliminados_duplicados"] == 1
    assert report["eliminados_fuera_rango"] == 1
    assert len(clean) == 2


def test_validate_dataset_detecta_vacio():
    issues = pp.validate_dataset(pd.DataFrame(columns=config.REQUIRED_COLUMNS))
    assert issues


def test_categoria_de_rangos_correctos():
    assert pp.category_for_value(1) == "Baja"
    assert pp.category_for_value(4) == "Baja"
    assert pp.category_for_value(5) == "Moderada"
    assert pp.category_for_value(6) == "Moderada"
    assert pp.category_for_value(7) == "Fuerte"
    assert pp.category_for_value(12) == "Fuerte"
