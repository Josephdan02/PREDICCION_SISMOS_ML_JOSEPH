"""Pruebas del entrenamiento: split agrupado, comparación de K y evaluación."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src import data_loader
from src import evaluation as ev
from src import prediction
from src import train_model


@pytest.fixture(scope="module")
def dataset():
    return data_loader.get_dataset()


@pytest.fixture(scope="module")
def artifact():
    return train_model.load_artifact()


# ---------------------------------------------------------------------------
# División agrupada
# ---------------------------------------------------------------------------
def test_split_no_comparte_eventos(dataset):
    train_df, test_df = train_model.make_group_split(dataset)
    grupos_train = set(train_df["evento_id"])
    grupos_test = set(test_df["evento_id"])
    assert grupos_train.isdisjoint(grupos_test), "Hay fuga de eventos entre train y test"


def test_split_respeta_proporcion_80_20(dataset):
    train_df, test_df = train_model.make_group_split(dataset)
    ratio = len(test_df) / len(dataset)
    assert 0.12 <= ratio <= 0.30, f"Proporción de prueba atípica: {ratio:.2f}"


def test_split_es_reproducible(dataset):
    a = train_model.make_group_split(dataset)
    b = train_model.make_group_split(dataset)
    assert list(a[0]["evento_id"]) == list(b[0]["evento_id"])


def test_todas_las_clases_estan_en_train(dataset):
    train_df, _ = train_model.make_group_split(dataset)
    assert set(train_df["categoria_intensidad"].unique()) == set(config.CLASS_ORDER)


# ---------------------------------------------------------------------------
# Pipeline y normalización
# ---------------------------------------------------------------------------
def test_pipeline_contiene_scaler_y_knn():
    pipe = train_model.build_pipeline(5)
    assert "scaler" in pipe.named_steps
    assert "knn" in pipe.named_steps
    assert pipe.named_steps["knn"].n_neighbors == 5


def test_pipeline_normaliza_dentro(dataset):
    train_df, _ = train_model.make_group_split(dataset)
    pipe = train_model.build_pipeline(5)
    pipe.fit(train_df[config.FEATURES], train_df[config.TARGET])
    scaler = pipe.named_steps["scaler"]
    assert scaler.mean_ is not None
    # StandardScaler guarda la media de CADA columna en mean_; la transformación
    # debe centrar los datos de entrenamiento en 0 (media) y 1 (desviación).
    transformed = scaler.transform(train_df[config.FEATURES])
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-8), "Media no centrada en 0"
    assert np.allclose(transformed.std(axis=0), 1, atol=1e-6), "Desviación no igual a 1"
    # Verificación de la propiedad clave: la media del scaler coincide con la del train.
    assert np.allclose(scaler.mean_, train_df[config.FEATURES].to_numpy().mean(axis=0))


def test_comparacion_de_valores_k(dataset):
    train_df, _ = train_model.make_group_split(dataset)
    results, folds, strategy, n_folds = train_model.compare_k_values(train_df)
    assert set(results) == set(config.K_CANDIDATES)
    assert n_folds >= 2
    assert "Group" in strategy
    for k, res in results.items():
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert 0.0 <= res[metric] <= 1.0
    assert len(folds) == len(config.K_CANDIDATES) * n_folds


def test_cv_agrupa_eventos(dataset):
    """Cada split de la CV debe separar eventos entre train y validación."""
    from src.train_model import _cv_splits
    train_df, _ = train_model.make_group_split(dataset)
    y = train_df[config.TARGET]
    groups = train_df[config.GROUP_COLUMN]
    splits, _ = _cv_splits(y, groups)
    for tr, va in splits:
        assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va]))


# ---------------------------------------------------------------------------
# Selección y artefacto
# ---------------------------------------------------------------------------
def test_seleccion_mejor_k_por_f1():
    res = {
        3: {"accuracy": 0.7, "precision_macro": 0.7, "recall_macro": 0.7, "f1_macro": 0.60},
        5: {"accuracy": 0.8, "precision_macro": 0.8, "recall_macro": 0.8, "f1_macro": 0.75},
        7: {"accuracy": 0.8, "precision_macro": 0.8, "recall_macro": 0.8, "f1_macro": 0.70},
        9: {"accuracy": 0.8, "precision_macro": 0.8, "recall_macro": 0.8, "f1_macro": 0.65},
    }
    best, _ = ev.select_best_k(res)
    assert best == 5


def test_desempate_prefiere_k_menor():
    res = {
        3: {"accuracy": 0.8, "precision_macro": 0.8, "recall_macro": 0.8, "f1_macro": 0.75},
        7: {"accuracy": 0.8, "precision_macro": 0.8, "recall_macro": 0.8, "f1_macro": 0.75},
    }
    best, _ = ev.select_best_k(res)
    assert best == 3


def test_artefacto_contiene_todo_lo_necesario(artifact):
    for key in ("pipeline", "metadata", "train_data", "test_data", "projection"):
        assert key in artifact
    meta = artifact["metadata"]
    for key in ("k_seleccionado", "metricas_test", "matriz_confusion",
                "fecha_entrenamiento", "clases"):
        assert key in meta
    assert meta["k_seleccionado"] in config.K_CANDIDATES
    assert meta["clases"] == config.CLASS_ORDER


def test_metricas_test_son_coherentes(artifact):
    m = artifact["metadata"]["metricas_test"]
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0.0 <= m[key] <= 1.0
    assert m["accuracy"] > 0.5, "El modelo debería superar claramente el azar (0.33)"


def test_modelo_supera_el_azar(artifact):
    baseline = 1 / len(config.CLASS_ORDER)
    assert artifact["metadata"]["metricas_test"]["accuracy"] > baseline


def test_matriz_confusion_tiene_tres_clases(artifact):
    cm = artifact["metadata"]["matriz_confusion"]
    assert len(cm) == 3
    total = sum(int(v) for fila in cm.values() for v in fila.values())
    assert total == artifact["metadata"]["total_test"]


def test_evaluacion_sobre_test_no_train(artifact):
    """El test reservado no debe solaparse con el train del artefacto."""
    train_ids = set(artifact["train_data"]["evento_id"])
    test_ids = set(artifact["test_data"]["evento_id"])
    assert train_ids.isdisjoint(test_ids)


def test_load_artifact_error_si_no_existe(tmp_path):
    with pytest.raises(train_model.TrainingError):
        train_model.load_artifact(tmp_path / "no_existe.joblib")


# ---------------------------------------------------------------------------
# Predicción con el modelo real
# ---------------------------------------------------------------------------
def test_prediccion_devuelve_categoria_valida(artifact):
    r = prediction.predict(artifact, magnitud=6.5, profundidad_km=30, distancia_km=60)
    assert r["categoria"] in config.CLASS_ORDER


def test_prediccion_probabilidades_suman_uno(artifact):
    r = prediction.predict(artifact, magnitud=6.5, profundidad_km=30, distancia_km=60)
    assert abs(sum(r["probabilidades"].values()) - 1.0) < 1e-6


def test_prediccion_devuelve_k_vecinos(artifact):
    r = prediction.predict(artifact, magnitud=6.5, profundidad_km=30, distancia_km=60, k=7)
    assert len(r["vecinos"]) == 7
    assert r["vecinos"]["distancia_escalada"].is_monotonic_increasing


def test_votos_suman_k(artifact):
    r = prediction.predict(artifact, magnitud=6.5, profundidad_km=30, distancia_km=60, k=5)
    assert sum(r["votos"].values()) == 5


def test_escenario_costa_fuerte_es_severo(artifact):
    r = prediction.predict(artifact, magnitud=7.8, profundidad_km=22, distancia_km=60)
    assert r["categoria"] in ("Fuerte", "Moderada")


def test_escenario_lejano_y_profundo_es_leve(artifact):
    r = prediction.predict(artifact, magnitud=4.8, profundidad_km=300, distancia_km=350)
    assert r["categoria"] in ("Baja", "Moderada")


def test_validacion_rechaza_valores_invalidos(artifact):
    meta = artifact["metadata"]
    assert prediction.validate_inputs(2.0, 20, 30, 5, meta)
    assert prediction.validate_inputs(6.0, -5, 30, 5, meta)
    assert prediction.validate_inputs(6.0, 20, -30, 5, meta)
    assert prediction.validate_inputs(6.0, 20, 30, 0, meta)


def test_prediccion_lanza_error_con_datos_invalidos(artifact):
    with pytest.raises(prediction.PredictionError):
        prediction.predict(artifact, magnitud=-1, profundidad_km=20, distancia_km=30)
