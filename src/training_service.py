"""Servicio de entrenamiento invocable desde la interfaz.

Envoltorio fino sobre ``src.train_model`` que garantiza la existencia del
dataset y devuelve un resumen legible del resultado.
"""

from __future__ import annotations

from . import config
from . import data_loader
from . import preprocessing as pp
from . import train_model


def retrain(force_dataset: bool = False) -> dict:
    """Regenera (opcionalmente) el dataset y reentrena el modelo."""
    config.ensure_directories()

    if force_dataset or not config.DATASET_PATH.exists():
        df = pp.build_default_dataset(force=force_dataset)
        df.to_csv(config.DATASET_PATH, index=False)
    else:
        df = data_loader.get_dataset()

    artifact = train_model.train(df, persist=True)
    meta = artifact["metadata"]
    return {
        "k": meta["k_seleccionado"],
        "accuracy": meta["metricas_test"]["accuracy"],
        "f1_macro": meta["metricas_test"]["f1_macro"],
        "registros": len(df),
        "fecha": meta["fecha_entrenamiento"],
    }
