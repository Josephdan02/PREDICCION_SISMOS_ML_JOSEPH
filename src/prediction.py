"""Motor de prediccion: categorias, probabilidades y vecinos mas cercanos.

Toda la clasificacion proviene EXCLUSIVAMENTE del Pipeline KNN entrenado.
No existe ninguna regla if/else que decida la categoria sismica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import train_model


class PredictionError(Exception):
    """Error controlado de validacion o de modelo no disponible."""


# ---------------------------------------------------------------------------
# Validacion
# ---------------------------------------------------------------------------
def validate_inputs(magnitud: float, profundidad_km: float,
                    distancia_km: float, k: int,
                    meta: dict | None = None) -> list[str]:
    """Valida los datos del simulador contra los rangos admitidos.

    Devuelve una lista de mensajes de error (vacia si todo es valido).
    """
    errors: list[str] = []
    try:
        magnitud = float(magnitud)
        profundidad_km = float(profundidad_km)
        distancia_km = float(distancia_km)
        k = int(k)
    except (TypeError, ValueError):
        return ["Los valores ingresados deben ser numéricos."]

    if not np.isfinite(magnitud) or not np.isfinite(profundidad_km) or not np.isfinite(distancia_km):
        return ["Los valores no pueden ser NaN ni infinitos."]

    if not config.MAGNITUDE_RANGE[0] <= magnitud <= config.MAGNITUDE_RANGE[1]:
        errors.append(
            f"La magnitud debe estar entre {config.MAGNITUDE_RANGE[0]} y "
            f"{config.MAGNITUDE_RANGE[1]}."
        )
    if profundidad_km < 0:
        errors.append("La profundidad no puede ser negativa.")
    elif profundidad_km > config.DEPTH_RANGE[1]:
        errors.append(f"La profundidad no puede superar {config.DEPTH_RANGE[1]} km.")
    if distancia_km < 0:
        errors.append("La distancia epicentral no puede ser negativa.")
    elif distancia_km > config.DISTANCE_RANGE[1]:
        errors.append(f"La distancia epicentral no puede superar {config.DISTANCE_RANGE[1]} km.")
    if k < 1:
        errors.append("El número de vecinos K debe ser al menos 1.")

    # Comprobacion de rango respecto al entrenamiento (advertencia estricta).
    if meta:
        rng_mag = meta.get("dataset", {}).get("rango_magnitud", config.MAGNITUDE_RANGE)
        rng_prof = meta.get("dataset", {}).get("rango_profundidad", config.DEPTH_RANGE)
        rng_dist = meta.get("dataset", {}).get("rango_distancia", config.DISTANCE_RANGE)
        if not (rng_mag[0] <= magnitud <= rng_mag[1]):
            errors.append(
                f"Magnitud fuera del rango observado durante el entrenamiento "
                f"({rng_mag[0]:.1f} – {rng_mag[1]:.1f})."
            )
        if not (rng_prof[0] <= profundidad_km <= rng_prof[1]):
            errors.append(
                f"Profundidad fuera del rango del entrenamiento "
                f"({rng_prof[0]:.1f} – {rng_prof[1]:.1f} km)."
            )
        if not (rng_dist[0] <= distancia_km <= rng_dist[1]):
            errors.append(
                f"Distancia fuera del rango del entrenamiento "
                f"({rng_dist[0]:.1f} – {rng_dist[1]:.1f} km)."
            )
        n_train = meta.get("dataset", {}).get("train_registros", 0)
        if k > max(n_train, 1):
            errors.append("K no puede superar la cantidad de registros de entrenamiento.")
    return errors


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------
def predict(artifact: dict, magnitud: float, profundidad_km: float,
            distancia_km: float, k: int | None = None) -> dict:
    """Predice categoria, probabilidades, votos y vecinos con el Pipeline.

    Si ``k`` difiere del K del modelo persistido, se crea (sobre la marcha) un
    clasificador con ese K usando el StandardScaler YA AJUSTADO del Pipeline.
    Esto permite explorar K en el simulador sin reentrenar el escalador y sin
    fuga de informacion.
    """
    from sklearn.neighbors import KNeighborsClassifier

    meta = artifact.get("metadata", {})
    k = int(k if k is not None else meta.get("k_seleccionado", config.DEFAULT_K))

    errors = validate_inputs(magnitud, profundidad_km, distancia_km, k, meta)
    if errors:
        raise PredictionError(" ".join(errors))

    pipeline = artifact["pipeline"]
    train_df: pd.DataFrame = artifact["train_data"]

    X_input = pd.DataFrame(
        [[float(magnitud), float(profundidad_km), float(distancia_km)]],
        columns=config.FEATURES,
    )

    model_k = meta.get("k_seleccionado", config.DEFAULT_K)
    if k == model_k:
        active_pipe = pipeline
        scaler = pipeline.named_steps["scaler"]
        knn = pipeline.named_steps["knn"]
    else:
        scaler = pipeline.named_steps["scaler"]
        knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
        knn.fit(scaler.transform(train_df[config.FEATURES]), train_df[config.TARGET])
        from sklearn.pipeline import Pipeline
        active_pipe = Pipeline([("scaler", scaler), ("knn", knn)])

    proba = active_pipe.predict_proba(X_input)[0]
    classes = list(active_pipe.classes_)
    pred_idx = int(np.argmax(proba))
    category = str(classes[pred_idx])

    distances, neighbor_idx = knn.kneighbors(
        scaler.transform(X_input), n_neighbors=int(k)
    )
    distances = distances[0]
    neighbor_idx = neighbor_idx[0]

    neighbors = train_df.iloc[neighbor_idx][
        ["evento_id", "fecha", "localidad", "departamento", "magnitud",
         "profundidad_km", "distancia_km", "intensidad_observada",
         "categoria_intensidad"]
    ].copy()
    neighbors.insert(0, "vecino", range(1, len(neighbors) + 1))
    neighbors["distancia_escalada"] = np.round(distances, 4)

    votes = {c: 0 for c in config.CLASS_ORDER}
    for cat in neighbors["categoria_intensidad"]:
        votes[str(cat)] = votes.get(str(cat), 0) + 1

    # Coordenadas de la consulta en la proyeccion PCA de visualizacion.
    projection = artifact.get("projection", {})
    query_2d = None
    neighbor_2d = None
    if projection:
        pca = projection["pca"]
        q_scaled = scaler.transform(X_input)
        query_2d = pca.transform(q_scaled)[0]
        neighbor_2d = projection["coords"][neighbor_idx]

    probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}

    return {
        "categoria": category,
        "probabilidades": probabilities,
        "prob_mayor": round(float(proba[pred_idx]), 4),
        "votos": votes,
        "vecinos": neighbors,
        "k": k,
        "k_modelo": model_k,
        "entradas": {
            "magnitud": float(magnitud),
            "profundidad_km": float(profundidad_km),
            "distancia_km": float(distancia_km),
        },
        "query_2d": query_2d,
        "neighbor_2d": neighbor_2d,
        "distancia_normalizada": float(distances.mean()),
    }


def load_ready_artifact() -> dict:
    """Carga el artefacto o genera error comprensible."""
    return train_model.load_artifact()
