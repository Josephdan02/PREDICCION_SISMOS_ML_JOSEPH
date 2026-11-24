"""Entrenamiento del Pipeline KNN de SISMO AI PERU.

Flujo:
    1. Cargar/limpiar dataset.
    2. Division 80/20 AGRUPADA por evento_id (GroupShuffleSplit).
    3. Comparar K = 3, 5, 7, 9 con validacion cruzada agrupada
       (StratifiedGroupKFold sobre el conjunto de entrenamiento).
    4. Seleccionar mejor K por F1-score macro.
    5. Entrenar el Pipeline final y evaluarlo UNA vez sobre el test reservado.
    6. Persistir Pipeline + metadatos con Joblib/JSON.

Se guarda ademas el diagrama de dispersion 2D proyectado del conjunto de
entrenamiento (mag, prof, dist -> 2 componentes) para poder VISUALIZAR los
vecinos seleccionados por el modelo en la interfaz.

Uso:
    python -m src.train_model
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from . import data_loader
from . import evaluation as ev
from . import preprocessing as pp


class TrainingError(Exception):
    """Error controlado durante el entrenamiento."""


# ---------------------------------------------------------------------------
# Split agrupado
# ---------------------------------------------------------------------------
def make_group_split(df: pd.DataFrame, test_size: float | None = None,
                     random_state: int | None = None):
    """Division 80/20 agrupada por evento_id (ningun evento cae en ambos lados)."""
    test_size = test_size if test_size is not None else config.TEST_SIZE
    random_state = random_state if random_state is not None else config.RANDOM_STATE

    groups = df[config.GROUP_COLUMN]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df[config.TARGET], groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df[config.GROUP_COLUMN]) & set(test_df[config.GROUP_COLUMN])
    if overlap:
        raise TrainingError("Fuga de grupos: hay eventos en train y test simultaneamente.")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Pipeline y validacion cruzada
# ---------------------------------------------------------------------------
def build_pipeline(k: int) -> Pipeline:
    """StandardScaler + KNeighborsClassifier dentro del Pipeline (sin fuga)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="uniform",
                                         metric="euclidean")),
        ]
    )


def _cv_splits(y: pd.Series, groups: pd.Series, n_splits: int = config.CV_SPLITS):
    """Genera folds agrupados y estratificados si es viable."""
    df_tmp = pd.DataFrame({"y": y.values, "g": groups.values})
    min_class = df_tmp.groupby("y")["g"].nunique().min()
    n_groups = df_tmp["g"].nunique()
    splits = min(n_splits, int(min_class), n_groups)
    strategy = "StratifiedGroupKFold"
    if splits < 2:
        # Fallback: pliegues agrupados sin estratificar.
        from sklearn.model_selection import GroupKFold
        strategy = "GroupKFold (fallback: clases/grupos insuficientes)"
        return list(GroupKFold(n_splits=max(2, min(n_splits, n_groups))).split(
            df_tmp, df_tmp["y"], df_tmp["g"])), strategy
    sgkf = StratifiedGroupKFold(n_splits=splits, shuffle=True,
                                random_state=config.RANDOM_STATE)
    return list(sgkf.split(df_tmp, df_tmp["y"], df_tmp["g"])), strategy


def compare_k_values(train_df: pd.DataFrame, k_values: list[int] | None = None):
    """Evalua cada K con validacion cruzada agrupada dentro de train."""
    k_values = k_values or config.K_CANDIDATES
    X = train_df[config.FEATURES]
    y = train_df[config.TARGET]
    groups = train_df[config.GROUP_COLUMN]

    splits, strategy = _cv_splits(y, groups)
    results: dict[int, dict] = {}
    fold_records: list[dict] = []

    for k in k_values:
        fold_metrics = []
        for fold_idx, (tr, va) in enumerate(splits, start=1):
            pipe = build_pipeline(k)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[va])
            m = ev.compute_metrics(y.iloc[va], pred)
            fold_metrics.append(m)
            fold_records.append({"k": k, "fold": fold_idx, **m})

        agg = {
            metric: round(float(np.mean([fm[metric] for fm in fold_metrics])), 4)
            for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        }
        agg["std_f1_macro"] = round(
            float(np.std([fm["f1_macro"] for fm in fold_metrics])), 4)
        agg["n_folds"] = len(splits)
        results[k] = agg

    return results, fold_records, strategy, len(splits)


def build_projection(pipe: Pipeline, train_df: pd.DataFrame) -> dict:
    """Proyeccion PCA 2D (ajustada en el espacio normalizado de train).

    Se usa unicamente para VISUALIZACION; la clasificacion real usa las 3
    variables originales normalizadas.
    """
    scaler: StandardScaler = pipe.named_steps["scaler"]
    X_scaled = scaler.transform(train_df[config.FEATURES])
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    return {
        "coords": coords,
        "pca": pca,
        "scaler": scaler,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }


# ---------------------------------------------------------------------------
# Entrenamiento completo
# ---------------------------------------------------------------------------
def train(df: pd.DataFrame | None = None, persist: bool = True) -> dict:
    """Entrena, evalua y (opcionalmente) persiste modelo + metadatos."""
    if df is None:
        df = data_loader.get_dataset()

    issues = pp.validate_dataset(df)
    if issues:
        raise TrainingError(" ".join(issues))

    train_df, test_df = make_group_split(df)

    cv_results, fold_records, cv_strategy, n_folds = compare_k_values(train_df)
    best_k, best_reason = ev.select_best_k(cv_results)

    # Modelo final con el mejor K, entrenado sobre TODO el conjunto de train.
    final_pipe = build_pipeline(best_k)
    final_pipe.fit(train_df[config.FEATURES], train_df[config.TARGET])

    # Evaluacion UNICA sobre el test reservado.
    y_test = test_df[config.TARGET]
    y_pred = final_pipe.predict(test_df[config.FEATURES])
    test_metrics = ev.compute_metrics(y_test, y_pred)
    cm = ev.confusion_matrix_df(y_test, y_pred)
    report = ev.classification_report_df(y_test, y_pred)
    # La matriz se serializa indexada por fila real ("Real X" -> {"Pred Y": n}),
    # que es como la consumen la interfaz y el generador de PDF.
    cm_dict = {
        f"Real {real}": {
            f"Pred {pred}": int(cm.loc[f"Real {real}", f"Pred {pred}"])
            for pred in config.CLASS_ORDER
        }
        for real in config.CLASS_ORDER
    }
    # El informe se serializa por categoria (incluye macro avg y weighted avg).
    report_dict = {
        str(idx): {
            "precision": round(float(row["precision"]), 4),
            "recall": round(float(row["recall"]), 4),
            "f1_score": round(float(row["f1_score"]), 4),
            "soporte": int(float(row["soporte"])),
        }
        for idx, row in report.iterrows()
    }

    projection = build_projection(final_pipe, train_df)

    metadata = {
        "modelo": config.MODEL_NAME,
        "version_proyecto": "1.0.0",
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "features": config.FEATURES,
        "target": config.TARGET,
        "clases": config.CLASS_ORDER,
        "k_seleccionado": int(best_k),
        "criterio_seleccion": best_reason,
        "k_comparados": config.K_CANDIDATES,
        "pipeline": ["StandardScaler", f"KNeighborsClassifier(n_neighbors={best_k})"],
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "split_estrategia": "GroupShuffleSplit agrupado por evento_id",
        "cv_estrategia": cv_strategy,
        "cv_folds": int(n_folds),
        "cv_resultados": {str(k): v for k, v in cv_results.items()},
        "cv_folds_detalle": fold_records,
        "metricas_test": test_metrics,
        "f1_macro_test": test_metrics["f1_macro"],
        "matriz_confusion": cm_dict,
        "informe_clasificacion": report_dict,
        "aciertos_test": int(np.sum(np.asarray(y_test) == np.asarray(y_pred))),
        "total_test": int(len(y_test)),
        "proyeccion_var_explicada": [round(v, 4) for v in projection["explained_variance"]],
        "dataset": {
            "total_registros": int(len(df)),
            "eventos": int(df[config.GROUP_COLUMN].nunique()),
            "departamentos": int(df["departamento"].nunique()),
            "tipos_registro": df["tipo_registro"].value_counts().to_dict()
                           if "tipo_registro" in df else {},
            "train_registros": int(len(train_df)),
            "train_eventos": int(train_df[config.GROUP_COLUMN].nunique()),
            "test_registros": int(len(test_df)),
            "test_eventos": int(test_df[config.GROUP_COLUMN].nunique()),
            "distribucion_clases": pp.class_distribution(df).to_dict(orient="records"),
            "rango_magnitud": [float(df["magnitud"].min()), float(df["magnitud"].max())],
            "rango_profundidad": [float(df["profundidad_km"].min()),
                                  float(df["profundidad_km"].max())],
            "rango_distancia": [float(df["distancia_km"].min()),
                                float(df["distancia_km"].max())],
        },
    }

    artifact = {
        "pipeline": final_pipe,
        "metadata": metadata,
        "projection": projection,
        "train_data": train_df,
        "test_data": test_df,
    }

    if persist:
        persist_artifacts(artifact)
        config.ensure_directories()
        test_df.to_csv(config.PROCESSED_DIR / "test_set.csv", index=False)

    return artifact


def persist_artifacts(artifact: dict) -> None:
    """Guarda Pipeline y metadatos en disco."""
    config.ensure_directories()
    joblib.dump(artifact, config.MODEL_PATH)
    with open(config.METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(artifact["metadata"], fh, indent=2, ensure_ascii=False)


def load_artifact(path=None) -> dict:
    """Carga el artefacto entrenado. Lanza error comprensible si no existe."""
    path = path or config.MODEL_PATH
    if not path.exists():
        raise TrainingError(
            "Modelo no entrenado. Ejecuta `python -m src.train_model` para generarlo."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise TrainingError("El archivo del modelo es invalido o esta corrupto.")
    return artifact


def main() -> None:
    config.ensure_directories()
    print("[1/4] Generando/cargando dataset academico simulado...")
    df = pp.build_default_dataset(force=True)
    df.to_csv(config.DATASET_PATH, index=False)
    print(f"     Registros: {len(df)} | Eventos: {df['evento_id'].nunique()}")

    print("[2/4] Entrenando y comparando K = 3, 5, 7, 9...")
    artifact = train(df)

    meta = artifact["metadata"]
    print(f"     K seleccionado: {meta['k_seleccionado']}")
    print(f"     CV estrategia: {meta['cv_estrategia']} ({meta['cv_folds']} folds)")
    for k, res in meta["cv_resultados"].items():
        print(f"       K={k}: F1 macro={res['f1_macro']:.4f}  acc={res['accuracy']:.4f}")
    print("[3/4] Evaluacion sobre test independiente:")
    for name, value in meta["metricas_test"].items():
        print(f"       {name}: {value}")
    print("[4/4] Modelo persistido en:", config.MODEL_PATH)


if __name__ == "__main__":
    main()
