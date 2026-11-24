"""Metricas y utilidades de evaluacion del clasificador KNN."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config


def compute_metrics(y_true, y_pred, labels: list[str] | None = None) -> dict:
    """Metricas principales con promedio macro sobre las 3 clases."""
    labels = labels or config.CLASS_ORDER
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro",
                                  labels=labels, zero_division=0)), 4),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro",
                               labels=labels, zero_division=0)), 4),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro",
                           labels=labels, zero_division=0)), 4),
    }


def confusion_matrix_df(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    """Matriz de confusion con etiquetas legibles (clases en orden canonico)."""
    labels = labels or config.CLASS_ORDER
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"Real {l}" for l in labels],
                        columns=[f"Pred {l}" for l in labels])


def classification_report_df(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    """Classification_report de sklearn como DataFrame."""
    labels = labels or config.CLASS_ORDER
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).transpose()
    rename = {
        "precision": "precision",
        "recall": "recall",
        "f1-score": "f1_score",
        "support": "soporte",
    }
    df = df.rename(columns=rename)
    return df.round(3)


def k_comparison_df(cv_results: dict[int, dict]) -> pd.DataFrame:
    """Construye la tabla comparativa de valores K ordenada por K."""
    rows = []
    for k in sorted(cv_results):
        res = cv_results[k]
        rows.append(
            {
                "K": k,
                "Accuracy": res["accuracy"],
                "Precision macro": res["precision_macro"],
                "Recall macro": res["recall_macro"],
                "F1-score macro": res["f1_macro"],
            }
        )
    return pd.DataFrame(rows)


def select_best_k(cv_results: dict[int, dict]) -> tuple[int, str]:
    """Selecciona el mejor K por F1 macro. Desempate: recall macro, luego K menor.

    Criterio documentado:
        1. Mayor F1-score macro de validacion cruzada.
        2. Si hay empate (a 4 decimales), mayor recall macro
           (menos costoso clasificar por debajo la severidad).
        3. Si persiste el empate, se elige el K mas pequeno
           (modelo mas simple, menor sesgo por vecinos lejanos).
    """
    best_k = None
    best_tuple = None
    for k, res in cv_results.items():
        key = (round(res["f1_macro"], 4), round(res["recall_macro"], 4), -k)
        if best_tuple is None or key > best_tuple:
            best_tuple = key
            best_k = k
    reason = (
        f"Seleccionado K={best_k} por mayor F1-score macro "
        f"({cv_results[best_k]['f1_macro']}), desempate por recall macro y K menor."
    )
    return best_k, reason


def fold_scores_summary(fold_records: list[dict]) -> pd.DataFrame:
    """Resumen estadistico (media y desviacion) de las metricas por fold."""
    df = pd.DataFrame(fold_records)
    return df
