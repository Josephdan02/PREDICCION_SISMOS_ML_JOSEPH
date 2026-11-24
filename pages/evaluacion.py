"""Página: Evaluación estadística sobre el conjunto de prueba independiente."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pages._shared import load_everything, require_model
from src import charts, config, ui


def render() -> None:
    ui.institutional_header()
    ui.page_title(
        "Evaluación del modelo",
        "Métricas calculadas sobre el conjunto de prueba reservado desde el inicio "
        "(no sobre los datos de entrenamiento).",
        eyebrow="MÉTRICAS Y VALIDACIÓN",
    )

    df, summary, artifact = load_everything()
    if not require_model(artifact):
        return

    meta = artifact["metadata"]
    metrics = meta.get("metricas_test", {})

    ui.alert(
        "info",
        f"Evaluación única del pipeline <b>K = {meta.get('k_seleccionado')}</b> sobre "
        f"<b>{meta.get('total_test', 0)}</b> registros de prueba "
        f"({meta.get('dataset', {}).get('test_eventos', 0)} eventos). "
        "El conjunto de prueba se reservó antes de comparar K y nunca se usó para entrenar.",
    )

    # ---------------- Indicadores principales ----------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%",
              f"{meta.get('aciertos_test', 0)}/{meta.get('total_test', 0)}")
    c2.metric("Precision macro", f"{metrics.get('precision_macro', 0) * 100:.1f}%")
    c3.metric("Recall macro", f"{metrics.get('recall_macro', 0) * 100:.1f}%")
    c4.metric("F1-score macro", f"{metrics.get('f1_macro', 0) * 100:.1f}%")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(charts.gauge(metrics.get("accuracy", 0), "Accuracy",
                                     color="#4f8cff"), width='stretch',
                        config={"displayModeBar": False})
    with g2:
        st.plotly_chart(charts.gauge(metrics.get("f1_macro", 0), "F1-score macro",
                                     color="#a78bfa"), width='stretch',
                        config={"displayModeBar": False})
    with g3:
        st.plotly_chart(charts.gauge(metrics.get("recall_macro", 0), "Recall macro",
                                     color="#34d399"), width='stretch',
                        config={"displayModeBar": False})

    st.divider()

    # ---------------- Matriz de confusión ----------------
    ui.section_title("Matriz de confusión", "Valores reales (filas) frente a valores predichos (columnas).",
                     "fa-table-cells")
    cm = meta.get("matriz_confusion", {})
    if not cm:
        ui.alert("warn", "No hay matriz de confusión almacenada. Reentrena el modelo.")
        return

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.plotly_chart(charts.confusion_heatmap(cm), width='stretch',
                        config={"displayModeBar": False})
    with right:
        matrix = pd.DataFrame(
            [[int(cm.get(f"Real {r}", {}).get(f"Pred {c}", 0)) for c in config.CLASS_ORDER]
             for r in config.CLASS_ORDER],
            index=[f"Real {c}" for c in config.CLASS_ORDER],
            columns=[f"Pred {c}" for c in config.CLASS_ORDER],
        )
        st.dataframe(matrix, width='stretch')
        total = matrix.values.sum()
        correct = sum(matrix.iloc[i, i] for i in range(len(config.CLASS_ORDER)))
        st.caption(
            f"Diagonal: {correct} de {total} casos correctos. Las filas suman el soporte real "
            "de cada categoría; las columnas, cuántas veces el modelo predijo esa categoría."
        )

    # ---------------- Informe por categoría ----------------
    ui.section_title("Informe de clasificación por categoría", None, "fa-clipboard-list")
    rep = meta.get("informe_clasificacion", {})
    if rep:
        def _row(key: str, label: str) -> dict:
            r = rep.get(key, {}) or {}
            return {
                "Categoría": label,
                "Rango MMI": config.INTENSITY_RANGE.get(key, ""),
                "Precision": round(float(r.get("precision", 0)), 3),
                "Recall": round(float(r.get("recall", 0)), 3),
                "F1-score": round(float(r.get("f1_score", r.get("f1-score", 0))), 3),
                "Soporte": int(float(r.get("soporte", r.get("support", 0)) or 0)),
            }

        rows = [_row(cat, cat) for cat in config.CLASS_ORDER]
        if rep.get("macro avg"):
            macro_row = _row("macro avg", "Promedio macro")
            macro_row["Rango MMI"] = ""
            rows.append(macro_row)
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        # Recuperar las métricas por clase para la interpretación.
        rep = {
            cat: {
                "precision": rows[i]["Precision"],
                "recall": rows[i]["Recall"],
                "f1_score": rows[i]["F1-score"],
            }
            for i, cat in enumerate(config.CLASS_ORDER)
        }

    # ---------------- Interpretación ----------------
    ui.section_title("Interpretación de resultados", None, "fa-lightbulb")
    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_macro", 0)
    baseline = 1 / len(config.CLASS_ORDER)

    per_class_lines = []
    for cat in config.CLASS_ORDER:
        r = rep.get(cat, {}) if rep else {}
        per_class_lines.append(
            f"<b>{cat}</b>: precision {float(r.get('precision', 0))*100:.0f}%, "
            f"recall {float(r.get('recall', 0))*100:.0f}%"
        )

    st.markdown(
        f"""
<div class="glass">
  <p style="font-size:12.8px;color:#c7d2ef;line-height:1.7;margin:0">
    La exactitud global es <b>{acc*100:.1f}%</b>, frente a un {baseline*100:.1f}% esperado por azar
    en un problema de tres clases. El F1-score macro ({f1*100:.1f}%) es la métrica de referencia
    porque pondera por igual a las tres categorías, incluso si alguna tiene menos registros.
    {' · '.join(per_class_lines)}.
  </p>
  <p style="font-size:12.2px;color:#97a3c2;line-height:1.7;margin:10px 0 0">
    Las confusiones más frecuentes ocurren entre categorías contiguas (Baja↔Moderada y
    Moderada↔Fuerte), lo que es esperable: la intensidad es un continuo y los puntos de corte
    I–IV / V–VI / VII están artificialmente definidos. La clase <b>Fuerte</b>, con menos ejemplos,
    suele presentar menor recall. Una matriz agrupada por evento evita inflar la métrica con
    localidades del mismo sismo en ambos lados.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Detalle de aciertos del test ----------------
    ui.section_title("Resultados sobre el conjunto de prueba", None, "fa-magnifying-glass-chart")
    test_df = artifact.get("test_data")
    if test_df is not None and not test_df.empty:
        pipe = artifact["pipeline"]
        preds = pipe.predict(test_df[config.FEATURES])
        detail = test_df[["evento_id", "localidad", "departamento", "magnitud",
                          "profundidad_km", "distancia_km", "categoria_intensidad"]].copy()
        detail["prediccion"] = preds
        detail["acierto"] = detail["categoria_intensidad"] == detail["prediccion"]
        filtro = st.radio("Mostrar", ["Todos", "Solo errores"], horizontal=True,
                          key="eval_filter")
        view = detail if filtro == "Todos" else detail[~detail["acierto"]]
        st.caption(f"{len(view)} registro(s) · acierto global mostrado: "
                   f"{detail['acierto'].mean()*100:.1f}%")
        st.dataframe(
            view.rename(columns={
                "evento_id": "Evento", "localidad": "Localidad",
                "departamento": "Departamento", "magnitud": "Magnitud",
                "profundidad_km": "Prof. (km)", "distancia_km": "Dist. (km)",
                "categoria_intensidad": "Real", "prediccion": "Predicho",
                "acierto": "Acierto",
            }),
            width='stretch', hide_index=True, height=340,
        )

    ui.disclaimer_footer()
