"""Generacion del informe PDF del proyecto con datos y metricas REALES."""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import config

ACCENT = colors.HexColor("#6d5bd0")
ACCENT_2 = colors.HexColor("#0ea5e9")
INK = colors.HexColor("#1f2330")
MUTED = colors.HexColor("#6b7280")
LIGHT = colors.HexColor("#f3f4f8")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("TitleCustom", parent=ss["Title"], fontName="Helvetica-Bold",
                          fontSize=24, textColor=ACCENT, spaceAfter=4, alignment=TA_CENTER))
    ss.add(ParagraphStyle("Subtitle", parent=ss["Normal"], fontSize=11,
                          textColor=MUTED, alignment=TA_CENTER, spaceAfter=14))
    ss.add(ParagraphStyle("H1Custom", parent=ss["Heading1"], fontName="Helvetica-Bold",
                          fontSize=15, textColor=ACCENT, spaceBefore=16, spaceAfter=6))
    ss.add(ParagraphStyle("H2Custom", parent=ss["Heading2"], fontName="Helvetica-Bold",
                          fontSize=11.5, textColor=INK, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("BodyJust", parent=ss["BodyText"], fontSize=9.4, leading=14.2,
                          alignment=TA_JUSTIFY, textColor=INK, spaceAfter=6))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], fontSize=8, textColor=MUTED,
                          leading=11))
    ss.add(ParagraphStyle("Warn", parent=ss["Normal"], fontSize=8.6, leading=12,
                          textColor=colors.HexColor("#92400e")))
    return ss


def _table(data, col_widths=None, header=True):
    tbl = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]))
    tbl.setStyle(TableStyle(style))
    return tbl


def _pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _report_row(rep: dict, cat: str) -> dict:
    """Extrae las metricas de una categoria de forma tolerante al formato."""
    r = rep.get(cat, {}) or {}
    return {
        "precision": r.get("precision", 0),
        "recall": r.get("recall", 0),
        "f1_score": r.get("f1_score", r.get("f1-score", 0)),
        "soporte": r.get("soporte", r.get("support", 0)),
    }


def build_report_pdf(meta: dict, dataset_summary: dict,
                     simulations: pd.DataFrame | None = None) -> bytes:
    """Construye el PDF en memoria y devuelve los bytes."""
    ss = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="SISMO AI PERU - Informe Tecnico", author="SENATI",
    )
    story: list = []

    # Portada.
    story.append(Spacer(1, 1.4 * cm))
    story.append(Paragraph(config.INSTITUTION, ss["Subtitle"]))
    story.append(Paragraph(config.PROJECT_NAME, ss["TitleCustom"]))
    story.append(Paragraph(config.PROJECT_TAGLINE, ss["Subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=16))

    ds = meta.get("dataset", {})
    info = [
        ["Campo", "Detalle"],
        ["Proyecto", "SISMO AI PERÚ — Clasificación de intensidad sísmica con KNN"],
        ["Institución", "SENATI — Reto individual de Inteligencia Artificial"],
        ["Modelo", f"{meta.get('modelo', 'KNN')} (StandardScaler + KNeighborsClassifier)"],
        ["K seleccionado", str(meta.get("k_seleccionado", "—"))],
        ["Registros del dataset", f"{ds.get('total_registros', 0)}"],
        ["Eventos únicos", f"{ds.get('eventos', 0)}"],
        ["Fecha de entrenamiento", meta.get("fecha_entrenamiento", "—")],
        ["Fecha de generación del informe", datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    story.append(_table(info, col_widths=[5.0 * cm, 11.4 * cm]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Advertencia:</b> " + config.DISCLAIMER, ss["Warn"]))
    story.append(PageBreak())

    # 1. Objetivo
    story.append(Paragraph("1. Objetivo", ss["H1Custom"]))
    story.append(Paragraph(
        "Desarrollar una aplicación web académica que estime la categoría de intensidad "
        "sísmica (Baja, Moderada o Fuerte) de un escenario hipotético en territorio peruano, "
        "utilizando un clasificador K-Nearest Neighbors entrenado con las variables magnitud, "
        "profundidad hipocentral y distancia epicentral. El sistema demuestra el ciclo completo "
        "de un proyecto de Machine Learning: preparación de datos, entrenamiento, validación "
        "agrupada, evaluación sobre un conjunto de prueba independiente, persistencia del modelo "
        "y su uso en una interfaz interactiva.", ss["BodyJust"]))

    # 2. Dataset
    story.append(Paragraph("2. Descripción del dataset", ss["H1Custom"]))
    tipos = ds.get("tipos_registro", {}) or {}
    tipos_txt = ", ".join(f"{k}: {v}" for k, v in tipos.items()) or "—"
    story.append(Paragraph(
        f"El dataset contiene <b>{ds.get('total_registros', 0)}</b> registros evento-localidad "
        f"agrupados en <b>{ds.get('eventos', 0)}</b> eventos, distribuidos en "
        f"<b>{ds.get('departamentos', 0)}</b> departamentos del Perú. Los registros por tipo son: "
        f"{tipos_txt}.", ss["BodyJust"]))
    story.append(Paragraph("2.1 Procedencia y naturaleza de los datos", ss["H2Custom"]))
    story.append(Paragraph(
        "Los catálogos públicos reales (USGS ComCat, IGP/CENSIS) documentan magnitud, profundidad "
        "y coordenadas, pero un catálogo de magnitudes no contiene una intensidad Mercalli "
        "observada por localidad de forma completa. El atributo <i>mmi</i> del catálogo USGS "
        "corresponde a una estimación (ShakeMap/estaciones) y no a observaciones exhaustivas, y "
        "solo está disponible para una fracción de eventos recientes. Por esta limitación, el "
        "dataset principal de este proyecto es <b>simulado con fines académicos</b>: se generó con "
        "un modelo probabilístico de atenuación (magnitudes Gutenberg-Richter truncadas, "
        "profundidades y distancias con variabilidad realista, ruido gaussiano sobre la "
        "intensidad) para evitar un etiquetado mediante una regla determinista. Todos los "
        "registros están marcados como <b>SIMULADO</b> y no se presentan como observaciones "
        "oficiales del IGP ni del USGS.", ss["BodyJust"]))

    story.append(Paragraph("2.2 Preparación y limpieza", ss["H2Custom"]))
    story.append(Paragraph(
        "Se normalizaron tipos de datos, se eliminaron registros con valores faltantes en columnas "
        "críticas, se descartaron duplicados exactos, se validaron los rangos físicos (magnitud, "
        "profundidad, distancia e intensidad entre I y XII) y se agruparon las intensidades en tres "
        "categorías: I–IV = Baja, V–VI = Moderada, VII–XII = Fuerte. Los registros sin coordenadas "
        "válidas se excluyen únicamente del mapa, no del entrenamiento.", ss["BodyJust"]))

    # 3. Metodología
    story.append(Paragraph("3. Metodología KNN", ss["H1Custom"]))
    story.append(Paragraph(
        "Se utilizó un Pipeline de Scikit-learn con <b>StandardScaler</b> + "
        "<b>KNeighborsClassifier</b>, de modo que la normalización (media 0, desviación 1) se "
        "calcula únicamente con los datos de entrenamiento y se aplica de forma automática a los "
        "nuevos datos, evitando la fuga de información. La división de los datos es "
        "<b>80% entrenamiento / 20% prueba</b> mediante <b>GroupShuffleSplit</b> agrupado por "
        "<b>evento_id</b>: todas las localidades de un mismo evento quedan en el mismo lado de la "
        "división.", ss["BodyJust"]))
    story.append(Paragraph(
        f"La validación cruzada se realizó exclusivamente dentro del conjunto de entrenamiento con "
        f"<b>{meta.get('cv_estrategia', 'StratifiedGroupKFold')}</b> "
        f"({meta.get('cv_folds', '—')} pliegues). Se compararon los valores K = "
        f"{', '.join(str(k) for k in meta.get('k_comparados', []))}.", ss["BodyJust"]))

    # 4. Comparación K
    story.append(Paragraph("4. Comparación de valores K", ss["H1Custom"]))
    cv = meta.get("cv_resultados", {})
    rows = [["K", "Accuracy", "Precision macro", "Recall macro", "F1 macro"]]
    for k in sorted(cv, key=lambda x: int(x)):
        r = cv[k]
        mark = "  ← seleccionado" if int(k) == meta.get("k_seleccionado") else ""
        rows.append([f"{k}{mark}", _pct(r["accuracy"]), _pct(r["precision_macro"]),
                     _pct(r["recall_macro"]), _pct(r["f1_macro"])])
    story.append(_table(rows, col_widths=[4.0 * cm, 3.0 * cm, 3.4 * cm, 3.0 * cm, 3.0 * cm]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Criterio de selección:</b> {meta.get('criterio_seleccion', '')} "
        "En caso de empate se prefiere el mayor recall macro y, si persiste, el K menor "
        "(modelo más simple).", ss["BodyJust"]))

    # 5. Evaluación
    story.append(Paragraph("5. Resultados de evaluación (conjunto de prueba)", ss["H1Custom"]))
    mt = meta.get("metricas_test", {})
    story.append(Paragraph(
        "Las siguientes métricas corresponden a la evaluación única del modelo seleccionado "
        "sobre el conjunto de prueba reservado desde el inicio, no sobre datos de entrenamiento.",
        ss["BodyJust"]))
    story.append(_table(
        [["Métrica", "Valor"],
         ["Accuracy", _pct(mt.get("accuracy"))],
         ["Precision macro", _pct(mt.get("precision_macro"))],
         ["Recall macro", _pct(mt.get("recall_macro"))],
         ["F1-score macro", _pct(mt.get("f1_macro"))],
         ["Aciertos / Total", f"{meta.get('aciertos_test', '—')} / {meta.get('total_test', '—')}"]],
        col_widths=[7.0 * cm, 9.4 * cm]))

    story.append(Paragraph("5.1 Matriz de confusión", ss["H2Custom"]))
    cm_data = meta.get("matriz_confusion", {})
    if cm_data:
        labels = [f"Pred {c}" for c in config.CLASS_ORDER]
        rows = [["Real \\ Pred"] + [c.replace("Pred ", "") for c in labels]]
        for real in config.CLASS_ORDER:
            key = f"Real {real}"
            rows.append([real] + [str(int(cm_data.get(key, {}).get(pred, 0))) for pred in labels])
        story.append(_table(rows, col_widths=[5.0 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Las filas representan la categoría real y las columnas la categoría predicha. "
            "Los valores de la diagonal son las clasificaciones correctas; los valores fuera de "
            "ella indican confusiones entre categorías contiguas o extremas.", ss["BodyJust"]))

    story.append(Paragraph("5.2 Informe por categoría", ss["H2Custom"]))
    rep = meta.get("informe_clasificacion", {})
    if rep:
        rows = [["Categoría", "Precision", "Recall", "F1", "Soporte"]]
        for cat in config.CLASS_ORDER:
            r = _report_row(rep, cat)
            rows.append([cat, _pct(r["precision"]), _pct(r["recall"]),
                         _pct(r["f1_score"]), str(int(float(r["soporte"] or 0)))])
        macro = _report_row(rep, "macro avg")
        if macro["precision"] or macro["recall"]:
            rows.append(["Promedio macro", _pct(macro["precision"]), _pct(macro["recall"]),
                         _pct(macro["f1_score"]), str(int(float(macro["soporte"] or 0)))])
        story.append(_table(rows, col_widths=[4.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.4 * cm]))

    # 6. Simulaciones
    if simulations is not None and not simulations.empty:
        story.append(Paragraph("6. Registros de simulaciones", ss["H1Custom"]))
        rows = [["Fecha", "Localidad", "M", "Prof.", "Dist.", "K", "Categoría", "Prob."]]
        for _, r in simulations.tail(20).iterrows():
            rows.append([
                str(r.get("fecha_hora", ""))[:16], str(r.get("departamento", "")),
                f"{float(r.get('magnitud', 0)):.1f}", f"{float(r.get('profundidad_km', 0)):.0f}",
                f"{float(r.get('distancia_km', 0)):.0f}", str(r.get("k", "")),
                str(r.get("categoria_estimada", "")), _pct(r.get("probabilidad", 0)),
            ])
        story.append(_table(rows, col_widths=[2.8 * cm, 3.0 * cm, 1.4 * cm, 1.7 * cm,
                                              1.7 * cm, 1.2 * cm, 2.4 * cm, 2.0 * cm]))
        story.append(Paragraph(
            "Las probabilidades mostradas corresponden a la proporción de votos de los vecinos "
            "más cercanos, no a probabilidades calibradas de ocurrencia de un terremoto.",
            ss["Small"]))
        section_conclusiones = "7"
    else:
        section_conclusiones = "6"

    # 7. Conclusiones
    story.append(Paragraph(f"{section_conclusiones}. Conclusiones", ss["H1Custom"]))
    story.append(Paragraph(
        f"El modelo KNN con K = {meta.get('k_seleccionado')} obtuvo una exactitud de "
        f"{_pct(mt.get('accuracy'))} y un F1-score macro de {_pct(mt.get('f1_macro'))} sobre el "
        "conjunto de prueba independiente. La separación agrupada por evento evitó que localidades "
        "de un mismo sismo estuvieran simultáneamente en entrenamiento y prueba, entregando una "
        "estimación más honesta del desempeño. La normalización dentro del Pipeline garantiza que "
        "las tres variables contribuyan de forma equilibrada al cálculo de distancias. El K "
        "seleccionado fue determinado únicamente por validación cruzada dentro del entrenamiento.",
        ss["BodyJust"]))
    siguiente = int(section_conclusiones) + 1
    story.append(Paragraph(f"{siguiente}. Limitaciones", ss["H1Custom"]))
    for txt in [
        "El dataset principal es simulado: las conclusiones se limitan a ese escenario experimental "
        "y no constituyen una caracterización de la sismicidad real del Perú.",
        "Las etiquetas de intensidad provienen de un modelo probabilístico de atenuación, no de "
        "observaciones instrumentales o de daño físico reportado por localidad.",
        "KNN es sensible a la escala y al número de vecinos; su frontera de decisión puede verse "
        "afectada por desbalance de clases y por registros ruidosos del catálogo.",
        "El sistema estima intensidades para escenarios hipotéticos. No predice la fecha, ubicación, "
        "ocurrencia ni magnitud de un terremoto futuro y no es un sistema de alerta temprana.",
    ]:
        story.append(Paragraph(f"• {txt}", ss["BodyJust"]))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#e5e7eb")))
    story.append(Paragraph(config.DISCLAIMER, ss["Small"]))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.1 * cm, "SISMO AI PERÚ — SENATI — Informe académico")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Página {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
