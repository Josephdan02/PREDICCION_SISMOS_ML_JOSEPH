"""SISMO AI PERU - paquete principal del sistema.

Modulos:
    config          -> Rutas, constantes y configuracion global.
    data_loader     -> Carga, validacion y resumen del dataset.
    preprocessing   -> Limpieza, normalizacion de esquema y generacion del dataset.
    usgs_client     -> Descarga opcional del catalogo real de USGS.
    train_model     -> Pipeline KNN, division agrupada, comparacion de K y evaluacion.
    evaluation      -> Metricas y utilidades de evaluacion del modelo.
    prediction      -> Prediccion, probabilidades y vecinos mas cercanos.
    map_service     -> Construccion del mapa interactivo con Folium.
    charts          -> Graficos Plotly reutilizables con la paleta del sistema.
    report_generator-> Informe PDF con datos reales del sistema.
"""

__version__ = "1.0.0"
