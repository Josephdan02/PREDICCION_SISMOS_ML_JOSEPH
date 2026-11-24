"""Configuracion central del proyecto SISMO AI PERU.

Todas las rutas son relativas a la raiz del proyecto para que la aplicacion
funcione igual en cualquier maquina.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_PATH = DATA_DIR / "sismos_peru.csv"
DEFAULT_DATASET_PATH = PROCESSED_DIR / "sismos_peru_default.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "knn_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

ASSETS_DIR = ROOT_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
STYLES_PATH = ASSETS_DIR / "styles.css"

REPORTS_DIR = ROOT_DIR / "reports"
CSV_EXPORT_PATH = REPORTS_DIR / "resultados_simulaciones.csv"

# ---------------------------------------------------------------------------
# Esquema del dataset
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "evento_id",
    "fecha",
    "latitud",
    "longitud",
    "localidad",
    "departamento",
    "magnitud",
    "escala_magnitud",
    "profundidad_km",
    "distancia_km",
    "intensidad_observada",
    "categoria_intensidad",
    "tipo_registro",
    "fuente",
]

# Variables predictoras: solo numericas continuas entran al Pipeline.
FEATURES = ["magnitud", "profundidad_km", "distancia_km"]
TARGET = "categoria_intensidad"
GROUP_COLUMN = "evento_id"

# Orden canonico de las clases (se usa en todo el sistema).
CLASS_ORDER = ["Baja", "Moderada", "Fuerte"]

# ---------------------------------------------------------------------------
# Escala de intensidad (Mercalli modificada, agrupada)
# ---------------------------------------------------------------------------
INTENSITY_GROUPS = {
    "Baja": list(range(1, 5)),        # I - IV
    "Moderada": [5, 6],               # V - VI
    "Fuerte": list(range(7, 13)),     # VII - XII
}

INTENSITY_RANGE = {cat: f"{min(v)}–{max(v)}" for cat, v in INTENSITY_GROUPS.items()}

CLASS_COLORS = {
    "Baja": "#22d3ee",       # cian
    "Moderada": "#a78bfa",   # violeta
    "Fuerte": "#f472b6",     # rosa electrico
}

# ---------------------------------------------------------------------------
# Paleta visual del sistema
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#070b18",
    "bg_soft": "#0d1428",
    "surface": "rgba(255, 255, 255, 0.045)",
    "border": "rgba(255, 255, 255, 0.10)",
    "text": "#e9eefc",
    "muted": "#97a3c2",
    "cyan": "#22d3ee",
    "violet": "#a78bfa",
    "blue": "#4f8cff",
    "pink": "#f472b6",
    "amber": "#fbbf24",
    "green": "#34d399",
}

PLOTLY_SEQUENCE = [
    "#4f8cff", "#a78bfa", "#22d3ee", "#f472b6", "#fbbf24", "#34d399",
]

# ---------------------------------------------------------------------------
# Hiperparametros y validaciones
# ---------------------------------------------------------------------------
K_CANDIDATES = [3, 5, 7, 9]
DEFAULT_K = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_SPLITS = 5

MODEL_NAME = "KNN"

# Registros del dataset academico por defecto (>= 500 exigidos).
SYNTHETIC_RECORDS = 600
# Version del generador sintetico: si cambia, el dataset por defecto se regenera.
DATASET_VERSION = 7

# Rangos admitidos por el simulador y la validacion.
MAGNITUDE_RANGE = (3.0, 9.5)
DEPTH_RANGE = (0.0, 700.0)
DISTANCE_RANGE = (0.0, 1000.0)

# Mapa
PERU_CENTER = (-9.19, -75.02)
PERU_ZOOM = 5
HUANUCO_COORDS = (-9.93, -76.24)

DISCLAIMER = (
    "Este sistema tiene fines educativos y experimentales. No predice terremotos, "
    "no constituye un sistema de alerta temprana ni reemplaza las evaluaciones oficiales."
)

INSTITUTION = "SENATI"
PROJECT_NAME = "SISMO AI PERÚ"
PROJECT_TAGLINE = "Sistema Inteligente de Análisis y Estimación Sísmica con KNN"


def ensure_directories() -> None:
    """Crea las carpetas de trabajo si no existen."""
    for folder in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
