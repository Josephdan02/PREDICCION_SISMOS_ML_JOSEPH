"""Preprocesamiento y generacion del dataset academico.

Responsabilidades:
    * Generacion de un dataset SIMULADO reproducible (modelo probabilistico
      de atenuacion, NO una regla determinista) con >= 500 registros.
    * Normalizacion de esquema y tipos.
    * Limpieza (nulos, duplicados, rangos) y validacion del dataset.
    * Derivacion de la categoria de intensidad a partir del MMI observado.

AVISO ACADEMICO:
    El dataset por defecto es 100 % SIMULADO. Los registros se identifican
    con ``tipo_registro = "SIMULADO"``. Nunca se presentan como observaciones
    reales del IGP ni del USGS.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config
from . import peru_reference as ref


# ---------------------------------------------------------------------------
# Utilidades de etiquetas
# ---------------------------------------------------------------------------
def mmi_to_category(mmi: float) -> str:
    """Agrupa una intensidad (1-12) en la categoria de 3 clases."""
    if pd.isna(mmi):
        raise ValueError("Intensidad vacía no puede clasificarse")
    mmi = float(mmi)
    for category, values in config.INTENSITY_GROUPS.items():
        if int(round(mmi)) in values:
            return category
    raise ValueError(f"Intensidad fuera de escala: {mmi}")


def category_for_value(intensity: float) -> str:
    """Igual que mmi_to_category pero recortando a 1..12."""
    value = int(np.clip(round(float(intensity)), 1, 12))
    for category, values in config.INTENSITY_GROUPS.items():
        if value in values:
            return category
    return "Fuerte"


# ---------------------------------------------------------------------------
# Generador de escenarios simulados
# ---------------------------------------------------------------------------
def _sample_gutenberg_richter(rng: np.random.Generator, n: int) -> np.ndarray:
    """Muestreo de magnitudes con distribucion Gutenberg-Richter truncada.

    P(M > m) ~ 10^(-b*(m - Mmin)), en [MAG_MIN, MAG_MAX].
    """
    b = ref.MAG_B_VALUE
    lo, hi = ref.MAG_MIN, ref.MAG_MAX
    u = rng.random(n)
    return lo - np.log10(1.0 - u * (1.0 - 10 ** (-b * (hi - lo)))) / b


def _sample_depths(rng: np.random.Generator, n: int, shallow_bias: float = 0.0) -> np.ndarray:
    """Mezcla de sismos superficiales, intermedios y profundos.

    ``shallow_bias`` desplaza probabilidad de las bandas intermedias/profundas
    hacia la banda superficial (usado para los eventos "cercanos").
    """
    weights = np.array(ref.DEPTH_WEIGHTS, dtype=float)
    if shallow_bias > 0:
        take = shallow_bias * weights[1:].sum()
        weights = weights.copy()
        weights[1:] *= (1 - shallow_bias)
        weights[0] += take
    weights = weights / weights.sum()
    choice = rng.choice([0, 1, 2], size=n, p=weights)
    depths = np.empty(n)
    for idx, band in enumerate((ref.DEPTH_LOW, ref.DEPTH_MID, ref.DEPTH_DEEP)):
        mask = choice == idx
        count = int(mask.sum())
        if count:
            depths[mask] = rng.uniform(band[0], band[1], size=count)
    return depths


def _sample_distances(rng: np.random.Generator, n: int) -> np.ndarray:
    """Distancias epicentrales con cola larga (Gamma truncada)."""
    d = rng.gamma(ref.DISTANCE_GAMMA_SHAPE, ref.DISTANCE_GAMMA_SCALE, size=n)
    return np.clip(d, 25.0, ref.DISTANCE_MAX)


def _sample_near_distances(rng: np.random.Generator, n: int) -> np.ndarray:
    """Distancias para eventos que producen intensidades medias/altas."""
    return rng.uniform(20.0, 110.0, size=n)


def _sample_far_distances(rng: np.random.Generator, n: int) -> np.ndarray:
    """Distancias para eventos que producen intensidades bajas."""
    return rng.uniform(150.0, ref.DISTANCE_MAX, size=n)


def synthetic_intensity(
    magnitudes: np.ndarray,
    depths: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Intensidad MMI sintetica con modelo de atenuacion + ruido gaussiano.

    I = A + B*mag - C*log10(R) - D*depth + epsilon,  epsilon ~ N(0, sigma)
    donde R = sqrt(distancia^2 + profundidad^2) es la distancia hipocentral.

    El ruido y los coeficientes hacen que la etiqueta NO sea una funcion
    determinista de las tres variables: hay solapamiento entre clases.
    """
    hypocentral = np.sqrt(distances ** 2 + (depths * 0.6) ** 2) + 5.0
    base = (
        ref.ATTENUATION_A
        + ref.ATTENUATION_B * magnitudes
        - ref.ATTENUATION_C * np.log10(hypocentral)
        - ref.ATTENUATION_D * depths
    )
    noise = rng.normal(0.0, ref.ATTENUATION_SIGMA, size=len(magnitudes))
    return np.clip(base + noise, 1.0, 12.0)


def generate_synthetic_dataset(
    n_records: int = 600,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Genera el dataset SIMULADO evento-localidad.

    Cada evento real se representa por 1..4 localidades (hasta 4), por lo que
    ``evento_id`` agrupa registros que NO deben separarse entre train y test.
    """
    rng = np.random.default_rng(random_state)

    departments = list(ref.DEPARTAMENTOS.keys())
    # Mayor densidad en la sismicidad del margen andino-costero.
    weights = np.array(
        [
            1.6 if d in {
                "Áncash", "Arequipa", "Ica", "Lima", "Moquegua", "Tacna",
                "La Libertad", "Lambayeque", "Piura", "Cusco", "Puno",
                "Ayacucho", "Junín", "Huánuco",
            } else 0.8
            for d in departments
        ]
    )
    weights = weights / weights.sum()

    rows: list[dict] = []
    evento_idx = 0
    start_date = datetime(2010, 1, 1)

    while len(rows) < n_records:
        evento_idx += 1
        dept = rng.choice(departments, p=weights)
        base_lat, base_lon, _ = ref.DEPARTAMENTOS[dept]

        # Epicentro cerca del departamento (no exactamente en su capital).
        epic_lat = base_lat + rng.normal(0, 0.35)
        epic_lon = base_lon + rng.normal(0, 0.30)

        mag = float(_sample_gutenberg_richter(rng, 1)[0])
        roll = rng.random()
        event_near = roll < ref.FRACCION_CERCANOS
        event_far = roll > 1.0 - ref.FRACCION_LEJANOS
        if event_far:
            # Evento distante/profundo: intensidad baja en localidades lejanas.
            depth = float(_sample_depths(rng, 1, shallow_bias=-0.30)[0])
        elif event_near:
            # Evento cercano: superficial, mayor magnitud y distancias cortas.
            depth = float(_sample_depths(rng, 1, shallow_bias=0.45)[0])
            if rng.random() < 0.35:
                mag = float(np.clip(mag + 0.7, ref.MAG_MIN, ref.MAG_MAX))
        else:
            depth = float(_sample_depths(rng, 1)[0])
        date = start_date + timedelta(days=float(rng.uniform(0, 5600)))

        n_localities = int(rng.choice([1, 1, 2, 2, 3, 4]))
        localities = ref.LOCALIDADES_EXTRA.get(dept, [ref.DEPARTAMENTOS[dept][2]])
        chosen = rng.choice(localities, size=min(n_localities, len(localities)), replace=False)

        for loc in chosen:
            if event_far:
                dist = float(_sample_far_distances(rng, 1)[0])
            elif event_near and rng.random() < 0.65:
                dist = float(_sample_near_distances(rng, 1)[0])
            else:
                dist = float(_sample_distances(rng, 1)[0])
            # Desplazamiento de la localidad respecto del epicentro, usando la
            # distancia epicentral y un azimut aleatorio.
            ang = rng.uniform(0, 2 * np.pi)
            km_per_deg = 111.0
            loc_lat = epic_lat + (dist / km_per_deg) * np.sin(ang)
            loc_lon = epic_lon + (dist / (km_per_deg * np.cos(np.radians(epic_lat)))) * np.cos(ang)

            intensity = float(
                synthetic_intensity(
                    np.array([mag]), np.array([depth]), np.array([dist]), rng
                )[0]
            )
            mmi = int(np.clip(round(intensity), 1, 12))

            rows.append(
                {
                    "evento_id": f"SIM-EQ{evento_idx:05d}",
                    "fecha": date.strftime("%Y-%m-%d"),
                    "latitud": round(float(loc_lat), 5),
                    "longitud": round(float(loc_lon), 5),
                    "localidad": str(loc),
                    "departamento": dept,
                    "magnitud": round(mag, 1),
                    "escala_magnitud": "Mw",
                    "profundidad_km": round(depth, 1),
                    "distancia_km": round(dist, 1),
                    "intensidad_observada": mmi,
                    "categoria_intensidad": mmi_to_category(mmi),
                    "tipo_registro": ref.TIPO_REGISTRO_SIMULADO,
                    "fuente": ref.FUENTE_SIMULADA,
                }
            )
            if len(rows) >= n_records:
                break

    df = pd.DataFrame(rows)
    return df[config.REQUIRED_COLUMNS]


# ---------------------------------------------------------------------------
# Normalizacion de esquema
# ---------------------------------------------------------------------------
def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura columnas, tipos y orden canonicos."""
    if df is None or df.empty:
        return pd.DataFrame(columns=config.REQUIRED_COLUMNS)

    df = df.copy()
    for col in config.REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[config.REQUIRED_COLUMNS]

    numeric_cols = ["latitud", "longitud", "magnitud", "profundidad_km",
                    "distancia_km", "intensidad_observada"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for col in ["evento_id", "localidad", "departamento", "escala_magnitud",
                "categoria_intensidad", "tipo_registro", "fuente"]:
        df[col] = df[col].astype("string").str.strip()

    # Derivar categoria si falta pero hay intensidad.
    missing_cat = df["categoria_intensidad"].isna() & df["intensidad_observada"].notna()
    if missing_cat.any():
        df.loc[missing_cat, "categoria_intensidad"] = df.loc[missing_cat,
            "intensidad_observada"].apply(category_for_value)

    return df


# ---------------------------------------------------------------------------
# Limpieza y validacion
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpia y valida. Devuelve (df_limpio, reporte_de_limpieza)."""
    df = normalize_schema(df)
    report = {"filas_iniciales": int(len(df))}

    if df.empty:
        report.update({"filas_finales": 0, "eliminados_nulos": 0,
                        "eliminados_duplicados": 0, "eliminados_fuera_rango": 0})
        return df, report

    # 1. Nulos en columnas criticas.
    critical = ["evento_id", "magnitud", "profundidad_km", "distancia_km",
                "intensidad_observada", "categoria_intensidad"]
    before = len(df)
    df = df.dropna(subset=critical)
    report["eliminados_nulos"] = before - len(df)

    # 2. Duplicados exactos.
    before = len(df)
    df = df.drop_duplicates()
    report["eliminados_duplicados"] = before - len(df)

    # 3. Rangos validos.
    before = len(df)
    df = df[
        df["magnitud"].between(*config.MAGNITUDE_RANGE)
        & df["profundidad_km"].between(*config.DEPTH_RANGE)
        & df["distancia_km"].between(*config.DISTANCE_RANGE)
        & df["intensidad_observada"].between(1, 12)
        & df["categoria_intensidad"].isin(config.CLASS_ORDER)
    ]
    report["eliminados_fuera_rango"] = before - len(df)

    # 4. Coordenadas validas (se conservan aunque falten: el mapa las excluye).
    lat_ok = df["latitud"].between(-90, 90)
    lon_ok = df["longitud"].between(-180, 180)
    df["tiene_coordenadas"] = (lat_ok & lon_ok).fillna(False)
    report["sin_coordenadas"] = int((~df["tiene_coordenadas"]).sum())

    df = df.reset_index(drop=True)
    report["filas_finales"] = int(len(df))
    return df, report


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo y porcentaje por categoria (orden canonico)."""
    counts = df["categoria_intensidad"].value_counts()
    out = pd.DataFrame(
        {"categoria_intensidad": config.CLASS_ORDER}
    )
    out["registros"] = out["categoria_intensidad"].map(counts).fillna(0).astype(int)
    total = out["registros"].sum()
    out["porcentaje"] = (out["registros"] / total * 100).round(1) if total else 0.0
    return out


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Devuelve lista de problemas detectados (vacia = dataset apto)."""
    issues: list[str] = []
    if df is None or df.empty:
        return ["El dataset está vacío o no se pudo cargar."]
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Faltan columnas obligatorias: {', '.join(missing)}.")
    if "categoria_intensidad" in df.columns:
        n_classes = df["categoria_intensidad"].nunique(dropna=True)
        if n_classes < 2:
            issues.append("Se requieren al menos 2 clases para entrenar el modelo.")
    if "evento_id" in df.columns and df["evento_id"].nunique() < 2:
        issues.append("Se requieren al menos 2 eventos distintos para la división agrupada.")
    return issues


def build_default_dataset(force: bool = False) -> pd.DataFrame:
    """Genera y guarda el dataset academico por defecto si no existe o esta desactualizado."""
    config.ensure_directories()
    version_file = config.PROCESSED_DIR / "dataset_version.txt"
    current_version = str(config.DATASET_VERSION)
    up_to_date = (
        config.DEFAULT_DATASET_PATH.exists()
        and version_file.exists()
        and version_file.read_text(encoding="utf-8").strip() == current_version
    )
    if up_to_date and not force:
        return pd.read_csv(config.DEFAULT_DATASET_PATH)

    df = generate_synthetic_dataset(config.SYNTHETIC_RECORDS)
    clean, _ = clean_dataset(df)
    clean.to_csv(config.DEFAULT_DATASET_PATH, index=False)
    version_file.write_text(current_version, encoding="utf-8")
    return clean
