"""Datos de referencia geografica del Peru (uso academico).

IMPORTANTE SOBRE LAS COORDENADAS:
    Las coordenadas de los departamentos son aproximadas y se usan SOLO para:
      (a) calcular la distancia epicentral en el generador de escenarios
          simulados, y
      (b) ubicar las localidades simuladas en el mapa.
    NO son epicentros de eventos reales. Cada registro simulado declara su
    origen en la columna ``tipo_registro`` y ``fuente``.

    Las localidades incluidas son toponimos y capitales/distritos conocidos
    para dar contexto textual; sus coordenadas son aproximadas.
"""

from __future__ import annotations

# Departamento -> (latitud, longitud, city/localidad representativa)
DEPARTAMENTOS: dict[str, tuple[float, float, str]] = {
    "Amazonas": (-6.23, -77.87, "Chachapoyas"),
    "Áncash": (-9.53, -77.53, "Huaraz"),
    "Apurímac": (-13.63, -72.88, "Abancay"),
    "Arequipa": (-16.41, -71.54, "Arequipa"),
    "Ayacucho": (-13.16, -74.22, "Ayacucho"),
    "Cajamarca": (-7.16, -78.51, "Cajamarca"),
    "Callao": (-12.06, -77.12, "Callao"),
    "Cusco": (-13.53, -71.97, "Cusco"),
    "Huancavelica": (-12.78, -74.97, "Huancavelica"),
    "Huánuco": (-9.93, -76.24, "Huánuco"),
    "Ica": (-14.07, -75.73, "Ica"),
    "Junín": (-12.06, -75.21, "Huancayo"),
    "La Libertad": (-8.11, -79.03, "Trujillo"),
    "Lambayeque": (-6.77, -79.84, "Chiclayo"),
    "Lima": (-12.05, -77.04, "Lima"),
    "Loreto": (-3.75, -73.25, "Iquitos"),
    "Madre de Dios": (-12.59, -69.18, "Puerto Maldonado"),
    "Moquegua": (-17.19, -70.93, "Moquegua"),
    "Pasco": (-10.68, -76.26, "Cerro de Pasco"),
    "Piura": (-5.19, -80.63, "Piura"),
    "Puno": (-15.84, -70.02, "Puno"),
    "San Martín": (-6.49, -76.37, "Tarapoto"),
    "Tacna": (-18.01, -70.25, "Tacna"),
    "Tumbes": (-3.57, -80.45, "Tumbes"),
    "Ucayali": (-8.38, -74.55, "Pucallpa"),
}

DEPARTAMENTOS_LIST = sorted(DEPARTAMENTOS.keys())

# Localidades adicionales por departamento (toponimos conocidos).
LOCALIDADES_EXTRA: dict[str, list[str]] = {
    "Áncash": ["Chimbote", "Casma", "Huaraz", "Yungay", "Caraz"],
    "Arequipa": ["Arequipa", "Camaná", "Mollendo", "Chala", "Atico"],
    "Lima": ["Lima", "Cañete", "Chosica", "Barranca", "Huaral", "San Vicente de Cañete"],
    "Ica": ["Ica", "Pisco", "Chincha Alta", "Nazca", "Paracas"],
    "La Libertad": ["Trujillo", "Pacasmayo", "Salaverry", "Santiago de Chuco"],
    "Lambayeque": ["Chiclayo", "Lambayeque", "Mórrope", "Pimentel"],
    "Piura": ["Piura", "Talara", "Paita", "Sechura", "Sullana"],
    "Tumbes": ["Tumbes", "Zorritos", "Zarumilla"],
    "Moquegua": ["Moquegua", "Ilo", "Omate"],
    "Tacna": ["Tacna", "Ilo", "Locumba"],
    "Cusco": ["Cusco", "Sicuani", "Espinar", "Quillabamba"],
    "Puno": ["Puno", "Juliaca", "Azángaro", "Lampa"],
    "Ayacucho": ["Ayacucho", "Puquio", "Huanta", "Lucanas"],
    "Huancavelica": ["Huancavelica", "Pampas", "Castrovirreyna"],
    "Junín": ["Huancayo", "Tarma", "Jauja", "La Merced", "Satipo"],
    "Huánuco": ["Huánuco", "Tingo María", "La Unión", "Ambo"],
    "Pasco": ["Cerro de Pasco", "Oxapampa", "Yanahuanca"],
    "Cajamarca": ["Cajamarca", "Jaén", "Chota", "San Ignacio"],
    "Amazonas": ["Chachapoyas", "Bagua", "Luya", "Rodríguez de Mendoza"],
    "San Martín": ["Tarapoto", "Moyobamba", "Juanjuí", "Tocache"],
    "Loreto": ["Iquitos", "Yurimaguas", "Contamana"],
    "Ucayali": ["Pucallpa", "Atalaya", "Padre Abad"],
    "Madre de Dios": ["Puerto Maldonado", "Iberia", "Mazuko"],
    "Apurímac": ["Abancay", "Andahuaylas", "Chuquibambilla"],
    "Callao": ["Callao", "La Punta", "Ventanilla"],
}

# ---------------------------------------------------------------------------
# Ley de atenuacion sintetica (usada SOLO por el generador de datos simulados)
# ---------------------------------------------------------------------------
# Forma: I = A + B*mag - C*log10(dist_hipo) - D*depth_km + ruido
# Los coeficientes se inspiran cualitativamente en modelos tipo Gutenberg-Richter /
# atenuacion de intensidad (I crece con mag, decrece con distancia y profundidad).
ATTENUATION_A = 3.1
ATTENUATION_B = 1.42
ATTENUATION_C = 2.10
ATTENUATION_D = 0.0050
ATTENUATION_SIGMA = 0.60  # desviacion estandar del ruido gaussiano (unidades MMI)

# Fraccion de registros de intensidad baja generados con epicentros lejanos.
FRACCION_LEJANOS = 0.26

# Fraccion minima de eventos "cercanos" que producen las clases Moderada/Fuerte.
FRACCION_CERCANOS = 0.55

# Distribucion de magnitudes (mezcla de Gutenberg-Richter truncada).
MAG_MIN = 4.5
MAG_MAX = 8.6
MAG_B_VALUE = 1.0  # pendiente b de Gutenberg-Richter para muestreo rejection-free

# Distribucion de profundidades (km): sismos superficiales e intermedios.
DEPTH_LOW = (5.0, 35.0)
DEPTH_MID = (35.0, 160.0)
DEPTH_DEEP = (160.0, 650.0)
DEPTH_WEIGHTS = (0.62, 0.30, 0.08)

# Distancias epicentrales (km) a localidades cercanas.
DISTANCE_GAMMA_SHAPE = 2.4
DISTANCE_GAMMA_SCALE = 60.0
DISTANCE_MAX = 420.0

# Metadatos de origen.
TIPO_REGISTRO_SIMULADO = "SIMULADO"
FUENTE_SIMULADA = "Generador sintético académico SISMO AI PERÚ (modelo de atenuación)"

FUENTE_OBSERVADA = "USGS ComCat — datos aportados por la comunidad (CDI)"
FUENTE_MMI = "USGS ComCat — MMI estimado (ShakeMap/estaciones)"

# Escenarios de demostracion del simulador (solo parametros de entrada).
DEMO_SCENARIOS = [
    {
        "nombre": "Sismo profundo de Iquitos",
        "departamento": "Loreto",
        "magnitud": 6.1,
        "profundidad_km": 145.0,
        "distancia_km": 30.0,
        "k": 5,
        "descripcion": "Evento intermedio-profundo en la Amazonía, lejos del margen costero.",
    },
    {
        "nombre": "Réplica moderada en Ica",
        "departamento": "Ica",
        "magnitud": 5.6,
        "profundidad_km": 32.0,
        "distancia_km": 42.0,
        "k": 5,
        "descripcion": "Sismo superficial cerca de una ciudad intermedia con alta sismicidad.",
    },
    {
        "nombre": "Gran evento frente a la costa",
        "departamento": "Áncash",
        "magnitud": 7.8,
        "profundidad_km": 22.0,
        "distancia_km": 85.0,
        "k": 5,
        "descripcion": "Escenario tipo terremoto costero de gran magnitud y poca profundidad.",
    },
]
