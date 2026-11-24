# SISMO AI PERÚ

**Sistema Inteligente de Análisis y Estimación Sísmica con KNN**

Proyecto académico para **SENATI** — Reto individual de Inteligencia Artificial.

Aplicación web que estima la **categoría de intensidad sísmica** (Baja, Moderada, Fuerte)
esperada en una localidad para un escenario hipotético, usando un clasificador
**K-Nearest Neighbors** entrenado, validado con división agrupada por evento y evaluado
sobre un conjunto de prueba independiente.

> ⚠️ **Este sistema tiene fines educativos y experimentales. No predice terremotos, no
> constituye un sistema de alerta temprana ni reemplaza las evaluaciones oficiales.**

---

## 1. Descripción y objetivo

El sistema cubre el ciclo completo de un proyecto de Machine Learning aplicado a datos
sísmicos:

1. Obtención y documentación del dataset.
2. Limpieza, validación de tipos y rangos físicos.
3. Ingeniería de variables y agrupación de intensidades en tres categorías.
4. Entrenamiento con `Pipeline(StandardScaler → KNeighborsClassifier)`.
5. División **80/20 agrupada por `evento_id`** para evitar fuga de información.
6. Comparación de **K = 3, 5, 7, 9** con validación cruzada agrupada.
7. Evaluación **única** sobre el conjunto de prueba reservado.
8. Persistencia del modelo y uso en un simulador interactivo.
9. Generación de informes PDF y exportaciones CSV.

### Alcance real

| El sistema **sí** hace | El sistema **no** hace |
|---|---|
| Estimar la categoría de intensidad de un escenario dado | Predecir la fecha de un terremoto |
| Mostrar la distribución de votos de los vecinos | Predecir la ubicación de un futuro sismo |
| Visualizar los K vecinos y sus distancias | Predecir la ocurrencia de un terremoto |
| Evaluar el modelo con métricas estadísticas | Predecir la magnitud de un sismo futuro |
| Generar informes y exportaciones | Ser un sistema oficial de alerta temprana |

---

## 2. Tecnologías

| Componente | Versión mínima | Uso |
|---|---|---|
| Python | 3.11+ | Lenguaje base |
| Streamlit | 1.32+ | Interfaz web |
| Pandas / NumPy | 2.0 / 1.24 | Procesamiento y cálculo |
| Scikit-learn | 1.3+ | Pipeline KNN, validación, métricas |
| Plotly | 5.18+ | Gráficos interactivos |
| Folium + streamlit-folium | 0.15 / 0.18 | Mapa geográfico |
| Joblib | 1.3+ | Persistencia del modelo |
| ReportLab | 4.0+ | Informe PDF |
| Matplotlib | 3.7+ | Dependencia de soporte |
| Pytest | 7.4+ | Pruebas automatizadas |

---

## 3. Estructura del proyecto

```
PREDICCION_SISMOS_ML_JOSEPH/
├── app.py                     # Punto de entrada Streamlit + navegación
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml            # Tema oscuro y carpeta estática
├── data/
│   ├── sismos_peru.csv        # Dataset activo (600 registros)
│   ├── raw/
│   ├── processed/             # Dataset por defecto, test_set.csv, versión
│   └── README_DATASET.md      # Procedencia y metodología de los datos
├── models/
│   ├── knn_pipeline.joblib    # Pipeline + metadata + test reservado
│   └── model_metadata.json    # Métricas legibles sin cargar el modelo
├── src/
│   ├── config.py              # Rutas, paleta, constantes, rangos
│   ├── peru_reference.py      # Departamentos, localidades, atenuación
│   ├── preprocessing.py       # Generación simulada, limpieza, validación
│   ├── data_loader.py         # Carga y resumen del dataset
│   ├── train_model.py         # Pipeline, split agrupado, comparación de K
│   ├── evaluation.py          # Métricas, matriz de confusión, selección de K
│   ├── prediction.py          # predict / predict_proba / kneighbors
│   ├── map_service.py         # Mapa Folium con filtros y leyenda
│   ├── charts.py              # Gráficos Plotly con la identidad visual
│   ├── report_generator.py    # Informe PDF con datos reales
│   ├── usgs_client.py         # Consulta opcional al catálogo USGS
│   ├── training_service.py    # Reentrenamiento desde la interfaz
│   └── ui.py                  # Componentes HTML/CSS reutilizables
├── pages/
│   ├── _shared.py             # Caché, estado y exportaciones
│   ├── landing.py             # Dashboard
│   ├── analisis_knn.py        # Algoritmo y comparación de K
│   ├── simulador.py           # Simulador conectado al modelo
│   ├── mapa_eventos.py        # Mapa interactivo
│   ├── dataset.py             # Exploración del dataset
│   ├── evaluacion.py          # Métricas del modelo
│   ├── reportes.py            # PDF y exportaciones
│   └── acerca.py              # Acerca del proyecto
├── assets/
│   ├── styles.css             # Identidad visual (glassmorphism)
│   └── fonts/                 # Font Awesome empaquetado (uso offline)
├── reports/                   # Informes y exportaciones generadas
└── tests/
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_prediction.py
```

---

## 4. Requisitos

- Python 3.11 o superior.
- ~300 MB de espacio libre (dependencias).
- Navegador moderno.
- Internet **solo** para cargar las teselas del mapa. El modelo, el dataset y la
  iconografía funcionan sin conexión una vez instalado.

---

## 5. Instalación

```powershell
# 1. Crear y activar un entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 6. Ejecución

```powershell
streamlit run app.py
```

Abrir <http://localhost:8501>.

Si el modelo no existe todavía, la interfaz lo indica y permite generarlo ejecutando el
entrenamiento (abajo) o el botón **Reentrenar modelo** de la sección *Análisis KNN*.

---

## 7. Entrenamiento del modelo

```powershell
python -m src.train_model
```

Este comando:

1. Genera el dataset académico simulado (600 registros / 280 eventos) y lo guarda en
   `data/sismos_peru.csv`.
2. Realiza la división 80/20 agrupada por `evento_id`.
3. Compara K = 3, 5, 7 y 9 con `StratifiedGroupKFold` (5 pliegues).
4. Selecciona el mejor K por F1-score macro.
5. Entrena el pipeline final y lo evalúa una sola vez sobre el test reservado.
6. Guarda `models/knn_pipeline.joblib` y `models/model_metadata.json`.

### Resultado obtenido en este entorno

| K | Accuracy (CV) | F1 macro (CV) |
|---|---|---|
| 3 | 71.15 % | 0.6711 |
| **5** | **74.29 %** | **0.7044** ← seleccionado |
| 7 | 72.82 % | 0.6787 |
| 9 | 72.20 % | 0.6660 |

**Evaluación sobre el conjunto de prueba (122 registros, 56 eventos):**

| Métrica | Valor |
|---|---|
| Accuracy | **75.41 %** (92 / 122) |
| Precision macro | 73.86 % |
| Recall macro | 71.49 % |
| F1-score macro | **72.40 %** |

### Criterio de selección de K (documentado)

1. Mayor **F1-score macro** de validación cruzada.
2. Si hay empate a 4 decimales, mayor **recall macro** (es preferible detectar la
   severidad).
3. Si persiste el empate, el **K menor** (modelo más simple).

---

## 8. Fuentes de datos y naturaleza del dataset

### Lo que sí está disponible en fuentes oficiales

- [IGP](https://www.gob.pe/igp) y [CENSIS](https://ultimosismo.igp.gob.pe/): catálogo de
  eventos con fecha, hora, magnitud, profundidad y coordenadas.
- [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/): catálogo
  global (ComCat) con los mismos campos, más atributos de intensidad **estimada**.

### La limitación encontrada

Un catálogo de **magnitudes** no contiene **intensidad Mercalli observada por localidad**.
El campo `mmi` del catálogo USGS corresponde a una estimación (ShakeMap/estaciones) y no a
observaciones sistemáticas de daño; además solo cubre una fracción de eventos modernos.
Tampoco existe un servicio público que entregue pares (evento, localidad) con intensidad
observada de forma completa para ≥ 500 registros.

### La solución adoptada

Se generó un dataset **evento-localidad** de **600 registros en 280 eventos** con un
modelo **probabilístico de atenuación**, no con una regla determinista:

```
I = 3.1 + 1.42·M − 2.10·log10(R) − 0.005·profundidad + ε,   ε ~ N(0, 0.60)
R = sqrt(distancia² + (0.6·profundidad)²) + 5
```

- Magnitudes con distribución **Gutenberg–Richter truncada** (4.5 – 8.6 Mw).
- Profundidades en tres bandas (superficial, intermedia, profunda).
- Distancias epicentrales con cola larga y eventos "cercanos"/"lejanos" para cubrir las
  tres clases.
- Ruido gaussiano que hace que la etiqueta **no sea** función determinista de las entradas.

**Distribución de clases resultante:** Baja 124 · Moderada 341 · Fuerte 135.

Todos los registros están marcados con `tipo_registro = SIMULADO` y
`fuente = "Generador sintético académico SISMO AI PERÚ (modelo de atenuación)"`.
**No se presentan como observaciones reales del IGP ni del USGS.**

> El sistema incluye además `src/usgs_client.py` para consultar USGS ComCat de forma
> opcional. Si se dispone de eventos con `mmi`/`cdi`, es posible construir un dataset mixto;
> en tal caso cada fila declara su origen y la interfaz lo advierte explícitamente.

### Esquema del CSV

| Columna | Descripción |
|---|---|
| `evento_id` | Identificador del evento (agrupa localidades) |
| `fecha` | Fecha del evento (YYYY-MM-DD) |
| `latitud`, `longitud` | Coordenadas de la localidad |
| `localidad`, `departamento` | Ubicación |
| `magnitud`, `escala_magnitud` | Magnitud original y su escala (Mw) |
| `profundidad_km` | Profundidad hipocentral |
| `distancia_km` | Distancia epicentral a la localidad |
| `intensidad_observada` | Intensidad en escala MMI (1–12) |
| `categoria_intensidad` | Baja (I–IV) · Moderada (V–VI) · Fuerte (VII–XII) |
| `tipo_registro` | SIMULADO u otra procedencia |
| `fuente` | Origen documentado del registro |

---

## 9. Funcionamiento del simulador

1. Se elige departamento/localidad (contexto geográfico, **no** entra al modelo como
   variable categórica) y se ajustan magnitud, profundidad, distancia epicentral y K.
2. Se validan los valores contra los rangos admitidos y contra los rangos del entrenamiento.
3. El **Pipeline KNN entrenado** ejecuta `predict()` para la categoría.
4. `predict_proba()` entrega la proporción de votos de los vecinos.
5. `kneighbors()` recupera los K vecinos más cercanos con su distancia normalizada.
6. Se muestran la categoría, las probabilidades, los votos, la tabla de vecinos y la
   posición del escenario en una proyección PCA 2D.

> **No existe ninguna regla `if/else` que decida la categoría sísmica.** La clasificación
> proviene exclusivamente del modelo. Los `if/else` se usan solo para validar formularios y
> mostrar mensajes.

**Interpretación de las probabilidades:** son la distribución de votos de los vecinos
(`predict_proba`), no probabilidades científicas calibradas de ocurrencia de un terremoto.

**Sobre un K distinto al del modelo:** si se elige otro K en el simulador, se reutiliza el
`StandardScaler` ya ajustado en entrenamiento y se construye un `KNeighborsClassifier` con
ese K. No hay reentrenamiento del escalador ni fuga de información.

---

## 10. Interpretación de métricas

| Métrica | Significado en este proyecto |
|---|---|
| **Accuracy** | Porcentaje de registros de prueba clasificados correctamente. El azar con 3 clases es 33.3 %. |
| **Precision macro** | Promedio simple de la precisión de las 3 clases: de lo predicho como X, cuánto era realmente X. |
| **Recall macro** | Promedio simple del recall de las 3 clases: de lo que era X, cuánto se detectó. |
| **F1-score macro** | Media armónica de precision y recall, promediada por clase. **Métrica principal** porque pondera igual a las clases, aunque tengan distinto número de registros. |
| **Matriz de confusión** | Filas = categoría real, columnas = categoría predicha. La diagonal son los aciertos. |

Se espera confusión entre categorías **contiguas** (Baja↔Moderada, Moderada↔Fuerte),
porque la intensidad es un continuo y los cortes I–IV / V–VI / VII son convencionales.

---

## 11. Pruebas

```powershell
python -m pytest tests -q
```

Se ejecutan **54 pruebas** que cubren:

- Generación reproducible del dataset y variabilidad de etiquetas (no determinista).
- Columnas, tipos, rangos, nulos, duplicados y consistencia intensidad ↔ categoría.
- Split agrupado: **cero eventos compartidos** entre train y test.
- Normalización del `StandardScaler` dentro del Pipeline.
- Comparación de K y agrupación de eventos en cada pliegue de la CV.
- Selección de K y criterio de desempate.
- Coherencia del artefacto y de las métricas; matriz de confusión de 3×3.
- `predict` / `predict_proba` / `kneighbors`, votos y orden de distancias.
- Validación de entradas inválidas y errores controlados.
- Generación del mapa (incluido el caso vacío) y del informe PDF.

---

## 12. Limitaciones

- El dataset principal es **simulado**: las conclusiones se limitan a ese escenario
  experimental y no caracterizan la sismicidad real del Perú.
- Las etiquetas provienen de un modelo de atenuación, no de observaciones instrumentales
  ni de daño físico reportado por localidad.
- KNN es sensible a la escala, al desbalance de clases y a registros ruidosos.
- El mapa requiere internet para las teselas cartográficas.
- El modelo estima efectos esperados para un **escenario de entrada**; no anticipa eventos.

---

## 13. Créditos

Proyecto académico **SENATI** — Inteligencia Artificial.
Fuentes de referencia: IGP, CENSIS, USGS ComCat/ShakeMap.
