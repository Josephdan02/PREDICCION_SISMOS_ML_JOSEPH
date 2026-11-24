# Procedencia y metodología del dataset

## Resumen

| Campo | Valor |
|---|---|
| Archivo activo | `data/sismos_peru.csv` |
| Registros | 600 evento-localidad |
| Eventos únicos | 280 |
| Departamentos | 25 |
| Tipo de registro | **100 % SIMULADO** |
| Fuente declarada | Generador sintético académico SISMO AI PERÚ |
| Generador | `src/preprocessing.py::generate_synthetic_dataset` |
| Versión del generador | `config.DATASET_VERSION` |
| Semilla | `config.RANDOM_STATE = 42` |
| Regenerar | `python -m src.train_model` |

## ¿Por qué es simulado?

Se investigaron las fuentes oficiales:

- **IGP / CENSIS** (<https://www.gob.pe/igp>, <https://ultimosismo.igp.gob.pe/>): publican
  catálogos de eventos con fecha, magnitud, profundidad y coordenadas, e intensidades
  reportadas de forma puntual, no como pares sistemáticos (evento, localidad).
- **USGS ComCat** (<https://earthquake.usgs.gov/earthquakes/search/>): ofrece el mismo tipo
  de campos más `mmi` (intensidad **estimada** por ShakeMap/estaciones) y `cdi`
  (intensidad reportada por la comunidad). Ninguno de los dos constituye una intensidad
  Mercalli **observada por localidad** con cobertura suficiente.

**Conclusión:** no es posible obtener ≥ 500 registros evento-localidad con intensidad
observada verificable desde servicios públicos. Por lo tanto, y siguiendo el lineamiento
académico del proyecto, el dataset es **simulado** y se identifica como tal en cada fila.

## Metodología de simulación

La etiqueta **no** se obtiene con una regla determinista. Se emplea un modelo probabilístico
de atenuación:

```
I = A + B·M − C·log10(R) − D·H + ε
R = sqrt(distancia² + (0.6·H)²) + 5
ε ~ N(0, σ)
```

| Parámetro | Valor | Significado |
|---|---|---|
| A | 3.1 | Constante de atenuación |
| B | 1.42 | Ganancia por unidad de magnitud |
| C | 2.10 | Atenuación logarítmica con la distancia |
| D | 0.005 | Atenuación por profundidad |
| σ | 0.60 | Dispersión (ruido) en unidades MMI |

Componentes estocásticos:

1. **Magnitud:** distribución Gutenberg–Richter truncada en [4.5, 8.6] Mw.
2. **Profundidad:** mezcla de bandas — superficial 5–35 km (62 %), intermedia 35–160 km
   (30 %), profunda 160–650 km (8 %).
3. **Distancias:** Gamma con cola larga; subconjuntos de eventos "cercanos" (20–110 km) y
   "lejanos" (150–420 km) para cubrir las tres clases.
4. **Ruido gaussiano:** genera solapamiento entre clases, de modo que el modelo debe aprender
   una frontera y no una fórmula.

La intensidad resultante se redondea a un entero 1–12 (MMI) y se agrupa en:

- **Baja:** I–IV
- **Moderada:** V–VI
- **Fuerte:** VII–XII

## Advertencia

Estos datos **no** son observaciones reales del IGP, CENSIS ni USGS. No deben usarse para
caracterizar la sismicidad del Perú ni para decisiones de seguridad. Su único propósito es
demostrar el funcionamiento técnico del algoritmo KNN en un escenario experimental
controlado.
