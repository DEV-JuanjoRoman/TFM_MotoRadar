# Entrega 3 — Modelo de datos y capa gold

**Proyecto:** MotoRadar — previsión de la demanda de motos por provincia y carnet
**Autor:** Juanjo

> La estructura descrita es la que existe en el repositorio. Cada capa se regenera con un script del pipeline (`pipeline/fase0` a `fase5`).

---

## 1. Resumen de la idea y de los datos

- **Problema.** El concesionario pide motos sin una previsión de la demanda de su provincia.
- **Solución.** Previsión mensual a 1-3 meses por provincia × segmento de carnet con un modelo global LightGBM, servida en una app que la traduce a pedido de stock.
- **Fuentes:**

| Fuente | Qué aporta |
|---|---|
| DGT, microdatos de matriculaciones | Todas las motos y ciclomotores matriculados en España, mes a mes desde 2015 |
| `holidays` | Festivos nacionales para calcular los días laborables de cada mes |
| `data/external/provincias.csv` | Código DGT → nombre de provincia, comunidad autónoma y coordenadas de la capital (para el mapa) |

---

## 2. Tecnología de almacenamiento

Ficheros planos versionados en Git, sin base de datos. La carga de trabajo es analítica y de solo lectura (entrenar un modelo y pintar gráficos), el volumen agregado es pequeño (< 1 MB) y no hay escrituras concurrentes. Una base de datos relacional añadiría complejidad sin aportar nada; en el proyecto 2 del máster ya usé MySQL para el caso transaccional de ventas de un concesionario, que es otro problema.

| Formato | Dónde | Por qué |
|---|---|---|
| **Texto compacto** (`clave:v1,v2,...`) | `data/raw/*.txt` | Una línea por serie y un valor por mes. Ocupa 125 KB para 11 años de datos, se lee a simple vista y Git muestra los cambios línea a línea cuando se añade un mes |
| **CSV** | `data/processed/`, `data/gold/previsiones.csv`, `results/` | Formato universal para tablas pequeñas |
| **Parquet** | `data/gold/gold_demanda_modelo.parquet` | Tabla de modelado de 72.000 filas y 35 columnas tipadas, comprimida |
| **Formato nativo de LightGBM** | `models/lgbm_final.txt` | El modelo final, legible y portable |

---

## 3. Capas de datos

```
data/
├── dgt_zip/                         [no versionado] ZIP mensuales originales de la DGT
├── raw/                             agregados de la DGT (salida de la fase 0)
│   ├── months.txt                   los 140 meses (201501 ... 202608)
│   ├── dgt_provincia_segmento.txt   208 series provincia|segmento, motos nuevas
│   ├── dgt_nacional_segmento.txt    32 series segmento|nuevo/usado|propulsión|titular
│   ├── dgt_marca_mensual.txt        52 series de marca (51 marcas + OTRAS)
│   ├── dgt_marca_segmento_12m.txt   marca × segmento, últimos 12 meses
│   └── dgt_top_modelos_12m.txt      100 modelos más matriculados, últimos 12 meses
├── external/provincias.csv          diccionario de provincias
├── processed/                       [regenerable] tablas limpias en formato largo
│   ├── matriculaciones_provincia_segmento.csv
│   ├── matriculaciones_nacional_detalle.csv
│   ├── matriculaciones_marca.csv
│   └── calendario.csv
└── gold/
    ├── gold_demanda_modelo.parquet  [regenerable] tabla de modelado
    └── previsiones.csv              previsión final que consume la app
```

| Capa | Contenido |
|---|---|
| **Raw** | Conteos agregados de la DGT tal cual salen de la extracción, sin limpieza adicional. No hay datos por vehículo. |
| **Processed** | Formato largo, tipos correctos (meses como periodos, segmento como categoría ordenada), nombres de provincia y comunidad, calendario. |
| **Gold** | Tabla de modelado con variables y objetivo, y la tabla de previsiones lista para la app. |

---

## 4. Definición de la capa gold

### `gold_demanda_modelo.parquet`

- **Descripción.** Una fila por serie, mes de origen y horizonte: "en el mes *t* sé esto de la serie; ¿cuántas motos habrá en *t + h*?".
- **Granularidad.** Provincia × segmento × mes de origen × horizonte (1, 2, 3).
- **Clave primaria.** (`provincia`, `segmento`, `origen`, `h`).
- **Registros.** 71.760 (208 series × 116 meses de origen, de diciembre de 2016 a julio de 2026, × hasta 3 horizontes).

| Campo | Tipo | Rol | Descripción |
|---|---|---|---|
| `provincia`, `segmento` | texto / categoría | identificador | Serie |
| `origen` | mes | auxiliar | Último mes conocido |
| `mes_objetivo` | mes | auxiliar | Mes que se predice (`origen + h`) |
| `h` | entero 1-3 | variable | Horizonte |
| `segmento_cod` | entero | variable | Segmento codificado (categórica en LightGBM) |
| `provincia_cod` | entero | variable | Provincia codificada (categórica en LightGBM) |
| `log_nivel` | real | variable | log(1 + media de los últimos 12 meses): tamaño de la serie |
| `tendencia_anual` | real | variable | Nivel de los últimos 12 meses frente a los 12 anteriores (log) |
| `media_3m_rel` | real | variable | Media de los 3 últimos meses relativa al nivel |
| `mismo_mes_anio_ant_rel` | real | variable | Valor del mes objetivo un año antes, relativo al nivel |
| `mismo_mes_hace2_rel` | real | variable | Ídem dos años antes |
| `yoy_3m` | real | variable | Ritmo de los 3 últimos meses frente al mismo periodo del año anterior |
| `lag0_rel` … `lag11_rel` | real | variable | Los 12 últimos meses, relativos al nivel |
| `mes_num`, `mes_sin`, `mes_cos` | entero / real | variable | Mes del año objetivo (y su codificación cíclica) |
| `dias_laborables` | entero | variable | Días laborables del mes objetivo (L-V sin festivos nacionales) |
| `dias_lab_anio_ant` | entero | variable | Días laborables del mismo mes del año anterior |
| `origen_covid` | 0/1 | variable | Los 12 meses de historia incluyen el confinamiento |
| `es_covid` | 0/1 | filtro | El mes objetivo es marzo-mayo de 2020 (se excluye del entrenamiento) |
| `y` | entero | objetivo | Motos matriculadas en el mes objetivo |
| `objetivo_rel` | real | objetivo del modelo | log(1 + y) − log_nivel |

**Por qué relativo al nivel.** Barcelona matricula unas 2.000 motos de 125 cc al mes y Soria menos de 20. Si el modelo predijera unidades, solo aprendería de las provincias grandes. Al predecir el logaritmo relativo a la media de la propia serie, un único modelo aprende patrones comunes (estacionalidad, efecto de los días laborables, inercia) de las 208 series a la vez.

### `previsiones.csv`

Una fila por provincia × segmento × mes previsto (624 filas): `prevision`, `inf_80` y `sup_80` (rango del 80 %), `naive_estacional` y `real_anio_anterior` para dar contexto.

---

## 5. Relaciones entre datos

- `matriculaciones_provincia_segmento.provincia` → `provincias.codigo_dgt` (N:1).
- Las tres tablas de matriculaciones (provincia, nacional y marca) son vistas distintas del mismo universo de motos nuevas: **para cada mes, la suma por provincias, la suma por marcas y el total nacional coinciden exactamente**. Es el principal control de calidad (`results/controles_calidad.csv`).
- `calendario.mes` → `mes_objetivo` de la tabla gold (N:1).
- La tabla gold se obtiene de la provincial mediante una transformación de ventana: cada fila mira 24 meses hacia atrás desde su origen.

---

## 6. Diccionario de datos (origen DGT)

| Campo DGT | Posición | Uso en el proyecto | Observaciones |
|---|---|---|---|
| `FEC_MATRICULA` | 1-8 | Mes | DDMMAAAA |
| `MARCA_ITV` | 18-47 | Marca | Se normalizan variantes (S.Y.M → SYM, HARLEY-DAVIDSON → HARLEY DAVIDSON, GAS GAS → GASGAS) |
| `MODELO_ITV` | 48-69 | Modelo | Texto libre de la ficha técnica |
| `COD_TIPO` | 92-93 | Filtro | 50-52 motocicletas, 90 ciclomotor de dos ruedas |
| `COD_PROPULSION` | 94 | Eléctrica / combustión | 2 = eléctrica |
| `CILINDRADA` | 95-99 | Segmento | cc |
| `COD_PROVINCIA_VEH` | 153-154 | Provincia | Domicilio del vehículo (dónde se usa), no la jefatura que lo matricula |
| `IND_NUEVO_USADO` | 179 | Nueva / usada importada | Solo las nuevas entran en el modelo |
| `PERSONA_FISICA_JURIDICA` | 180 | Particular / empresa | Para detectar automatriculaciones |
| `KW` | 228-234 | Segmento | Potencia en kW |

---

## 7. Problemas de calidad (y cómo se tratan)

- **Fichero de ancho fijo sin nombres de columna.** Una posición mal puesta lo estropea todo. → Prueba automática que compara el parser con líneas reales clasificadas a mano.
- **Marcas escritas de varias formas.** → Tabla de normalización.
- **Motos eléctricas con 0 cc.** → El segmento se decide por kW (≤ 11 kW → A1).
- **Confinamiento (marzo-mayo de 2020).** La DGT apenas tramitó matriculaciones: abril de 2020 tuvo menos de 1.500 motos. → Esos meses no se usan como objetivo y se marca la historia que los contiene.
- **Picos de automatriculación** antes de cambios de norma (diciembre de 2016, diciembre de 2024). → Se documentan; el modelo usa pérdida L2 sobre el logaritmo, menos sensible a picos que en unidades.
- **Series con muchos ceros** (ciclomotores en provincias pequeñas). → `log1p` y modelo global.

---

## 8. Decisiones de limpieza

- Solo motos **nuevas**: las usadas importadas (2,5 %) no responden a la demanda del concesionario.
- Provincia del **vehículo**, no de la jefatura de matriculación: refleja dónde vive el comprador.
- No se descartan las motos de empresa: también son ventas del concesionario (renting, flotas de reparto).
- Criterio de registro válido: tipo de vehículo de moto y línea completa (≥ 240 caracteres).

---

## 9. Riesgos del modelo de datos

- **Lo más sólido:** la serie provincial, con 11 años completos y cuadre exacto con el total nacional.
- **Lo más incierto:** los segmentos se infieren de cc y kW de la ficha técnica; una ficha errónea clasificaría mal una moto. El impacto es bajo (son casos aislados frente a 2,3 millones de motos).
- **Si hubiera que simplificar:** pasar a nivel de comunidad autónoma, donde las series son más grandes y estables.
