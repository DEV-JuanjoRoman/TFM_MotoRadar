# Entrega 4 — Diseño del análisis y estrategia de modelado

**Proyecto:** MotoRadar — previsión de la demanda de motos por provincia y carnet
**Autor:** Juanjo

Cada decisión se presenta con su justificación y, debajo, el **resultado obtenido** al ejecutar el pipeline (`python pipeline/ejecutar_todo.py`). Todas las cifras salen de los ficheros de `results/`.

---

## 1. Problema que se busca resolver

### 1.1 Qué ocurre hoy

El concesionario hace sus pedidos a las marcas con 1-3 meses de antelación (plazos de fábrica y transporte) y estima la demanda a ojo, normalmente "como el mismo mes del año pasado". En un mercado que ha crecido un 71 % entre 2015 y 2025 y un 21 % en lo que va de 2026, ese método se queda sistemáticamente corto: en la validación, el "como el año pasado" infraestima la demanda un 13 %.

### 1.2 Quién usa el resultado y para qué decisión

El gerente o jefe de compras de un concesionario multimarca. Decisión: **cuántas motos de cada segmento (AM, 125, A2, A) pedir para los próximos 1-3 meses**.

### 1.3 Resultado concreto

Para cada provincia y segmento, las matriculaciones previstas en los 3 meses siguientes con un rango probable del 80 %, y una conversión a unidades a pedir según la cuota y el stock del concesionario.

---

## 2. Análisis de datos

### 2.1 Preguntas

1. ¿Cómo evoluciona el mercado y qué segmentos ganan o pierden peso?
2. ¿Qué estacionalidad tiene cada segmento?
3. ¿Qué marcas están cambiando el mercado?
4. ¿Se puede predecir la demanda provincial mejor que con los métodos actuales? ¿Cuánto mejor y con qué seguridad?

### 2.2 Resultados del análisis exploratorio (`pipeline/fase2_eda.py`)

| Hallazgo | Dato |
|---|---|
| Crecimiento del mercado | 147.966 motos nuevas en 2015 → 253.577 en 2025 (+71 %) |
| 2026 | 201.216 de enero a agosto, +20,9 % frente a 2025 |
| Ciclomotores | Del 10,8 % del mercado en 2015 al 4,7 % en 2026 |
| A2 | Del 20,0 % al 25,1 %: el comprador sube de cilindrada |
| Marcas chinas nuevas (Zontes, Voge, QJMotor, CFMoto, Kove, Cyclone) | Del 0,6 % en 2020 al 24,8 % en 2026. Zontes es ya la 3.ª marca de España |
| Estacionalidad | Pico en julio en AM, A1 y A2; en mayo en el segmento A |
| Concentración | Barcelona + Madrid = 27 % del mercado; las 10 primeras provincias, 64 % |
| Eléctricas | Máximo del 8,5 % en 2020, coincidiendo con el auge del motosharing; 3,0 % en 2026 |
| Automatriculaciones | Picos en diciembre de 2024 (8.965 motos a nombre de empresa, antes de la Euro 5+) y diciembre de 2016 (antes de la Euro 4) |

Figuras: `reports/figures/01` a `07`.

### 2.3 Hipótesis

- **H1.** Un modelo global entrenado con las 208 series predice mejor que los métodos por serie, porque las provincias pequeñas tienen poco historial propio pero comparten patrones con las grandes. **Se confirma** (§7).
- **H2.** La estacionalidad y los días laborables explican buena parte de la variación mensual. **Se confirma**: las variables de calendario (mes del año y días laborables) suman el 36 % de la importancia del modelo.
- **H3.** El error crece con el horizonte. **Se confirma, pero poco**: 14,6 % a 1 mes y 15,8 % a 3 meses.

---

## 3. Modelos planteados

### 3.1 Tipo de tarea

Regresión de series temporales de conteo, multihorizonte (1-3 meses), con 208 series jerárquicas (provincia × segmento) que suman el total nacional.

### 3.2 Alternativas

| Modelo | Por qué se incluye |
|---|---|
| **Naive estacional** (mismo mes del año anterior) | Es lo que hace hoy el concesionario. Es la referencia de negocio. |
| **Naive estacional con tendencia** | El mismo mes del año anterior corregido por cómo van los 3 últimos meses frente a un año antes. Es lo que haría un analista con una hoja de cálculo. |
| **Holt-Winters (ETS)** amortiguado sobre log1p | Modelo estadístico clásico de series temporales, ajustado serie a serie. |
| **LightGBM global** | Un único modelo de gradient boosting para todas las series, con variables normalizadas por el nivel de cada serie. |

### 3.3 Justificación del recorte

Se descartan **Prophet** (se ajusta serie a serie, es lento para 208 series × 23 reentrenamientos y para series mensuales cortas cumple el mismo papel que ETS) y las **redes recurrentes (LSTM)**: con 140 meses por serie no hay datos suficientes para que superen a un gradient boosting bien planteado, y su coste de mantenimiento no compensa para un concesionario.

---

## 4. Datos de entrada

- **Tabla gold:** `data/gold/gold_demanda_modelo.parquet` (ver Entrega 3).
- **Variables (27):** horizonte, segmento, provincia, tamaño de la serie, tendencia anual, 12 retardos relativos, mismo mes del año anterior y de hace dos años, ritmo reciente, mes del año, días laborables y marca de historia con COVID.
- **Provincia como variable categórica:** se añadió tras comprobarlo en la mitad de los orígenes de validación (el WAPE bajó del 15,6 % al 15,3 %). Captura particularidades locales, como el pico de primavera de las 125 cc en Baleares (probablemente flotas de alquiler para la temporada turística).
- **Descartadas:** la proporción de motos de empresa o de eléctricas del mes objetivo, porque no se conocen en el momento de predecir.
- **Información disponible en el momento de la predicción:** todas las variables se calculan con datos hasta el mes de origen. Hay una prueba automática que lo garantiza (`test_sin_fuga_de_datos`): si se cambian los datos posteriores al origen, las variables no cambian.

---

## 5. Salida y forma de consumo

- **Salida:** `data/gold/previsiones.csv`, con previsión puntual y rango del 80 % por provincia, segmento y mes.
- **Consumo:** la app de Streamlit la muestra por provincia y la convierte en pedido. No necesita LightGBM instalado: lee la tabla precalculada, lo que hace la app ligera y fácil de desplegar.
- **Incertidumbre:** el rango del 80 % se calcula con los cuantiles del error relativo observado en la validación, por segmento y horizonte.

---

## 6. Diseño y selección del modelo

### 6.1 Proceso

1. Construir la tabla de modelado (fase 1).
2. Entrenar y evaluar los cuatro modelos con validación de origen móvil (fase 3).
3. Analizar los errores y calibrar los intervalos (fase 4).
4. Reentrenar el ganador con todos los datos y generar la previsión (fase 5).

### 6.2 Preprocesamiento

- Objetivo: `log(1 + y) − log(1 + nivel_12m)`; la predicción se deshace con `exp`.
- Se excluyen como objetivo los meses del confinamiento.
- LightGBM: 900 árboles, tasa de aprendizaje 0,03, 31 hojas, submuestreo del 80 % de filas y variables, pérdida L2. Se probaron L1 y Huber en la mitad de los orígenes y dieron el mismo error que L2 (15,6 %).

### 6.3 Criterio y regla de decisión

**Métrica principal: WAPE** (error absoluto total / matriculaciones totales). Se lee como "de cada 100 motos, la previsión se equivoca en X". Se elige porque pondera cada provincia por su peso en el mercado (como le importa al negocio) y, a diferencia del MAPE, no explota en series con meses de 0 o 1 motos. Métricas secundarias: WAPE del total nacional, MAE y sesgo.

**Regla:** el modelo elegido debe mejorar al mejor baseline con un intervalo de confianza del 95 % que no incluya el 0, y ganar en la mayoría de los meses de validación.

---

## 7. Validación y evaluación

### 7.1 Separación de datos

**Validación de origen móvil** con 23 orígenes, de septiembre de 2024 a julio de 2026. En cada origen se entrena solo con los ejemplos cuyo mes objetivo ya era conocido y se predicen los 3 meses siguientes. Son 13.728 predicciones fuera de muestra. No hay partición aleatoria: en series temporales mezclaría el futuro con el pasado.

### 7.2 Resultado obtenido

| Modelo | WAPE provincia × segmento | WAPE total nacional | Sesgo |
|---|---|---|---|
| Naive estacional (año anterior) | 21,4 % | 16,7 % | +12,8 % |
| Naive estacional con tendencia | 22,7 % | 16,2 % | −0,5 % |
| Holt-Winters (ETS) | 18,5 % | 13,7 % | +11,3 % |
| **LightGBM global** | **15,2 %** | **10,3 %** | +5,0 % |

**Mejora de LightGBM (IC 95 % por bootstrap de meses de origen):**

| Frente a | Mejora del WAPE | IC 95 % | Meses en que gana |
|---|---|---|---|
| Naive estacional | 28,9 % | 22,5 % – 35,3 % | 22 de 23 |
| Naive con tendencia | 33,0 % | 26,4 % – 39,0 % | 22 de 23 |
| Holt-Winters | 17,8 % | 9,4 % – 25,9 % | 15 de 23 |

Se cumple la regla de decisión: LightGBM mejora a todos los baselines con intervalos que no incluyen el 0.

### 7.3 Análisis de errores (`pipeline/fase4_analisis_errores.py`)

- **Por horizonte:** 14,6 % (1 mes), 15,3 % (2), 15,8 % (3).
- **Por segmento:** AM 26,1 %, A1 14,4 %, A2 14,7 %, A 14,9 %. Los ciclomotores son el segmento más difícil: son series pequeñas, en caída y con matriculaciones de flotas de reparto.
- **Por tamaño de provincia:** del 13,6 % en las 13 más grandes al 29,2 % en las 13 más pequeñas. LightGBM gana en los cuatro grupos.
- **Dónde se concentra el error:** Barcelona y Madrid en 125 cc, las series más grandes, suman el 9 % del error absoluto total, aunque su WAPE es bajo (10-13 %).
- **Fallo más visible:** diciembre de 2024. El mercado matriculó 23.600 motos, un pico de automatriculaciones antes de la Euro 5+, y ningún modelo lo anticipó.
- **Sesgo:** LightGBM se queda corto un 5,0 % de media en un mercado que acelera. Se probó corregirlo multiplicando por el error de los 12 meses anteriores: el sesgo baja a −0,5 %, pero el WAPE empeora al 16,3 %. **Se descarta.**

### 7.4 Intervalos de predicción

Se calibran con la primera mitad de la validación y se comprueban en la segunda: el rango del 80 % contiene el valor real el **81 %** de las veces (AM 81 %, A1 82 %, A2 86 %, A 77 %). Están bien calibrados.

### 7.5 Criterio de aceptación

Mejorar al método actual del concesionario al menos un 15 % con significación estadística. **Se alcanza (28,9 %).** Si no se hubiera alcanzado, el plan era agregar a nivel de comunidad autónoma.

---

## 8. Riesgos y alternativas

- **¿Matricular es vender?** Casi siempre, pero las automatriculaciones distorsionan algunos meses. Se vigila la proporción de motos a nombre de empresa (10 % en 2026).
- **Fuga de datos:** controlada con la construcción por origen y con una prueba automática.
- **Cambios estructurales:** la llegada de nuevas marcas acelera el mercado más deprisa de lo que el histórico enseña; por eso el sesgo positivo. Mitigación: reentrenar cada mes (el pipeline tarda unos minutos) y mostrar siempre el rango.
- **Series pequeñas:** error alto en provincias pequeñas y en AM; la app muestra el rango para que el usuario vea la incertidumbre.

---

## 9. Cambios respecto a entregas anteriores

- Se añadió el baseline "naive con tendencia" para que la comparación fuera más exigente.
- Se descartó la corrección de sesgo tras probarla (§7.3).
- Se añadió la provincia como variable categórica tras comprobar que mejoraba (§4).
- Se incorporó a la app el asesor de carnet, que reutiliza la idea 2 de la Entrega 1.
