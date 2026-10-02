# Entrega 1 — Ideas de producto

**Autor:** Juanjo · TFM del Máster en Data Science & IA (Evolve Academy)

---

## Idea 1: Previsión de demanda de motos para concesionarios (seleccionada)

**Problema o necesidad que aborda**
Un concesionario de motos decide cada mes qué y cuánto pedir a las marcas sin saber cuántas motos se van a vender en su zona. Lo habitual es pedir "como el año pasado" y corregir a ojo. El resultado es conocido en el sector: stock parado (dinero inmovilizado y motos que se quedan de un modelo a otro) o roturas de stock en temporada alta, que son ventas que se lleva otro concesionario. Además, el mercado está cambiando muy deprisa: los ciclomotores casi han desaparecido, crece la cilindrada media (A2) y las marcas chinas han pasado de no existir a vender una de cada cuatro motos nuevas en España.

**Por qué puede tener impacto o valor**
La DGT publica cada mes, en abierto, todas las matriculaciones de España con marca, modelo, cilindrada, potencia y provincia. Con esos datos se puede predecir la demanda de los próximos meses en cada provincia y para cada tipo de carnet (AM, A1/125, A2, A), que es justo como se organiza el negocio: el carnet del cliente decide qué moto puede comprar. Una herramienta que traduzca esa previsión a "cuántas motos pedir de cada segmento" ayuda a decidir con datos y no con intuición, y es útil para cualquier concesionario de España, no solo para uno.

**Qué me motiva personalmente**
Soy aficionado a las motos y ya trabajé este dominio en el máster: en el proyecto de SQL diseñé la base de datos de ventas de una red de concesionarios y en el de IA generativa construí un asistente RAG sobre la normativa de motos y carnets. El TFM es la oportunidad de juntar todo en un producto completo, con datos reales y un modelo validado con rigor.

## Idea 2: Asistente de normativa y carnets de moto (RAG)

**Problema o necesidad que aborda**
Las dudas sobre qué moto se puede conducir con cada carnet, la convalidación del B para 125 cc, la limitación de potencia para el A2, la ITV o las etiquetas ambientales generan muchas consultas en autoescuelas y concesionarios. La información está dispersa entre el BOE, la web de la DGT y foros con información desactualizada.

**Por qué puede tener impacto o valor**
Un asistente con recuperación aumentada (RAG) sobre la normativa oficial daría respuestas citando la fuente, reduciendo errores y tiempo de atención.

**Qué me motiva personalmente**
Es la continuación natural de mi proyecto 7. Lo descarto como TFM principal porque el reto de ciencia de datos (medir y validar) es menor que en la idea 1, pero incorporo una parte de él a la app: un asesor de carnet determinista para el vendedor.

## Idea 3: Tasador de motos de segunda mano

**Problema o necesidad que aborda**
Comprar o vender una moto usada es opaco: los precios de los anuncios varían mucho para el mismo modelo y año, y el comprador no sabe si paga de más.

**Por qué puede tener impacto o valor**
Un modelo de regresión del precio en función de marca, modelo, año, kilómetros y provincia daría una referencia objetiva para compradores, vendedores y concesionarios que aceptan motos a cambio.

**Qué me motiva personalmente**
Es un problema que he vivido como comprador. Lo descarto porque no hay datos abiertos de anuncios: habría que hacer *scraping* de portales cuyos términos de uso lo impiden, con un riesgo legal y de continuidad que no existe en la idea 1, donde la fuente es pública y oficial.
