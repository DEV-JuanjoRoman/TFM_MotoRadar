# Guion del vídeo de defensa — MotoRadar

**Duración objetivo: 10 minutos** (si te piden menos, mira los recortes al final).
Las diapositivas (11) siguen este guion una a una y llevan estas mismas ideas en las notas del orador.
Graba con OBS o Loom: pantalla completa + tu cámara en una esquina. Ten abiertas dos cosas: la presentación y la app (`python -m streamlit run app.py`, o la URL de Streamlit Cloud), con la provincia en **Madrid**.

Consejo: no lo leas palabra por palabra. Léelo dos veces en voz alta antes de grabar y luego habla con tus palabras siguiendo las ideas en negrita.

---

## 0:00 – 0:40 · Diapositiva 1: Portada

> Hola, soy Juanjo y este es mi Trabajo Fin de Máster del Máster en Data Science e IA de Evolve Academy: **MotoRadar**, un sistema que predice cuántas motos se van a matricular en cada provincia de España durante los próximos tres meses, para que un concesionario sepa **cuántas motos pedir**.
>
> Lo he construido con **datos oficiales de la DGT**: todas las motos matriculadas en España desde 2015, más de 2,3 millones.

## 0:40 – 1:40 · Diapositiva 2: El problema

> Un concesionario hace sus pedidos a las marcas con uno a tres meses de antelación. ¿Y cómo decide cuántas motos pide? Normalmente **"como el año pasado"**.
>
> El problema es que el mercado ya no es el del año pasado: en 2025 se vendieron un 71 % más de motos que en 2015 y este año vamos un 21 % por encima. Si pides como el año pasado, te quedas corto en plena temporada, y eso son ventas que se lleva otro. Y si te pasas, tienes dinero parado en el almacén.
>
> Mi pregunta fue: **¿se puede prever la demanda de cada provincia mejor que "como el año pasado", solo con datos públicos?**

## 1:40 – 2:30 · Diapositiva 3: Los datos

> La DGT publica cada mes un fichero con todos los vehículos matriculados en España, uno por línea: unas 130.000 líneas al mes. He descargado 140 meses, de enero de 2015 a agosto de 2026.
>
> Me quedo con motos y ciclomotores y clasifico cada moto por el **carnet que exige**, porque así funciona el negocio: el carnet del cliente decide qué moto puede comprar. **AM** para ciclomotores, **A1** para 125, **A2** hasta 35 kW y **A** para el resto. Lo hago con la cilindrada y la potencia de la ficha técnica.
>
> El resultado son **208 series**: 52 provincias por 4 segmentos. Y un control de calidad que me gusta mucho: para cada mes, la suma por provincias, la suma por marcas y el total nacional **cuadran exactamente**.

## 2:30 – 3:40 · Diapositivas 4 y 5: Lo que dicen los datos

> Antes de modelar, miré el mercado, y hay tres cosas que me parecen muy relevantes para un concesionario:
>
> Primero, **cambia el mix**: los ciclomotores han pasado del 11 % al 5 % del mercado, y el A2 sube. El cliente sube de cilindrada.
>
> Segundo, **la temporada manda**: julio es el pico en casi todos los segmentos y mayo en las motos grandes.
>
> Y tercero, la gran noticia (diapositiva 5): **las marcas chinas**. Zontes, Voge, QJMotor, CFMoto… En 2020 eran el 0,6 % del mercado. En 2026 son **una de cada cuatro motos nuevas**. Zontes ya es la tercera marca de España, solo por detrás de Honda y Yamaha.
>
> Esto es importante para el modelo: el mercado está cambiando más rápido de lo que su propia historia enseña.

## 3:40 – 5:00 · Diapositiva 6: Cómo predigo

> Comparé cuatro modelos, de menos a más sofisticado:
>
> - El **"como el año pasado"**, que es lo que hace hoy el concesionario.
> - El año pasado **corregido por la tendencia** reciente, lo que haría alguien con un Excel.
> - **Holt-Winters**, el clásico de series temporales, ajustado provincia a provincia.
> - Y **LightGBM global**: un único modelo que aprende de las 208 series a la vez.
>
> La clave del modelo global es que no predice unidades, sino **cuánto se desvía cada mes de lo normal en esa serie**. Así Soria, que vende 10 motos al mes, aprende de Barcelona, que vende 2.000, porque las dos tienen la misma temporada.
>
> Y para evaluarlo hice una **validación de origen móvil**: me pongo en septiembre de 2024, entreno solo con lo que se sabía entonces, predigo los tres meses siguientes, y lo comparo con lo que pasó. Luego avanzo un mes y repito. **23 veces**, casi 14.000 predicciones que el modelo nunca había visto. Es exactamente como funcionaría en la realidad.

## 5:00 – 6:30 · Diapositivas 7 y 8: Resultados

> La métrica es el **WAPE**: de cada 100 motos que se matriculan, en cuántas se equivoca la previsión.
>
> "Como el año pasado" se equivoca en **21 de cada 100**. Holt-Winters en 18,5. **MotoRadar en 15.** Es un **29 % menos de error** que el método actual, y a nivel nacional el error baja al 10 %.
>
> Y no es suerte: con un bootstrap por meses, la mejora está entre el 22 y el 35 % con un 95 % de confianza. Gana en **22 de los 23 meses** al método actual, y también supera a Holt-Winters de forma significativa.
>
> (Diapositiva 8) Aquí se ve mes a mes: la línea negra es la realidad, la azul lo que predijo MotoRadar un mes antes y la gris discontinua "como el año pasado". La azul va pegada a la negra.
>
> Y como ningún modelo es perfecto, doy un **rango**: un mínimo y un máximo probables. Lo calibré para que acierte el 80 % de las veces y, en datos que no había visto, **acierta el 81 %**.

## 6:30 – 7:10 · Diapositiva 9: Limitaciones (sé honesto)

> Hay cosas que el modelo no hace bien, y es importante decirlas:
>
> - En **diciembre de 2024** hubo un pico de 23.600 motos: los concesionarios matricularon a su nombre antes de que entrara la norma Euro 5+. Eso no se puede prever con el histórico.
> - Como el mercado acelera, el modelo se queda **un 5 % corto** de media. Probé a corregirlo y empeoraba el error, así que lo dejé documentado.
> - Y en provincias pequeñas y en ciclomotores el error es mayor.

## 7:10 – 9:20 · DEMO de la app (cambia a la app)

> Pero un modelo en un notebook no le sirve a un concesionario. Así que lo he convertido en una **aplicación**.

1. **Previsión** (Madrid): *"Elijo mi provincia, Madrid. Arriba veo cuántas motos se van a matricular en septiembre, octubre y noviembre, con su rango y la comparación con el año pasado. Abajo, la historia y la previsión de cada segmento."* → Pulsa en **125 cc** para enseñar la banda del rango.
2. **Pedido**: *"Aquí está lo más útil. Pongo mis ventas del último año y mi stock. MotoRadar calcula mi cuota en Madrid, la aplica a la previsión y me dice cuántas motos pedir de cada segmento."* → Cambia el escenario a **Prudente** y a **Optimista** para que se vea que cambia el pedido.
3. **Mercado**: *"Aquí veo cómo está el mercado: las marcas chinas en naranja, el mapa de provincias…"* → Cambia el ranking a **Media (A2)**.
4. **Fiabilidad**: *"Y para que el gerente se fíe, la app le enseña cuánto se equivoca el modelo y lo que predijo en su provincia frente a lo que pasó."*
5. **Carnets**: *"Por último, un asesor para el vendedor. Un cliente de 30 años con el carnet B desde hace 5 años: puede llevar 125. Si marco que tiene el A2…"* → marca A2 para ver cómo cambia.

## 9:20 – 10:00 · Diapositiva 10: Conclusiones

> Para terminar:
>
> - Con datos públicos de la DGT se puede prever la demanda de motos de cada provincia **un 29 % mejor** que como se hace hoy.
> - Un modelo global que aprende de todas las provincias funciona mejor que uno por provincia.
> - Y lo más importante: no se queda en un notebook. Está en una **app que un concesionario puede usar mañana** para decidir su pedido.
>
> Como siguientes pasos, me gustaría conectarla con la base de datos de ventas del concesionario, que ya diseñé en el proyecto de SQL, y añadir variables como los cambios de normativa.
>
> Todo el código, los datos y la memoria están en GitHub. Muchas gracias.

---

## Si te piden menos tiempo

- **7 minutos:** quita la diapositiva 5 (marcas chinas: menciónalo en una frase en la 4), resume la 9 en una frase y deja la demo en Previsión + Pedido.
- **5 minutos:** portada (20 s), problema (40 s), datos (30 s), resultados (1 min 30), demo Previsión + Pedido (1 min 30), conclusiones (30 s).

## Antes de grabar (lista rápida)

- [ ] La app abre sin errores y está en Madrid.
- [ ] La presentación está en modo presentación.
- [ ] Notificaciones del ordenador desactivadas.
- [ ] Micrófono probado (graba 10 segundos y escúchalo).
- [ ] Agua cerca. Si te equivocas, sigue: un vídeo natural queda mejor que uno perfecto.
