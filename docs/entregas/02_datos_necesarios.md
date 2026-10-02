# Entrega 2 — Selección de idea y análisis de los datos necesarios

**Proyecto:** MotoRadar — previsión de la demanda de motos por provincia y tipo de carnet para planificar el stock de un concesionario
**Autor:** Juanjo

---

## 1. Idea seleccionada

**Problema que resuelve.** El concesionario de motos planifica sus pedidos a las marcas con meses de antelación y sin una previsión de cuántas motos se venderán en su provincia. Decide por intuición o copiando el año anterior, en un mercado que crece un 15-20 % al año y cuya composición cambia rápido (más A2, menos ciclomotores, irrupción de marcas chinas). Equivocarse por arriba inmoviliza dinero en stock; equivocarse por abajo supone perder ventas en primavera y verano, cuando se concentra la demanda.

**Solución planteada.** Un modelo de predicción de las matriculaciones de motos nuevas a 1, 2 y 3 meses vista para cada combinación de provincia (52) y segmento de carnet (AM, A1, A2, A), entrenado con los microdatos oficiales de la DGT desde 2015. Se compara de forma rigurosa con los métodos que usaría un profesional (el mismo mes del año anterior, con y sin tendencia, y Holt-Winters) mediante validación de origen móvil.

**MVP.** Una aplicación web en Streamlit con cinco vistas: *Previsión* (demanda prevista en la provincia con su rango probable), *Pedido* (convierte la previsión en unidades a pedir según la cuota y el stock del concesionario), *Mercado* (tendencias de segmentos, marcas y modelos), *Fiabilidad* (cuánto se equivoca el modelo) y *Carnets* (asesor para el vendedor sobre qué motos puede llevar cada cliente).

---

## 2. Datos necesarios

**Variables.**

- **Variable objetivo:** número de motos nuevas matriculadas por mes, provincia y segmento.
- **Para construir el segmento:** tipo de vehículo, cilindrada (cc) y potencia (kW) de la ficha técnica. El carnet se asigna así: ciclomotor → AM; hasta 125 cc y 11 kW → A1 (o B con 3 años); hasta 35 kW → A2; más de 35 kW → A.
- **Contexto:** marca y modelo (cuotas, ranking), propulsión (eléctrica o de combustión), si la moto es nueva o usada importada, y si el titular es una persona o una empresa (renting, flotas y automatriculaciones).
- **Calendario:** mes, días laborables (festivos nacionales) y marca de los meses del confinamiento de 2020.
- **Variables derivadas:** retardos de 0 a 11 meses, el mismo mes del año anterior y de hace dos años, medias recientes y tendencia anual, todo normalizado por el tamaño de cada serie.

**Granularidad.** Mensual y a nivel de provincia × segmento (208 series). Los microdatos son por vehículo, pero se agregan porque la decisión del concesionario es mensual y provincial, y porque a nivel de vehículo el dato no aporta y sí aumenta el riesgo de privacidad.

**Profundidad histórica.** Enero de 2015 a agosto de 2026: 140 meses, más de 11 años, con varios ciclos estacionales completos, el confinamiento de 2020 y dos cambios de norma Euro.

**Volumen.** Unos 18 millones de registros de vehículos en origen (todas las categorías), de los que 2,36 millones son motos y ciclomotores (2,31 millones nuevos). Tras agregar: 29.120 observaciones provincia × segmento × mes y una tabla de modelado de 71.760 filas.

**Imprescindibles frente a deseables.** Imprescindibles: los microdatos de matriculaciones de la DGT. Deseables: el calendario de festivos (mejora la previsión de meses como agosto o abril) y, en el futuro, las ventas reales del concesionario para calcular su cuota con exactitud.

---

## 3. Fuentes de datos

| Fuente | Qué aporta | Acceso | Formato | Histórico |
|---|---|---|---|---|
| **DGT — Microdatos de matriculaciones (fichero mensual)** | Una línea por vehículo matriculado: fecha, marca, modelo, tipo, cc, kW, propulsión, provincia, nuevo/usado, tipo de titular | Abierta, gratuita, sin registro | ZIP con texto de ancho fijo (~12 MB/mes) | Desde diciembre de 2014 |
| **Librería `holidays`** | Festivos nacionales de España | Abierta (pip) | Python | Todos los años |
| **Ventas del concesionario** (opcional) | Ventas propias de 12 meses por segmento | Las introduce el usuario en la app | Formulario | Últimos 12 meses |

Enlace de la fuente principal: `https://www.dgt.es/menusecundario/dgt-en-cifras/matraba-listados/matriculaciones-automoviles-mensual.html`. Cada mes se descarga de `https://www.dgt.es/microdatos/salida/AAAA/M/vehiculos/matriculaciones/export_mensual_mat_AAAAMM.zip`.

**Estabilidad.** La DGT publica el fichero cada mes desde 2014 con el mismo diseño de registro, lo que se ha comprobado leyendo meses de 2019 y 2026 con las mismas posiciones.

**Riesgos detectados.**

- Matricular no es vender: un concesionario puede matricular motos a su nombre para cumplir objetivos (automatriculación). Se vigila con la variable de titular (empresa).
- Cambios de normativa provocan picos artificiales (diciembre de 2016 con la Euro 4 y diciembre de 2024 con la Euro 5+).
- El fichero es de ancho fijo y sin cabecera de columnas: un error de posiciones clasificaría mal las motos. Se mitiga con pruebas automáticas sobre líneas reales.

---

## 4. Privacidad y protección de datos

- **Datos personales.** El fichero no contiene nombres ni DNI. El número de bastidor llega enmascarado y la localidad del vehículo es el único dato potencialmente sensible.
- **Medidas.** No se guarda ningún dato a nivel de vehículo: el script de descarga agrega en memoria por mes, provincia y segmento, y solo escribe conteos. Las series más pequeñas (p. ej. ciclomotores en Soria) son conteos de una o dos motos por mes sin ninguna información que permita identificar a nadie.
- **Uso académico.** Los datos son públicos y se reutilizan según las condiciones de reutilización de la información del sector público (Ley 37/2007), citando a la DGT como fuente.

---

## 5. Viabilidad

- **¿Se pueden obtener los datos?** Sí, y ya se han obtenido: 140 meses descargados y agregados, con controles de coherencia (la suma por provincias, por marcas y el total nacional cuadran exactamente).
- **¿Calidad e histórico suficientes?** Sí: 11 años mensuales permiten aprender la estacionalidad y validar el modelo en 23 meses distintos.
- **¿Realizable en el curso?** Sí: el volumen agregado es pequeño (cabe en un portátil) y el modelado es reproducible en minutos.
- **Parte más arriesgada.** Las provincias pequeñas tienen pocas motos al mes y mucho ruido. Se aborda con un modelo global que aprende de todas las series a la vez.
- **Alternativa si la fuente fallara.** Las estadísticas mensuales que publica ANESDOR (asociación del sector) dan los totales del mercado por segmento, con menos detalle; servirían al menos para el modelo nacional.
