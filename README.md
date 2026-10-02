# MotoRadar

**Juanjo** · TFM del Máster en Data Science & Inteligencia Artificial (Evolve Academy)

Aplicación web que ayuda a un concesionario de motos a decidir **cuántas motos pedir de cada tipo** para los próximos meses. Predice las matriculaciones de motos nuevas de los 3 meses siguientes en cada provincia y para cada tipo de carnet (ciclomotor AM, 125 cc A1, media cilindrada A2 y alta cilindrada A), con un rango probable, y convierte esa previsión en un pedido según la cuota y el stock del concesionario.

Forma parte del TFM *MotoRadar: previsión de la demanda de motos por provincia y carnet con datos abiertos de la DGT*.

## Resultado principal

Validado durante 23 meses (septiembre de 2024 a julio de 2026) reentrenando cada mes solo con la información disponible en ese momento:

| Modelo | Error (WAPE) provincia × segmento | Error total nacional |
|---|---|---|
| "Como el año pasado" (método actual) | 21,4 % | 16,7 % |
| Año pasado + tendencia | 22,7 % | 16,2 % |
| Holt-Winters | 18,5 % | 13,7 % |
| **MotoRadar (LightGBM global)** | **15,2 %** | **10,3 %** |

MotoRadar reduce el error del método actual un **29 %** (IC 95 %: 22-35 %) y gana en 22 de los 23 meses. El rango del 80 % acierta el 81 % de las veces fuera de muestra.

## Qué hace la app

- **Previsión.** Motos nuevas previstas en la provincia para los próximos 3 meses, por segmento, con rango del 80 % y comparación con el año anterior.
- **Pedido.** Introduces tus ventas de 12 meses y tu stock; calcula tu cuota y sugiere cuántas motos pedir de cada segmento en escenario prudente, central u optimista.
- **Mercado.** Evolución del mercado, segmentos, ranking de marcas (con la irrupción de las chinas), modelos más vendidos y mapa provincial.
- **Fiabilidad.** Error de cada modelo, error por segmento y, para tu provincia, lo que pasó frente a lo que se predijo.
- **Carnets.** Asesor para el vendedor: qué motos puede llevar un cliente con sus permisos y qué carnet le falta para subir de segmento.

## Ejecutar en local

Python 3.12 o superior.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La app solo lee tablas precalculadas (`data/gold/previsiones.csv`, `results/`, `models/metadata.json`), así que no necesita LightGBM y se despliega tal cual en Streamlit Community Cloud (punto de entrada: `app.py`).

## Reproducir el pipeline

```bash
python -m pip install -r requirements-pipeline.txt
python pipeline/ejecutar_todo.py            # fases 1 a 5 + pruebas, ~8 min
python pipeline/ejecutar_todo.py --rapido   # sin Holt-Winters, ~2 min
```

| Script | Qué hace |
|---|---|
| `pipeline/fase0_descarga_dgt.py` | Descarga los microdatos mensuales de la DGT, filtra motos, asigna el carnet y agrega (`--desde 2015-01 --hasta 2026-08`) |
| `pipeline/fase1_preparacion.py` | Controles de calidad, tablas limpias y capa gold |
| `pipeline/fase2_eda.py` | Análisis exploratorio y figuras |
| `pipeline/fase3_backtest.py` | Validación de origen móvil de los 4 modelos |
| `pipeline/fase4_analisis_errores.py` | Errores por segmento, provincia y horizonte; intervalos; experimento de sesgo |
| `pipeline/fase5_modelo_final.py` | Modelo final y previsión que consume la app |

El cuaderno [`notebooks/MotoRadar_TFM.ipynb`](notebooks/MotoRadar_TFM.ipynb) recorre el proyecto con todas las salidas.

## Pruebas

```bash
python tests/test_motoradar.py
```

10 pruebas: integridad y cuadre de los datos, que el lector del fichero de la DGT clasifica igual que la extracción original sobre líneas reales, que las variables no usan información del futuro, métricas, cálculo del pedido y reglas de carnets.

## Datos

| Fichero | Contenido | Fuente |
|---|---|---|
| `data/raw/dgt_provincia_segmento.txt` | Motos nuevas por mes (2015-01 a 2026-08), provincia (52) y segmento (4) | DGT, microdatos de matriculaciones |
| `data/raw/dgt_nacional_segmento.txt` | Total nacional por segmento, nueva/usada, eléctrica/combustión y particular/empresa | DGT |
| `data/raw/dgt_marca_mensual.txt` | Motos nuevas por marca y mes | DGT |
| `data/raw/dgt_marca_segmento_12m.txt`, `dgt_top_modelos_12m.txt` | Marcas por segmento y 100 modelos más vendidos (últimos 12 meses) | DGT |
| `data/external/provincias.csv` | Nombre, comunidad y coordenadas de la capital de cada provincia | Elaboración propia |

Los agregados ocupan 125 KB y no contienen ningún dato por vehículo. Fuente: [DGT — Microdatos de matriculaciones](https://www.dgt.es/menusecundario/dgt-en-cifras/matraba-listados/matriculaciones-automoviles-mensual.html).

## Documentación

- [Memoria del TFM](docs/memoria.md)
- Entregas: [1. Ideas de producto](docs/entregas/01_ideas_producto.md) · [2. Datos necesarios](docs/entregas/02_datos_necesarios.md) · [3. Modelo de datos](docs/entregas/03_modelo_datos.md) · [4. Análisis y modelado](docs/entregas/04_analisis_modelado.md)
- Defensa: [guion del vídeo](docs/defensa/guion_video.md)

## Estructura

```
app.py                  aplicación Streamlit (punto de entrada del despliegue)
motoradar/              lógica compartida: datos, variables, modelos, métricas, negocio
pipeline/               scripts reproducibles fase 0 a fase 5
data/raw/               agregados de la DGT
data/external/          diccionario de provincias
data/gold/              previsiones que consume la app
models/                 modelo final (formato nativo de LightGBM) y metadatos
results/                tablas de resultados
reports/figures/        figuras de la memoria y la defensa
notebooks/              cuaderno de exploración con todas las salidas
tests/                  pruebas
docs/                   memoria, entregas y material de la defensa
```
